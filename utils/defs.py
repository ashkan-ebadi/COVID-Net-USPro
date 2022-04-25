import numpy as np
import torch
from torch.utils.data import Dataset, BatchSampler

def extract_sample(n_way, n_support, n_query, datax, datay, prob_type, include_COVID):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (torch.Tensor) dataset of images
      datay (torch.Tensor): dataset of labels 0-3
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  sample = []
  sample_class = []

  if prob_type == 2: 
    index_y = [[datay != 3], [datay == 3]] # [0] = negative labels, [1] = positive labels
  elif prob_type == 3: 
    index_y = [[datay == 0], [datay == 2], [datay==3]] # [0] = normal, [1] = Pneumonia, [2] = COVID
  else: # prob_type == 4
    if include_COVID:
      index_y = [[datay == 0], [datay == 1], [datay == 2], [datay==3]]
    else: 
      index_y = [[datay == 0], [datay == 1], [datay == 2]]
  for i in range(0,prob_type): 
    datax_cls = datax[index_y[i]] 
    perm=datax_cls[torch.randperm(datax_cls.size()[0])]
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
    sample_class.append(i) # get real class nums
  
  sample_ten = torch.stack(sample)
  sample = sample_ten.permute(0,1,4,2,3).float()
  
  return({
      'images': sample,
      'sample_class': sample_class, 
      'n_way': n_way,
      'n_support': n_support,
      'n_query': n_query
      })

def euclidean_dist(x, y):
  """
  Computes euclidean distance btw x and y
  Args:
      x (torch.Tensor): shape (n, d). n usually n_way*n_query
      y (torch.Tensor): shape (m, d). m usually n_way
  Returns:
      torch.Tensor: shape(n, m). For each query, the distances to each centroid
  """
  n = x.size(0)
  m = y.size(0)
  d = x.size(1)
  assert d == y.size(1)

  x = x.unsqueeze(1).expand(n, m, d)
  y = y.unsqueeze(0).expand(n, m, d)

  return torch.pow(x - y, 2).sum(2)

class USDataset(Dataset): 
    def __init__(self, ims, labels, label_is_num):
      'Initialization'
      self.ims = ims
      self.labels = labels
      self.label_is_num = label_is_num
    def __len__(self):
      'Denotes the total number of samples'
      return len(self.labels)
    def __getitem__(self, index):
      'Generates one sample of data'
      # Select sample
      X = self.ims[index]
      if self.label_is_num:
        y = self.labels[index]
      else:
      # Order the data in the following class: 
        y = 0 if (self.labels[index] == "Normal") else 1 if (self.labels[index] == "Other") else 2 if (self.labels[index] == "Pneumonia") else 3

      return X, y

class BalancedBatchSampler(BatchSampler):
  """
  BatchSampler - from a dataset, samples n_classes and within these classes samples n_samples.
  Returns batches of size n_classes * n_samples
  """
  def __init__(self, labels, n_classes, n_samples):
    self.labels = labels
    self.labels_set = list(set(self.labels))
    self.label_to_indices = {label: np.where(self.labels == label)[0] for label in self.labels_set}
    for l in self.labels_set:
      np.random.shuffle(self.label_to_indices[l])
    self.used_label_indices_count = {label: 0 for label in self.labels_set}
    self.count = 0
    self.n_classes = n_classes
    self.n_samples = n_samples
    self.n_dataset = len(self.labels)
    self.batch_size = self.n_samples * self.n_classes

  def __iter__(self):
    self.count = 0
    while self.count + self.batch_size < self.n_dataset:
      classes = np.random.choice(self.labels_set, self.n_classes, replace=False) 
      indices = []
      for class_ in classes: 
        indices.extend(self.label_to_indices[class_][self.used_label_indices_count[class_]:self.used_label_indices_count[class_] + self.n_samples])
        self.used_label_indices_count[class_] += self.n_samples
        if self.used_label_indices_count[class_] + self.n_samples > len(self.label_to_indices[class_]):
          np.random.shuffle(self.label_to_indices[class_])
          self.used_label_indices_count[class_] = 0
      yield indices
      self.count += self.n_classes * self.n_samples

  def __len__(self):
      return self.n_dataset // self.batch_size


def save_ckp(state, checkpoint_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    epoch_loss = checkpoint['epoch_loss']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], epoch_loss