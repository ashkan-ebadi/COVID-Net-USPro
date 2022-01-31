import numpy as np
import torch

# function for creating sample 

def extract_sample(n_way, n_support, n_query, datax, datay, include_COVID):
  """
  Picks random sample of size n_support+n_querry, for n_way classes
  Args:
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      datax (np.array): dataset of images
      datay (np.array): dataset of labels
  Returns:
      (dict) of:
        (torch.Tensor): sample of images. Size (n_way, n_support+n_query, (dim))
        (int): n_way
        (int): n_support
        (int): n_query
  """
  if include_COVID:
    K = np.random.choice(np.unique(datay), n_way, replace=False)
  else: 
    # if include_COVID is false, take out all COVID instances
    possible_choices = [cls for cls in datay if cls != "COVID"]
    K = np.random.choice(np.unique(possible_choices), n_way, replace=False)
  sample = []
  sample_class = []
  for cls in K:
    datax_cls = datax[datay == cls]
    perm = np.random.permutation(datax_cls)
    sample_cls = perm[:(n_support+n_query)]
    sample.append(sample_cls)
    sample_class.append(cls) # get real class names 
  sample = np.array(sample)
  sample = torch.from_numpy(sample).float()
  sample = sample.permute(0,1,4,2,3)
  
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