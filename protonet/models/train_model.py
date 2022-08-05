from tqdm.notebook import tnrange
import torch.optim as optim
import torch
import numpy as np

from utils.defs import extract_sample, USDataset, BalancedBatchSampler, save_ckp
from utils.train_utils import EarlyStopping

def train(model, optimizer, prob_type, batch_size, train_x, train_y, n_way, n_support, n_query, start_epoch, n_epoch, epoch_size, init_loss, checkpoint_path):
  """
  Trains the protonet
  Args:
      model: initialized model.
      optimizer: optimizer for training. 
      prob_type: Number of ways/classes to classify. Set to 2, 3, or 4. 
      batch_size: batch size for dataloader to load correct amount of images. 
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      start_epoch (int): current training epoch. 
      n_epoch (int): total training epoch. 
      epoch_size (int): number of episodes in each epoch. 
      init_loss (int): initial loss.  
      checkpoint_path (str): path to directory to save an intermediate model during training. 
  """

  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
  early_stopping = EarlyStopping(5, 0, init_loss)
  epoch = 0 #epochs trained so far

  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:7" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True

  train_y_num = []
  for y in train_y: 
    train_y_num.append(0 if y=="Normal" else 1 if y=="Other" else 2 if y=="Pneumonia" else 3)
  train_y_num = np.array(train_y_num)

  sample_dataset = USDataset(train_x, train_y, label_is_num=False)

  # experiment with balanced batch
  params_1 = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} 
  sample_dataset_num_label = USDataset(train_x, train_y_num, label_is_num=True)
  train_balanced_sampler = BalancedBatchSampler(sample_dataset_num_label.labels, n_classes=4, n_samples=batch_size)
  train_loader = torch.utils.data.DataLoader(sample_dataset, batch_sampler=train_balanced_sampler, **params_1)

  # while epoch < max_epoch:
  for epoch in range(start_epoch, start_epoch+n_epoch):
    running_loss = 0.0
    running_acc_sklearn = 0
    running_precision = 0.0
    running_recall = 0.0

    for episode in tnrange(epoch_size, desc="Epoch {:d} train".format(epoch+1)):
      try: 
        local_batch, local_labels = next(iter(train_loader))      
        # Transfer to GPU
        local_batch = local_batch.to(device)
        local_labels=torch.tensor(np.asarray(local_labels).astype(np.int))
        local_labels = local_labels.to(device)

        # extract sample depending on the problem type, and reorder things 
        sample = extract_sample(n_way, n_support, n_query, local_batch, local_labels, prob_type, include_COVID=True)

        # delete local batch and labels after generating samples to free memory 
        del local_batch, local_labels
        torch.cuda.empty_cache()

        optimizer.zero_grad()
        loss, output = model.set_forward_loss(sample, device=device)
        running_loss += float(output['loss']) 
        running_acc_sklearn += output['accuracy']
        running_precision += output['precision']
        running_recall += output['recall']
        loss.backward()
        optimizer.step()
      
      except StopIteration: 
        break
      
    epoch_loss = running_loss / epoch_size
    epoch_acc_sklearn = running_acc_sklearn / epoch_size
    epoch_precision = running_precision / epoch_size
    epoch_recall = running_recall / epoch_size

    early_stopping(epoch_loss)
    if early_stopping.early_stop: 
      break 

    print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc_sklearn))
    for i in range(0, len(epoch_precision)): 
      print('class {:d} -- Precision: {:.4f} Recall: {:.4f}'.format(i, epoch_precision[i], epoch_recall[i]))

    # create checkpoint variable and add important data
    checkpoint = {
        'epoch': epoch+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch_loss': epoch_loss,
    }
    # save checkpoint
    save_ckp(checkpoint, checkpoint_path)

    epoch += 1
    scheduler.step(epoch_loss)
  