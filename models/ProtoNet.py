from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from utils.defs import euclidean_dist

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

def load_protonet_conv(**kwargs):
  """
  Loads the prototypical network model
  Arg:
      x_dim (tuple): dimension of input image
      hid_dim (int): dimension of hidden layers in conv blocks
      z_dim (int): dimension of embedded image
  Returns:
      Model (Class ProtoNet)
  """
  x_dim = kwargs['x_dim']
  hid_dim = kwargs['hid_dim']
  z_dim = kwargs['z_dim']

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
    
  encoder = nn.Sequential(
    conv_block(x_dim[0], hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, hid_dim),
    conv_block(hid_dim, z_dim),
    Flatten()
  )
    
  return ProtoNet(encoder)

class ProtoNet(nn.Module):
  def __init__(self, encoder):
    """
    Args:
    encoder : CNN encoding the images in sample
    n_way (int): number of classes in a classification task
    n_support (int): number of labeled examples per class in the support set
    n_query (int): number of labeled examples per class in the query set
    """
    super(ProtoNet, self).__init__()
    # self.encoder = encoder # no cuda
    self.encoder = encoder.cuda() # on remote

  def set_forward_loss(self, sample):
    """
    Computes loss, accuracy and output for classification task
    Args:
      sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
    Returns:
      torch.Tensor: shape(2), loss, accuracy and y_hat
    """
    # sample_images = sample['images'] # on local no cuda
    sample_images = sample['images'].cuda() # on remote 
    sample_class = sample['sample_class']

    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']

    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]

    #target indices are 0 ... n_way-1
    # Order the target indices in order of: 0=Normal, 1=Other, 2=Pneumonia, 3=COVID 
    target_inds = np.zeros((n_way,n_query))
    for i,cls in enumerate(sample_class): 
      target_inds[i, :] = 0 if (cls == "Normal") else 1 if (cls == "Other") else 2 if (cls == "Pneumonia") else 3
    
    # target_ind.reshape((target_ind.shape[0], target_ind.shape[1], 1))
    target_inds_tens = torch.from_numpy(target_inds.astype(int))
    target_inds_tens = torch.reshape(target_inds_tens, (target_inds.shape[0], target_inds.shape[1], 1))
    # print("target_ind_tens" , target_inds_tens)
    # print(target_inds_tens.shape)  

    target_inds_tens = target_inds_tens.cuda() #on remote
      
    # Original, class non-controlled version 
    # target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    # target_inds = Variable(target_inds, requires_grad=False)
    # print(target_inds)
    # print(target_inds.shape) # 3,5,1 i.e. n_way, n_query, 1
    # target_inds = target_inds.cuda()

    #encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                  x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)

    z = self.encoder.forward(x)
    z_dim = z.size(-1) #usually 64

    #create class prototype 
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
    z_query = z[n_way*n_support:]

    #compute distances
    dists = euclidean_dist(z_query, z_proto)

    #compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
    loss_val = -log_p_y.gather(2, target_inds_tens).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)

    # acc_val = torch.eq(y_hat, target_inds_tens.squeeze()).float().mean()

    # compute metrics 
    # y_true = target_inds.squeeze().shape()
    y_true_np = target_inds_tens.squeeze().cpu().detach().numpy().flatten()
    y_pred_np = y_hat.cpu().detach().numpy().flatten() 

    accuracy = metrics.accuracy_score(y_true_np, y_pred_np)
    cls_labels = np.arange(0, n_way, 1, dtype=int)

    precision = metrics.precision_score(y_true_np, y_pred_np, average=None, zero_division=1, labels=cls_labels)
    recall = metrics.recall_score(y_true_np, y_pred_np, average=None, zero_division=1, labels=cls_labels)

    # need to define 
    # auc = metrics.roc_auc_score(y_true_np, y_prob, average=None, multi_class="ovr", labels=[0,1,2])

    return loss_val, {
      'loss': loss_val.item(),
      # 'acc': acc_val.item(),
      'y_hat': y_hat, 
      'target_class': target_inds_tens.squeeze(), 
      'y_true_np': y_true_np, 
      'y_pred_np': y_pred_np, 
      'accuracy': accuracy, 
      'recall': recall,
      'precision': precision,
      # 'auc': auc
      }