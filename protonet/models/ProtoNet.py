from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
from torch.autograd import Variable
import torchvision.models as models  

from utils.defs import euclidean_dist

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()

  def forward(self, x):
    return x.view(x.size(0), -1)

def load_protonet_conv(model_type, prob_type):
  """
  Loads the prototypical network model
  Arg: 
    model_type: 0-basic CNN, 1-ResNet18, 2-ResNet18 with trainable last 4 conv layers, 
                3-ResNet50, 4-ResNet50 with trainable last 3 conv layers, 
                5-VGG16 with trainable last 4 conv layers. 
  Returns:
      Model (Class ProtoNet)
  """

  def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2))

  if model_type==0: 
    x_dim=(3,224,224)
    hid_dim=64
    z_dim=64
    encoder = nn.Sequential(
      conv_block(x_dim[0], hid_dim),
      conv_block(hid_dim, hid_dim),
      conv_block(hid_dim, hid_dim),
      conv_block(hid_dim, z_dim),
      Flatten()
    )
  
  elif model_type==1:
    encoder = models.resnet18(pretrained=True)
    num_ftrs = encoder.fc.in_features
    encoder.fc = nn.Linear(num_ftrs, prob_type) # reinitialize last layer to have out_feature, corresponding to num_class
    # Freeze all but last layer
    for name, param in encoder.named_parameters():
        if not 'fc' in name: 
            param.requires_grad = False
    
  elif model_type==2:
    encoder = models.resnet18(pretrained=True)
    num_ftrs = encoder.fc.in_features
    encoder.fc = nn.Linear(num_ftrs, prob_type) # reinitialize last layer to have out_feature, corresponding to num_class
    # Freeze all but last layer (4 conv layers) and fc 
    for name, param in encoder.named_parameters(): 
        if 'fc' in name or 'layer4' in name: 
            param.requires_grad = True
        else: 
            param.requires_grad = False
  
  elif model_type==3:
    encoder = models.resnet50(pretrained=True)
    num_ftrs = encoder.fc.in_features
    encoder.fc = nn.Linear(num_ftrs, prob_type) # reinitialize last layer to have out_feature, corresponding to num_class
    # Freeze all but last layer
    for name, param in encoder.named_parameters():
        if not 'fc' in name: 
            param.requires_grad = False
    
  elif model_type==4:
    encoder = models.resnet50(pretrained=True)
    num_ftrs = encoder.fc.in_features
    encoder.fc = nn.Linear(num_ftrs, prob_type) # reinitialize last layer to have out_feature, corresponding to num_class
    # Freeze all but last layer 4.2 with 3 conv layers 
    for name, param in encoder.named_parameters(): 
        if 'fc' in name or 'layer4.2' in name: 
            param.requires_grad = True
        else: 
            param.requires_grad = False

  elif model_type==5: 
    encoder = models.vgg16(pretrained=True)
    num_ftrs = encoder.classifier[-1].in_features
    encoder.classifier[-1] = nn.Linear(num_ftrs, prob_type)
    # freeze all but the last 4 conv layers past [20]
    for name, param in encoder.features[:20].named_parameters(): 
        param.requires_grad = False
  return ProtoNet(encoder)

class ProtoNet(nn.Module):
  def __init__(self, encoder):
    """
    Args:
      encoder : CNN encoding the images in sample
    """
    super(ProtoNet, self).__init__()
    self.encoder = encoder

  def set_forward_loss(self, sample, device):
    """
    Computes loss, accuracy and output for classification task
    Args:
      sample (torch.Tensor): shape (n_way, n_support+n_query, (dim)) 
      device: set to a specific CUDA when trained or remote. On local set to CPU. 
    Returns:
      torch.Tensor: shape(10), loss, y_hat, target_class, accuracy, precision, recall, 
                                y_true_np, y_pred_np, 
                                confusion_matrix, probabilities, 
    """

    sample_images = sample['images']
    n_way = sample['n_way']
    n_support = sample['n_support']
    n_query = sample['n_query']
    x_support = sample_images[:, :n_support]
    x_query = sample_images[:, n_support:]
 
    target_inds_tens = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, n_query, 1).long()
    target_inds_tens = Variable(target_inds_tens, requires_grad=False)
    # move to specified device 
    target_inds_tens = target_inds_tens.to(device)
    self.encoder = self.encoder.to(device)

    # encode images of the support and the query set
    x = torch.cat([x_support.contiguous().view(n_way * n_support, *x_support.size()[2:]),
                  x_query.contiguous().view(n_way * n_query, *x_query.size()[2:])], 0)
    
    z = self.encoder.forward(x)
    z_dim = z.size(-1) 

    # create class prototype 
    z_proto = z[:n_way*n_support].view(n_way, n_support, z_dim).mean(1)
    z_query = z[n_way*n_support:]

    # compute distances
    dists = euclidean_dist(z_query, z_proto)

    # compute probabilities
    log_p_y = F.log_softmax(-dists, dim=1).view(n_way, n_query, -1)
    loss_val = -log_p_y.gather(2, target_inds_tens).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)

    # acc_val = torch.eq(y_hat, target_inds_tens.squeeze()).float().mean()

    # compute metrics 
    y_true_np = target_inds_tens.squeeze().cpu().detach().numpy().flatten()
    y_pred_np = y_hat.cpu().detach().numpy().flatten() 

    accuracy = metrics.accuracy_score(y_true_np, y_pred_np)
    cls_labels = np.arange(0, n_way, 1, dtype=int)

    precision = metrics.precision_score(y_true_np, y_pred_np, average=None, labels=cls_labels)
    recall = metrics.recall_score(y_true_np, y_pred_np, average=None, zero_division=1, labels=cls_labels)
    cm = metrics.confusion_matrix(y_true_np, y_pred_np)
    
    return loss_val, {
      'loss': loss_val.item(),
      'y_hat': y_hat, 
      'target_class': target_inds_tens.squeeze(), 
      'y_true_np': y_true_np, 
      'y_pred_np': y_pred_np, 
      'x_query': x_query, 
      'accuracy': accuracy, 
      'recall': recall,
      'precision': precision,
      'confusion_matrix': cm, 
      'probabilities': log_p_y, 
      }