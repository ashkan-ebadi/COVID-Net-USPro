from tqdm.notebook import tnrange
import torch.optim as optim

from utils.defs import extract_sample

def train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size):
  """
  Trains the protonet
  Args:
      model
      optimizer
      train_x (np.array): images of training set
      train_y(np.array): labels of training set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      max_epoch (int): max epochs to train on
      epoch_size (int): episodes per epoch
  """
  #divide the learning rate by 2 at each epoch, as suggested in paper
  scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.5, last_epoch=-1)
  epoch = 0 #epochs done so far
  stop = False #status to know when to stop

  while epoch < max_epoch and not stop:
    running_loss = 0.0
    # running_acc = 0.0
    running_acc_sklearn = 0
    running_precision = 0.0
    running_recall = 0.0
    running_auc = 0.0

    for episode in tnrange(epoch_size, desc="Epoch {:d} train".format(epoch+1)):
      # training without covid data 
      sample = extract_sample(n_way, n_support, n_query, train_x, train_y, include_COVID=False)
      optimizer.zero_grad()
      loss, output = model.set_forward_loss(sample)
      running_loss += output['loss']
      # running_acc += output['acc']
      running_acc_sklearn += output['accuracy']
      running_precision += output['precision']
      running_recall += output['recall']
      # running_auc += output['auc']
      
      # print('y_true_np', output['y_true_np'])
      # print('y_pred_np', output['y_pred_np'])

      loss.backward()
      optimizer.step()
    
    epoch_loss = running_loss / epoch_size
    # epoch_acc = running_acc / epoch_size
    epoch_acc_sklearn = running_acc_sklearn / epoch_size
    epoch_precision = running_precision / epoch_size
    epoch_recall = running_recall / epoch_size
    epoch_auc = running_auc / epoch_size


    print('Epoch {:d} -- Loss: {:.4f} Acc: {:.4f}'.format(epoch+1,epoch_loss, epoch_acc_sklearn))
    for i in range(0, len(epoch_precision)): 
      print('class {:d} -- Precision: {:.4f} Recall: {:.4f}'.format(i, epoch_precision[i], epoch_recall[i]))

    epoch += 1
    scheduler.step()