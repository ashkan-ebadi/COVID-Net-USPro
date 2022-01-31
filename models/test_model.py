from tqdm.notebook import tnrange
import torch.optim as optim

from utils.defs import extract_sample

def test(model, test_x, test_y, n_way, n_support, n_query, test_episode):
  """
  Tests the protonet
  Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
  """
  running_loss = 0.0
  running_accuracy = 0.0
  running_precision = 0.0
  running_recall = 0.0
  for episode in tnrange(test_episode): 
    sample = extract_sample(n_way, n_support, n_query, test_x, test_y, include_COVID=True)
    loss, output = model.set_forward_loss(sample)
    running_loss += output['loss']
    running_accuracy += output['accuracy']
    running_precision += output['precision']
    running_recall += output['recall']

  avg_loss = running_loss / test_episode
  avg_acc = running_accuracy / test_episode
  avg_precision = running_precision / test_episode
  avg_recall = running_recall / test_episode

  print('-- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
  for i in range(0, len(avg_precision)): 
    print('class {:d} -- Precision: {:.4f} Recall: {:.4f}'.format(i, avg_precision[i], avg_recall[i]))