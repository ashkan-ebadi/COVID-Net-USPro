from tqdm.notebook import tnrange
import torch
import numpy as np

from utils.defs import extract_sample, USDataset, BalancedBatchSampler

def test(model, prob_type, batch_size, test_x, test_y, n_way, n_support, n_query, test_episode):
  """
  Tests the protonet
  Args:
      model: trained model
      prob_type: Number of ways/classes to classify. Set to 2, 3, or 4. 
      batch_size: batch size for dataloader to load correct amount of images. 
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
  """
  # CUDA for PyTorch
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:7" if use_cuda else "cpu")
  torch.backends.cudnn.benchmark = True

  # Order the data in the following class: 
  test_y_num = []
  for y in test_y: 
    test_y_num.append(0 if y=="Normal" else 1 if y=="Other" else 2 if y=="Pneumonia" else 3)
  test_y_num = np.array(test_y_num)
  sample_dataset = USDataset(test_x, test_y, label_is_num=False)

  # experiment with balanced batch
  params_1 = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} 
  sample_dataset_num_label = USDataset(test_x, test_y_num, label_is_num=True)
  test_balanced_sampler = BalancedBatchSampler(sample_dataset_num_label.labels, n_classes=4, n_samples=batch_size)
  test_loader = torch.utils.data.DataLoader(sample_dataset, batch_sampler=test_balanced_sampler, **params_1)
  
  # initialize metrics to keep track of 
  running_loss = 0.0
  running_accuracy = 0.0
  running_precision = 0.0
  running_recall = 0.0
  cm_size = prob_type*prob_type
  running_cm = np.zeros(cm_size).reshape((prob_type,prob_type))

  for episode in tnrange(test_episode): 
    local_batch, local_labels = next(iter(test_loader))  
      # Transfer to GPU
    local_batch = local_batch.to(device)
    local_labels=np.asarray(local_labels).astype(np.int)
    local_labels = torch.tensor(local_labels).to(device)

    with torch.no_grad(): 
      #extract samples with covid data
      sample = extract_sample(n_way, n_support, n_query, local_batch, local_labels, prob_type, include_COVID=True)
      loss, output = model.set_forward_loss(sample, device=device)
    
    running_loss += float(output['loss'])
    running_accuracy += output['accuracy']
    running_precision += output['precision']
    running_recall += output['recall']
    running_cm = np.add(running_cm, output['confusion_matrix'] )

  avg_loss = running_loss / test_episode
  avg_acc = running_accuracy / test_episode
  avg_precision = running_precision / test_episode
  avg_recall = running_recall / test_episode
  total_cm = running_cm

  print('-- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
  for i in range(0, len(avg_precision)): 
    print('class {:d} -- Precision: {:.4f} Recall: {:.4f}'.format(i, avg_precision[i], avg_recall[i]))
  
  print(total_cm)