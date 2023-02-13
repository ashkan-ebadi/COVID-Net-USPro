from tqdm.notebook import tnrange
import torch
import numpy as np

from utils.defs import extract_sample, USDataset, BalancedBatchSampler

def test_gradcam(model, test_x, test_y, n_way, n_support, n_query, test_episode):
  """
  Tests the protonet for gradcam analysis. Used for binary classification encoder. 
  Args:
      model: trained model
      test_x (np.array): images of testing set
      test_y (np.array): labels of testing set
      n_way (int): number of classes in a classification task
      n_support (int): number of labeled examples per class in the support set
      n_query (int): number of labeled examples per class in the query set
      test_episode (int): number of episodes to test on
  """
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda:7" if use_cuda else "cpu")
  
  probType = 2
  target_index = 1
  opposing_target_index = 0 

  running_loss = 0.0
  running_accuracy = 0.0
  running_precision = 0.0
  running_recall = 0.0
  
  true_pos = 0
  true_neg = 0
  false_pos = 0
  false_neg = 0
  true_pos_possibility = 0
  true_neg_possibility = 0 


  # Order the data in the following class: 
  test_y_num = []
  for y in test_y: 
    test_y_num.append(0 if y=="Normal" else 1 if y=="Other" else 2 if y=="Pneumonia" else 3)
  test_y_num = np.array(test_y_num)
  
  sample_dataset = USDataset(test_x, test_y, label_is_num=False)

  # experiment with balanced batch
  params_1 = {'num_workers': 1, 'pin_memory': True} if use_cuda else {} 
  sample_dataset_num_label = USDataset(test_x, test_y_num, label_is_num=True)
  test_balanced_sampler = BalancedBatchSampler(sample_dataset_num_label.labels, n_classes=4, n_samples=60)
  test_loader = torch.utils.data.DataLoader(sample_dataset, batch_sampler=test_balanced_sampler, **params_1)
  

  for episode in tnrange(test_episode):
    local_batch, local_labels = next(iter(test_loader))  
    # Transfer to GPU
    local_batch = local_batch.to(device)
    local_labels=np.asarray(local_labels).astype(np.int)
    local_labels = torch.tensor(local_labels).to(device)
    with torch.no_grad(): 
      sample = extract_sample(n_way, n_support, n_query, local_batch, local_labels, 2, include_COVID=True)
      loss, output = model.set_forward_loss(sample, device)
    running_loss += output['loss']
    running_accuracy += output['accuracy']
    running_precision += output['precision']
    running_recall += output['recall']

    possibility = output['probabilities']
    y_true = output['y_true_np']
    y_pred = output['y_pred_np']
    query_samples = output['x_query']
    
    # get true positives and their associated predicted possibilities
    if y_true[target_index] == y_pred[target_index]: 
      if type(true_pos) == int and type(true_pos_possibility) == int: 
        true_pos = query_samples[target_index]
        true_pos_possibility = possibility[target_index]
      else: 
        true_pos = torch.cat((true_pos, query_samples[target_index]), dim=0)
        true_pos_possibility = torch.cat((true_pos_possibility, possibility[target_index]), dim=0)
      
      # print("true pos: ",true_pos.shape)
      # print("true pos possibility: ", true_pos_possibility)
    
    if y_true[opposing_target_index] == y_pred[opposing_target_index]: 
      if type(true_neg) == int:
        true_neg = query_samples[opposing_target_index]
        true_neg_possibility = possibility[opposing_target_index]
      else: 
        true_neg = torch.cat((true_neg, query_samples[opposing_target_index]), dim=0)
        true_neg_possibility = torch.cat((true_neg_possibility, possibility[opposing_target_index]), dim=0)
      
      # print("true neg: ", true_neg.shape)
      # print("true neg possibility: ", true_neg_possibility)
    
    # get falses
    # false positives: take a normal image as a COVID image
    if y_pred[opposing_target_index] == target_index: 
      if type(false_pos) == int:
        false_pos = query_samples[opposing_target_index]
      else: 
        false_pos = torch.cat((false_pos, query_samples[opposing_target_index]), dim=0)
      # print("false pos: ", false_pos.shape)
    # false negatives: take a COVID image as normal
    elif y_pred[target_index] == opposing_target_index: 
      if type(false_neg) == int:
        false_neg = query_samples[target_index]
      else: 
        false_neg = torch.cat((false_neg, query_samples[target_index]), dim=0)
      # print("false neg: ", false_neg.shape)

  avg_loss = running_loss / test_episode
  avg_acc = running_accuracy / test_episode
  avg_precision = running_precision / test_episode
  avg_recall = running_recall / test_episode

  print('true_pos.shape', true_pos.shape) if type(true_pos) != int else print("true_pos == 0")
  print('true_neg.shape', true_neg.shape) if type(true_neg) != int else print("true_neg == 0")
  print('false_pos.shape', false_pos.shape) if type(false_pos) != int else print("false_pos == 0")
  print('false_neg.shape', false_neg.shape) if type(false_neg) != int else print("false_neg == 0")

  print('-- Loss: {:.4f} Acc: {:.4f}'.format(avg_loss, avg_acc))
  for i in range(0, len(avg_precision)): 
    print('class {:d} -- Precision: {:.4f} Recall: {:.4f}'.format(i, avg_precision[i], avg_recall[i]))

  return true_pos, true_pos_possibility, true_neg, true_neg_possibility, false_pos, false_neg