import pandas as pd
import argparse
import torch
import torch.optim as optim

from data.load_data import read_images
from data.load_data_with_score import read_images_with_score
from models.ProtoNet import load_protonet_conv
from models.train_model import train
from models.test_model import test
from utils.defs import load_ckp

def main(): 
    parser = argparse.ArgumentParser(description='COVID-Net Training Script')
    parser.add_argument('--bs', default=400, type=int, help='Batch size')
    parser.add_argument('--model', default=1, type=int, help='Select model for encoder: 0=4xConv basic CNN, 1=resnet18, 2=resnet18+trainable conv layers, 3=resnet50, 4=resnet50+trainable conv layers, 5=vgg16+trainable conv layers')
    parser.add_argument('--FSlist', default="3,20,20,10,200,4,20,20,200", type=str, help='Few-shot setting list. Requires 9 arguments.')
    parser.add_argument('--classificationType', default=4, type=int, help='Type of classification type. Can be either 2 (binary), 3 (COVID, Normal, Pneumonia) or default 4 classes.')
    parser.add_argument('--LUSS', default=1, type=int, help='0=filter based on LUSS, 1 = No filtering.')
    args = parser.parse_args()

    fs_settings = [int(item) for item in args.FSlist.split(',')]
    if len(fs_settings) == 9: 
        train_n_way = fs_settings[0]
        train_n_support = fs_settings[1]
        train_n_query = fs_settings[2]

        train_epoch = fs_settings[3]
        train_episode = fs_settings[4]

        test_n_way = fs_settings[5]
        test_n_support = fs_settings[6]
        test_n_query = fs_settings[7]
        test_episode = fs_settings[8]

        print("1) train_n_way, 2) train_n_support, 3) train_n_query, 4) train_epoch, 5) train_episode 6) test_n_way 7) test_n_support 8) test_n_query 9) test_episode")
        print(args)
    else: 
        print("Please input 9 arguments in the following order: 1) train_n_way, 2) train_n_support, 3) train_n_query, 4) train_epoch, 5) train_episode \
            6) test_n_way 7) test_n_support 8) test_n_query 9) test_episode")
        return -1

    '''Load Data'''
    # on remote, with score filtering
    # train data directory: 
    train_directory = '../../COVID-US/image/clean/train' 
    test_directory = '../../COVID-US/image/clean/test'

    if args.LUSS==0: 
        # load data with certain scores 
        df_LUS = pd.read_csv('LUSS_metadata.csv')
        # modify the target score for each class here. 
        target_vids = df_LUS[(df_LUS['class']=='COVID') & (df_LUS['LUSS']==2)].id.tolist() \
            + df_LUS[(df_LUS['class']=='COVID') & (df_LUS['LUSS']==3)].id.tolist() \
            + df_LUS[(df_LUS['class']=='Normal') & (df_LUS['LUSS']==0)].id.tolist() \
            + df_LUS[(df_LUS['class']=='Other') & (df_LUS['LUSS']==0)].id.tolist() \
            + df_LUS[(df_LUS['class']=='Pneumonia') & (df_LUS['LUSS']==3)].id.tolist() 
        print("target_vids length: ", len(target_vids))
        trainx, trainy = read_images_with_score(train_directory, target_vids)  
        testx, testy = read_images_with_score(test_directory, target_vids)  
        print("Train data size: ", trainx.shape, trainy.shape, "Test data size: ", testx.shape, testy.shape)
    else: 
        trainx, trainy = read_images(train_directory)  
        testx, testy = read_images(test_directory)  
        print("Train data size: ", trainx.shape, trainy.shape, "Test data size: ", testx.shape, testy.shape)

    model_name = 'model_'+str(args.model)
    prob_type = args.classificationType

    model = load_protonet_conv(args.model, prob_type)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    train_x = trainx
    train_y = trainy
    batch_size = args.bs

    # save models after each 5 epochs during training: 
    n_epoch = 5
    checkpoint_path = f"/home/jessy.song/prototypical_network/protonet/save_checkpoints/{model_name}.pt" # remote
    epoch_iter = int(train_epoch / n_epoch)
    init_loss = 0
    start_epoch = 0 
    for i in range(0,epoch_iter): 
        if i == 0:
            train(model, optimizer, prob_type, batch_size, train_x, train_y, train_n_way, train_n_support, train_n_query, start_epoch, n_epoch, train_episode, init_loss, checkpoint_path)
        else: 
            model, optimizer, start_epoch, epoch_loss = load_ckp(checkpoint_path, model, optimizer)
            train(model, optimizer, prob_type, batch_size, train_x, train_y, train_n_way, train_n_support, train_n_query, start_epoch, n_epoch, train_episode, epoch_loss, checkpoint_path)
    
    # save final trained model. 
    model_path = f"save_models/{model_name}.pth" # remote
    torch.save(model.state_dict(), model_path) 

    # test model.
    model_saved = load_protonet_conv(args.model, prob_type)
    model_saved.load_state_dict(torch.load(model_path)) 

    test_x = testx
    test_y = testy
    test(model_saved, prob_type, batch_size, test_x, test_y, test_n_way, test_n_support, test_n_query, test_episode)

if __name__ == "__main__":
    main()