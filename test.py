import numpy as np 
import multiprocessing as mp
import os
import torch.optim as optim

from data.load_data import read_subdir_image, read_images
from models.ProtoNet import load_protonet_conv
from models.train_model import train
from models.test_model import test

def main(): 

    # location of train and test images 
    trainx, trainy = read_images('../../COVID-US/image/clean/train')  
    testx, testy = read_images('../../COVID-US/image/clean/test')  
    print(trainx.shape, trainy.shape, testx.shape, testy.shape)

    model = load_protonet_conv(
    x_dim=(3,224,224),
    hid_dim=64,
    z_dim=64,
    )

    optimizer = optim.Adam(model.parameters(), lr = 0.001)

    n_way = 3
    n_support = 5 # per class or support size ? make sure it is per class 
    n_query = 5

    train_x = trainx
    train_y = trainy

    max_epoch = 10
    epoch_size = 1000

    train(model, optimizer, train_x, train_y, n_way, n_support, n_query, max_epoch, epoch_size)

    # test
    n_way = 4
    n_support = 5
    n_query = 5

    test_x = testx
    test_y = testy

    test_episode = 1000

    test(model, test_x, test_y, n_way, n_support, n_query, test_episode)

if __name__ == "__main__":
    main()