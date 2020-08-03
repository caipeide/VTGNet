from PIL import Image
import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import argparse
from model.model_vtgnet import FeatExtractor, TrajGenerator
from data_loader import load_split_train_test

interval_before = 11 # 1.5 s
interval_after = 22 # 3 s

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VTGNet training')
    parser.add_argument('--direction', default=2, type=int, choices=[0,1,2],
                    help='0: keep straight; 1: turn right; 2: turn left')
    parser.add_argument('--load_weights', default='True', type=str, choices=['True', 'False'],
                    help='load pre-trained weights on Robotcar or not')
    parser.add_argument('--batch_size', default=15, type=int,
                    help='batch size')
    args = parser.parse_args()
    command_info = args.direction

    if command_info == 0:
        train_for = 'straight'
    elif command_info == 1:
        train_for = 'right'
    else:
        train_for = 'left'

    # Device configuration
    torch.cuda.set_device(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_name = 'VTGNet'
    csv_path = './VTG-Driving-Dataset/dataset_' + train_for + '.csv'

    writer = SummaryWriter(log_dir='./VTGNet_training/log/' + train_for + '/')

    # Hyper-parameters
    learning_rate = 0.0001
    feature_size = 512
    num_epochs = 600
    batch_size = args.batch_size
    save_step = 1000
    model_path = './VTGNet_training/weights/' + train_for + '/'
    os.makedirs(model_path, exist_ok=True)

    trainloader, testloader = load_split_train_test(csv_path,valid_size=0.125,batch_size=batch_size)

    
    model1 = FeatExtractor(feature_size=feature_size).to(device)
    model2 = TrajGenerator(feature_size =feature_size).to(device)

    
    criterion = nn.MSELoss()
    params = list(model2.parameters()) + list(model1.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    if args.load_weights == 'True':
        c_1 = torch.load('./models/weights/vtgnet/' + str(command_info) + '-model1.pth', map_location=lambda storage, loc: storage)
        c_2 = torch.load('./models/weights/vtgnet/' + str(command_info) + '-model2.pth', map_location=lambda storage, loc: storage)
        model1.load_state_dict(c_1)
        model2.load_state_dict(c_2)
        

    # start training
    val_count = 0
    epoch_start = 0
    total_step = len(trainloader)
    for epoch in range(epoch_start, num_epochs):
        
        for i, sample_batched in enumerate(trainloader):
            command = sample_batched['command']
            images = sample_batched['image'].to(device)
            info_history = sample_batched['history'].to(device)
            info_future = sample_batched['future'].to(device)

            batch_size_actual = images.size(0)
            features = torch.Tensor(batch_size_actual,images.size(1),feature_size).to(device)

            for p in range(images.size(1)):
                features[:,p,:] = model1(images[:,p,:,:,:])

            outputs, logvar, attentions = model2(features, info_history)
            info_future.resize_(batch_size_actual,interval_after * 3)

            l2_loss = torch.pow((outputs - info_future.type(torch.FloatTensor).cuda()), 2)

            loss = torch.mean((torch.exp(-logvar) * l2_loss + logvar) * 0.5)

            l2_loss_ = l2_loss.reshape(batch_size_actual, 22, 3)

            l2_loss_y = l2_loss_[:,:,0]
            l2_loss_x = l2_loss_[:,:,1]
            l2_loss_v = l2_loss_[:,:,2]


            model2.zero_grad()
            model1.zero_grad()
            loss.backward()
        
            optimizer.step()

            writer.add_scalars('Train/train_loss_uncertain_'+model_name + '_'+train_for, {'Uncertain Loss': loss.item()}, len(trainloader)*epoch + i+1 )
            writer.add_scalars('Train/train_loss_l2_'+model_name + '_'+train_for, {'Uncertain Loss': torch.mean(l2_loss).item()}, len(trainloader)*epoch + i+1 )

            writer.add_scalars('Train/Y_train_loss_l2_'+model_name + '_'+train_for, {'Uncertain Loss': torch.mean(l2_loss_y).item()}, len(trainloader)*epoch + i+1 )
            writer.add_scalars('Train/X_train_loss_l2_'+model_name + '_'+train_for, {'Uncertain Loss': torch.mean(l2_loss_x).item()}, len(trainloader)*epoch + i+1 )
            writer.add_scalars('Train/V_train_loss_l2_'+model_name + '_'+train_for, {'Uncertain Loss': torch.mean(l2_loss_v).item()}, len(trainloader)*epoch + i+1 )

            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

            # save the weights
            if (len(trainloader)*epoch + i+1) % save_step == 0:
                val_count = val_count + 1
                torch.save(model2.state_dict(), os.path.join(
                    model_path, '{}-model2-{}-{}.pth'.format(command_info, val_count, epoch)))
                torch.save(model1.state_dict(), os.path.join(
                    model_path, '{}-model1-{}-{}.pth'.format(command_info, val_count, epoch)))
                torch.save(optimizer.state_dict(), os.path.join(
                    model_path, '{}-optimizer-{}-{}.pth'.format(command_info, val_count, epoch)))
                print('check point saved...')
                  
                # validate after saving the weights
                model1.eval()
                model2.eval()
                test_loss = []
                test_l2_loss = []

                test_l2_loss_y = []
                test_l2_loss_x = []
                test_l2_loss_v = []

                with torch.no_grad():
                    for _,sample_batched_val in enumerate(testloader):

                        images_val = sample_batched_val['image'].to(device)
                        info_history_val = sample_batched_val['history'].to(device)
                        info_future_val = sample_batched_val['future'].to(device)
                        
                        batch_size_actual_val = images_val.size(0)
                        features_val = torch.Tensor(batch_size_actual_val,images_val.size(1),feature_size).to(device) # 2,12,100
                        for p in range(images_val.size(1)):
                            features_val[:,p,:] = model1(images_val[:,p,:,:,:]) # 2,12,3,224,224

                        outputs_val, logvar, attentions = model2(features_val, info_history_val)                        
                        info_future_val.resize_(batch_size_actual_val,interval_after * 3)

                        l2_loss_val = torch.pow((outputs_val - info_future_val.type(torch.FloatTensor).cuda()), 2)

                        loss_val = torch.mean((torch.exp(-logvar) * l2_loss_val + logvar) * 0.5)

                        l2_loss_ = l2_loss_val.reshape(batch_size_actual, 22, 3)

                        l2_loss_y = l2_loss_[:,:,0]
                        l2_loss_x = l2_loss_[:,:,1]
                        l2_loss_v = l2_loss_[:,:,2]
                        
                        print('-----validation: ',_,'/',len(testloader),'loss_val: ',loss_val.item())

                        test_loss.append(loss_val.item())
                        test_l2_loss.append(torch.mean(l2_loss_val).item())

                        test_l2_loss_y.append(torch.mean(l2_loss_y).item())
                        test_l2_loss_x.append(torch.mean(l2_loss_x).item())
                        test_l2_loss_v.append(torch.mean(l2_loss_v).item())

                average_test_loss = np.mean(np.array(test_loss))
                average_test_l2_loss = np.mean(np.array(test_l2_loss))

                average_test_l2_loss_y = np.mean(np.array(test_l2_loss_y))
                average_test_l2_loss_x = np.mean(np.array(test_l2_loss_x))
                average_test_l2_loss_v = np.mean(np.array(test_l2_loss_v))

                print('-----Epoch [{}/{}], Validation Loss: {:.4f}'.format(epoch+1, num_epochs, average_test_loss))                    
                model1.train()
                model2.train()
                writer.add_scalars('Validation/val_loss_uncertain_'+ model_name + '_'+train_for, {'Validation Loss': average_test_loss}, val_count )
                writer.add_scalars('Validation/val_loss_l2_'+ model_name + '_'+train_for, {'Validation Loss': average_test_l2_loss}, val_count )

                writer.add_scalars('Validation/Y_val_loss_l2_'+ model_name + '_'+train_for, {'Validation Loss': average_test_l2_loss_y}, val_count )
                writer.add_scalars('Validation/X_val_loss_l2_'+ model_name + '_'+train_for, {'Validation Loss': average_test_l2_loss_x}, val_count )
                writer.add_scalars('Validation/V_val_loss_l2_'+ model_name + '_'+train_for, {'Validation Loss': average_test_l2_loss_v}, val_count )