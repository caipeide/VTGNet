from PIL import Image
import os
import torch
import torch.utils.data
import pandas as pd
import numpy as np
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
from model.model_vtgnet import FeatExtractor, TrajGenerator
import cv2
from scipy import interpolate

save_path = './test_results/'
os.makedirs(save_path, exist_ok=True)


interval_before = 11 # 1.5 s
interval_after = 22 # 3 s
feature_size = 512

# Device configuration
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model1_s = FeatExtractor(feature_size=feature_size).to(device) # keep straight
model2_s = TrajGenerator(feature_size=feature_size).to(device)
model1_r = FeatExtractor(feature_size=feature_size).to(device) # turn right
model2_r = TrajGenerator(feature_size=feature_size).to(device)
model1_l = FeatExtractor(feature_size=feature_size).to(device) # turn left
model2_l = TrajGenerator(feature_size=feature_size).to(device)

model_path = './model/weights/'
model1_l.load_state_dict(torch.load(model_path + '2-model1.pth', map_location=lambda storage, loc: storage)) # model1 -- Feature Extractor
model2_l.load_state_dict(torch.load(model_path + '2-model2.pth', map_location=lambda storage, loc: storage)) # model2 -- Trajectory Generator

model1_r.load_state_dict(torch.load(model_path + '1-model1.pth', map_location=lambda storage, loc: storage))
model2_r.load_state_dict(torch.load(model_path + '1-model2.pth', map_location=lambda storage, loc: storage))

model1_s.load_state_dict(torch.load(model_path + '0-model1.pth', map_location=lambda storage, loc: storage))
model2_s.load_state_dict(torch.load(model_path + '0-model2.pth', map_location=lambda storage, loc: storage))

model1_s.eval()
model2_s.eval()
model1_r.eval()
model2_r.eval()
model1_l.eval()
model2_l.eval()

# camera parameters
fx = 983.044006
fy = 983.044006
cx = 6.095593000000e+02
cy = 1.728540000000e+02

csv_path = 'test_data/vtgnet/data_reference.csv'
data = pd.read_csv(csv_path, header=None)

results = []

with torch.no_grad():
    for idx in range(len(data)):
        print('{}/{}'.format(idx+1,data.shape[0]))
        
        # get command
        command = data.iloc[idx,0]
        # history info
        info_st_index = 1 + 12
        info_st_index_2 = (info_st_index + 4*(interval_before+1))
        info_history = data.iloc[idx, info_st_index:info_st_index_2].to_numpy().reshape(-1,4)
        info_history_net = info_history[:,0:3]
        info_history_net = torch.from_numpy(info_history_net.astype('float')).unsqueeze(0).to(device)
        info_future = data.iloc[idx, info_st_index_2:].to_numpy().reshape(-1,4)

        local_x_history = info_history[:,0]
        local_y_history = info_history[:,1]
        spd_history = info_history[:,2]
        yaw_history = info_history[:,3]

        local_x_future = info_future[:,0]
        local_y_future = info_future[:,1]
        spd_future = info_future[:,2]
        yaw_future = info_future[:,3]

        image = []
        for k in range(1, 1 + interval_before+1):
            image.append( transforms.Resize((224,224))(Image.open(data.iloc[idx,k])) )
        image = torch.stack( [transforms.ToTensor()(image[k]) for k in range(len(image))], dim=0 )
        for ii in range(image.size(0)):
            image[ii,:,:,:] = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(image[ii,:,:,:])
        image = image.unsqueeze(0).to(device)

        # keep straight
        if command == 0:
            features = torch.Tensor(1,12,feature_size).to(device)

            for p in range(12):
                features[:,p,:] = model1_s(image[:,p,:,:,:])

            outputs, logvar, attentions = model2_s(features, info_history_net)
        # turn right
        if command == 1 :
            features = torch.Tensor(1,12,feature_size).to(device)

            for p in range(12):
                features[:,p,:] = model1_r(image[:,p,:,:,:])
            outputs, logvar, attentions = model2_r(features, info_history_net) 
        # turn left
        if command == 2:
            features = torch.Tensor(1,12,feature_size).to(device)

            for p in range(12):
                features[:,p,:] = model1_l(image[:,p,:,:,:])
            outputs, logvar, attentions = model2_l(features, info_history_net)

        var = logvar.exp().reshape(-1,3).cpu().detach().numpy()
        attentions = attentions.squeeze(0).cpu().detach().numpy()

        planned = (outputs.reshape(-1,3).cpu().detach().numpy())
        local_x_planned = planned[:,0]
        local_y_planned = planned[:,1]
        spd_planned = planned[:,2]

        local_x_planned_var = var[:,0]
        local_y_planned_var = var[:,1]
        spd_planned_var = var[:,2]

        img_center = cv2.imread(data.iloc[idx,1 + interval_before])

        local_x_planned_sigma = np.sqrt(local_x_planned_var)
        local_y_planned_sigma = np.sqrt(local_y_planned_var)
        spd_planned_sigma = np.sqrt(spd_planned_var)

        xx = np.arange(1,22+0.001,1)
        f_x_planned = interpolate.interp1d(xx, local_x_planned, kind='cubic')
        f_y_planned = interpolate.interp1d(xx, local_y_planned, kind='cubic')

        f_x_gt = interpolate.interp1d(xx, local_x_future, kind='cubic')
        f_y_gt = interpolate.interp1d(xx, local_y_future, kind='cubic')

        f_x_planned_sigma = interpolate.interp1d(xx, local_x_planned_sigma, kind='cubic')
        f_y_planned_sigma = interpolate.interp1d(xx, local_y_planned_sigma, kind='cubic')

        x_new = np.arange(1, 22, 0.05)

        local_x_planned = f_x_planned(x_new)
        local_y_planned = f_y_planned(x_new)

        local_x_planned_sigma = f_x_planned_sigma(x_new)
        local_y_planned_sigma = f_y_planned_sigma(x_new)

        local_x_future = f_x_gt(x_new)
        local_y_future = f_y_gt(x_new)

        # draw the trajectory: ground truth
        X = local_x_future + 0.12
        Y = 1.52
        Z = local_y_future + (0.17+1.55)

        filtered_idx = np.where(Z>0)
        Z = Z[filtered_idx]
        X = X[filtered_idx]

        X = X/Z
        Y = Y/Z
        us = (fx*X + cx).astype(int)
        vs = (fy*Y + cy).astype(int)

        vertices_gt = np.stack((us, vs)).transpose().reshape(-1,2)

        # draw the trajectory: generated values
        X0 = local_x_planned + 0.12
        Y0 = 1.52
        Z0 = local_y_planned + (0.17+1.55)

        filtered_idx = np.where(Z0>0)
        
        Z0 = Z0[filtered_idx]
        X0 = X0[filtered_idx]

        # draw the uncertainty area.
        if command == 0 or idx == 19 or idx == 2:
            X1 = local_x_planned + 0.12 - local_x_planned_sigma
            Y1 = 1.52
            Z1 = local_y_planned + (0.17+1.55)

            filtered_idx = np.where(Z1>0)
            Z1 = Z1[filtered_idx]
            X1 = X1[filtered_idx]

            X2 = local_x_planned + 0.12 + local_x_planned_sigma
            Y2 = 1.52
            Z2 = local_y_planned + (0.17+1.55)

            filtered_idx = np.where(Z2>0)
            Z2 = Z2[filtered_idx]
            X2 = X2[filtered_idx]
        else:
            X1 = local_x_planned + 0.12 
            Y1 = 1.52
            Z1 = local_y_planned + (0.17+1.55) - local_y_planned_sigma

            filtered_idx = np.where(Z1>0)
            Z1 = Z1[filtered_idx]
            X1 = X1[filtered_idx]

            X2 = local_x_planned + 0.12
            Y2 = 1.52
            Z2 = local_y_planned + (0.17+1.55) + local_y_planned_sigma

            filtered_idx = np.where(Z2>0)
            Z2 = Z2[filtered_idx]
            X2 = X2[filtered_idx]

        X0 = X0/Z0
        Y0 = Y0/Z0
        us0 = (fx*X0 + cx).astype(int)
        vs0 = (fy*Y0 + cy).astype(int)

        X1 = X1/Z1
        Y1 = Y1/Z1
        us1 = (fx*X1 + cx).astype(int)
        vs1 = (fy*Y1 + cy).astype(int)

        X2 = X2/Z2
        Y2 = Y2/Z2
        us2 = (fx*X2 + cx).astype(int)
        vs2 = (fy*Y2 + cy).astype(int)
        
        vertices1 = np.stack((us1, vs1)).transpose().reshape(-1,2)
        vertices0 = np.stack((us0, vs0)).transpose().reshape(-1,2)
        vertices2 = np.stack((us2, vs2)).transpose().reshape(-1,2)

        overlay = img_center.copy()
        cv2.fillPoly(overlay, [np.concatenate((vertices1, vertices2[::-1]), axis=0)], (0,0,255))

        alpha = 0.6
        img_center = cv2.addWeighted(overlay, alpha, img_center, 1-alpha, 0 )

        cv2.polylines(img_center, [vertices_gt], False, (0,255,0),3)
        cv2.polylines(img_center, [vertices0], False, (0,0,0),3)

        results.append(img_center)

row1 = np.hstack(results[:4])
row2 = np.hstack(results[4:8])
row3 = np.hstack(results[8:12])
row4 = np.hstack(results[12:16])
row5 = np.hstack(results[16:20])

cv2.imwrite('./test_results/' + 'vtgnet_result.jpg', np.vstack([row1, row2, row3, row4, row5]))
print('vtgnet_result.jpg saved in ./test_results/')
        
    
