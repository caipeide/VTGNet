from PIL import Image
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import SubsetRandomSampler

interval_before = 11
interval_after = 22

class myDataset(Dataset):

    def __init__(self, csv_path, transform=None):

        self.Images_frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.Images_frame)

    def __getitem__(self, idx):
        image = []
        command = self.Images_frame.iloc[idx,0]
        
        for k in range(3, 3 + interval_before+1):
            image.append( Image.open('./VTG-Driving-Dataset/' + self.Images_frame.iloc[idx,k]) )
        
        # history info
        info_st_index = 3 + interval_after + interval_before + 1 #34
        info_st_index_2 = (info_st_index + 3*(interval_before+1)) # 34+48
        info_history = self.Images_frame.iloc[idx, info_st_index:info_st_index_2].values
        info_history = info_history.astype('float').reshape(-1, 3)
        
        # future info
        info_future = self.Images_frame.iloc[idx, info_st_index_2:].values
        info_future = info_future.astype('float').reshape(-1, 3)
        
        sample = {'command': command, 'image': image, 'history': info_history, 'future':info_future}

        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):

    def __call__(self, sample):
        command, image, info_history, info_future = sample['command'], sample['image'], sample['history'], sample['future']
        for k in range(len(image)):
            image[k] = transforms.Resize((224,224))(image[k])
        return {'command': command,
                'image': torch.stack( [transforms.ToTensor()(image[k]) for k in range(len(image))], dim=0 ),
                'history': torch.from_numpy(info_history),
                'future': torch.from_numpy(info_future)}

class Normalize(object):

    def __call__(self, sample):
        command, image, info_history, info_future = sample['command'], sample['image'], sample['history'], sample['future']
        
        for i in range(image.size(0)):
            image[i,:,:,:] = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])(image[i,:,:,:]) 

        return {'command':command, 'image': image, 'history': info_history, 'future': info_future}

def load_split_train_test(datadir, batch_size=64, num_workers=4, valid_size = .125, shuffle_train=True, shuffle_val = False):
    train_transforms = transforms.Compose([ToTensor(), Normalize()])
    test_transforms = transforms.Compose([ToTensor(), Normalize()])

    train_data = myDataset(datadir,transform=train_transforms)
    test_data = myDataset(datadir,transform=test_transforms)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    np.random.seed(123)
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    trainloader = DataLoader(train_data,
                sampler=train_sampler, batch_size=batch_size, num_workers=num_workers,drop_last=True)
    testloader =  DataLoader(test_data,
                sampler=test_sampler, batch_size=batch_size, num_workers=num_workers,drop_last=True)
    return trainloader, testloader