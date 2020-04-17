import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class Dataset(torch.utils.data.Dataset):
    # Create customized dataset
    def __init__(self, eegdata, ecgdata, gsrdata, ground_truth=None):
        self.X_eeg = eegdata
        self.X_ecg = ecgdata
        self.X_gsr = gsr_data
        self.y = ground_truth
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, i): 
        X_eeg = torch.Tensor(self.X_eeg[i,:, :, :].T).unsqueeze(0)
        X_ecg = torch.Tensor(self.X_ecg[i,:].T).unsqueeze(0)
        X_gsr = torch.Tensor(self.X_gsr[i,:]).unsqueeze(0)
        y = self.y[i].long()
        return X_eeg, X_ecg, X_gsr, y
    

def flatten_feature(self, x):
    num_feature = 1
    for d in x.size()[1:]:
        num_feature *= d
    return num_feature
    

class EEG_Model(torch.nn.Module):
    def __init__(self):
        super(EEG_Model, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 4), padding=(1, 1, 2))
        self.bn1 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=(3, 3, 4), padding=(1, 1, 2))
        self.bn2 = nn.BatchNorm3d(32)
        
        self.fc1 = nn.Linear(1024, 128)
        self.fc2 = nn.Linear(128, 4)

        self.pool = nn.MaxPool3d((2, 2, 4))
        self.dropout3d = nn.Dropout3d()
        self.dropout1d = nn.Dropout()

        
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bn1(x)
        x = self.dropout3d(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bn2(x)
        x = self.dropout3d(x)
        x = x.view(-1, flatten_feature(x))
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)

        return x


class ECG_Model(torch.nn.Module):
    def __init__(self):
        super(ECG_Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(2,3),padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(2,3),padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(448, 80)
        self.fc2 = nn.Linear(80, 4)

        self.maxpool = nn.MaxPool2d((2, 3))
        self.dropout1d = nn.Dropout()
        self.dropout2d = nn.Dropout2d()

        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout2d(x)
        
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.maxpool(x)
        x = self.dropout2d(x)
        
        x = x.view(-1, flatten_feature(x))
        x = F.relu(self.fc1(x))
        x = self.dropout1d(x)
        x = self.fc2(x)

        return x


class GSR_Model(torch.nn.Module):
    def __init__(self):
        super(GSR_Model, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.fc1 = nn.Linear(4096*2, 2048)
        self.fc2 = nn.Linear(2048, 512)
        self.fc3 = nn.Linear(512, 80)
        self.fc4 = nn.Linear(80, 4)
        
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.dropout(x)

        x = x.view(-1, flatten_feature(x))
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)

        return x
    

class Fusion_Model(torch.nn.Module):
    def __init__(self, eeg_model, ecg_model, gsr_model):
        super(Fusion_Model, self).__init__()
        self.eeg_model = eeg_model
        self.ecg_model = ecg_model
        self.gsr_model = gsr_model
        
        self.fc = nn.Linear(12, 4)        

        
    def forward(self, x_eeg, x_ecg, x_gsr):
        output_eeg = self.eeg_model(x_eeg)
        
        output_ecg = self.ecg_model(x_ecg)
        output_gsr = self.gsr_model(x_gsr)
        output_fusion = torch.cat((output_eeg, output_ecg, output_gsr), 1)
        x = F.relu(output_fusion)
        x = self.fc(x)
        
        return x