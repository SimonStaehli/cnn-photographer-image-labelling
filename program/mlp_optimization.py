import optuna
import pickle

import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utility import train_network, calculate_metrics

# Objective Function to Optmize
def objective(trial):
        
    class TestNet(nn.Module):
        
        def __init__(self, n_classes):
            super(TestNet, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=6, stride=5, padding=2) 
            self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
            self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=3) 
            self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
            self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=2, stride=3, padding=2)  
            
            self.fc_inputs =  4*4*384
            self.n_layers = trial.suggest_int('n_layers', 1, 4)
            if self.n_layers == 1:
                fc1 = trial.suggest_int('fc1', 500, 8000)
                self.fc1  = nn.Linear(in_features=self.fc_inputs, out_features=fc1) 
                self.fc_out = nn.Linear(in_features=fc1 , out_features=n_classes)
            elif self.n_layers == 2:
                fc1 = trial.suggest_int('fc1', 500, 8000)
                fc2 = trial.suggest_int('fc2', 500, 8000)
                self.fc1  = nn.Linear(in_features=self.fc_inputs, out_features=fc1) 
                self.fc2  = nn.Linear(in_features=fc1, out_features=fc2)
                self.fc_out = nn.Linear(in_features=fc2 , out_features=n_classes)
            elif self.n_layers == 3:
                fc1 = trial.suggest_int('fc1', 500, 8000)
                fc2 = trial.suggest_int('fc2', 500, 8000)
                fc3 = trial.suggest_int('fc3', 500, 8000)
                self.fc1  = nn.Linear(in_features=self.fc_inputs, out_features=fc1) 
                self.fc2  = nn.Linear(in_features=fc1, out_features=fc2)
                self.fc3  = nn.Linear(in_features=fc2, out_features=fc3)
                self.fc_out = nn.Linear(in_features=fc3 , out_features=n_classes)
            elif self.n_layers == 4:
                fc1 = trial.suggest_int('fc1', 500, 8000)
                fc2 = trial.suggest_int('fc2', 500, 8000)
                fc3 = trial.suggest_int('fc3', 500, 8000)
                fc4 = trial.suggest_int('fc4', 500, 8000)
                self.fc1  = nn.Linear(in_features=self.fc_inputs, out_features=fc1) 
                self.fc2  = nn.Linear(in_features=fc1, out_features=fc2)
                self.fc3  = nn.Linear(in_features=fc2, out_features=fc3)
                self.fc4  = nn.Linear(in_features=fc3, out_features=fc4)
                self.fc_out = nn.Linear(in_features=fc4 , out_features=n_classes)

        def forward(self,x):
            x = F.relu(self.conv1(x))
            x = self.maxpool1(x)
            x = F.relu(self.conv2(x))
            x = self.maxpool2(x)
            x = F.relu(self.conv3(x))
            x = self.maxpool2(x)
            x = x.reshape(x.shape[0], -1)
            if self.n_layers == 1:
                x = F.relu(self.fc1(x))
            elif self.n_layers==2:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
            elif self.n_layers==3:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
            elif self.n_layers==4:
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = F.relu(self.fc3(x))
                x = F.relu(self.fc4(x))
            x = self.fc_out(x)
            return x
        
    model = TestNet(n_classes=28)
    criterion_ = nn.CrossEntropyLoss()
    optimizer_ = optim.SGD(model.parameters(), lr=.004, momentum=.95)
    dtype = torch.float32
    model, loss, acc = train_network(model=model, criterion=criterion_, 
                                     optimizer=optimizer_,
                                     dataloader_train=dataloader_train, n_epochs=35)
    metrics = calculate_metrics(model=model, dl_train=dataloader_train, dl_test=dataloader_test)
        
    return metrics['accuracy'][1]


if __name__ == '__main__':
    # Config Params
    N_TRIALS = 20
    
    print(20*'=', 'Hyperparameter Tuning with Optuna - Study Started', 20*'=')
    
    # Define a Composal of image transformation used for image load
    print('INFO || Load Data')
    transform_images = Compose([Resize((227, 227)), 
                                ToTensor(),
                                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    train_data = ImageFolder("data/train", transform=transform_images)
    test_data = ImageFolder("data/test", transform=transform_images)
    
    print('INFO || Defining Dataloaders for Train and Test')
    dataloader_train = DataLoader(train_data, batch_size=150, shuffle=True, 
                                  num_workers=24, pin_memory=True)
    dataloader_test = DataLoader(test_data, batch_size=150, shuffle=True, 
                                 num_workers=24, pin_memory=True)
    
    print(20*'=', 'Starting Hyperparam-Tuning', 20*'=')
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)
    
    print('INFO || Saving best Results in Pickle-File')
    with open('best_trial.pkl', 'wb') as pkl_file:
        pickle.dump(study.best_trial, pkl_file)
    