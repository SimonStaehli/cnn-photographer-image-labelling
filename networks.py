import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self, n_classes):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 ) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2) 
        # nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)  
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1) 
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1) 
        # nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1  = nn.Linear(in_features= 6*6*256, out_features= 4096) 
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=n_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class NarrowAlexNet(nn.Module):
    def __init__(self, n_classes):
        super(NarrowAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 24, kernel_size= 11, stride=4, padding=0 ) 
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride= 1, padding= 2) 
        # nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=36, out_channels=54, kernel_size=3, stride= 1, padding= 1)  
        self.conv4 = nn.Conv2d(in_channels=54, out_channels=81, kernel_size=3, stride=1, padding=1) 
        self.conv5 = nn.Conv2d(in_channels=81, out_channels=122, kernel_size=3, stride=1, padding=1) 
        # nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1  = nn.Linear(in_features= 6*6*122, out_features= 3000) 
        self.fc2  = nn.Linear(in_features= 3000, out_features= 2000)
        self.fc3 = nn.Linear(in_features= 2000, out_features=n_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class FlatAlexNet(nn.Module):
    def __init__(self, n_classes):
        super(FlatAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=0 ) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2) 
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)  
        # nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1  = nn.Linear(in_features= 6*6*384, out_features=4096) 
        self.fc2  = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=n_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    