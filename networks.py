import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
    Implementation of AlexNet.
    """
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
    """
    Implementation of AlexNet with Reduced Wideness of Network.
    """
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
    """
    Implementation of AlexNet with Reduced Deepness of Network.
    """
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
    
    
class FlatAlexNetLS(nn.Module):
    """
    FlatAlexNet (LS=LowerStride) Network with reduced amount of stride which results in higher amount of Neurons in Input Layer of FC.
    """
    def __init__(self, n_classes):
        super(FlatAlexNetLS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=11, stride=3, padding=0 ) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=6, stride=2, padding= 2) 
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=4, stride=2, padding= 1)  
        self.fc1  = nn.Linear(in_features= 15*15*384, out_features=4096) 
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
    
    
class FlatAlexNetHS(nn.Module):
    """
    FlatAlexNet (HS=HigherStride) Network with reduced amount of stride which results in lower amount of Neurons in Input Layer of FC.
    """
    def __init__(self, n_classes):
        super(FlatAlexNetHS, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=11, stride=5, padding=2) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=6, stride=3, padding= 2) 
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=3, padding=1)  
        self.fc1  = nn.Linear(in_features= 3*3*384, out_features=4096) 
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


class FlatAlexNetLowKernel(nn.Module):
    """
    FlatAlexNet Low Kernel Network with reduced kernel size which results in higher amount of Neurons in Input Layer of FC.
    """
    def __init__(self, n_classes):
        super(FlatAlexNetLowKernel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=6, stride=5, padding=2) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=3) 
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=2, stride=3, padding=2)  
        self.fc1  = nn.Linear(in_features= 4*4*384, out_features=4096) 
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
    
class FlatAlexNetHighKernel(nn.Module):
    """
    FlatAlexNet high Kernel Network with increased kernel size which results in lower amount of Neurons in Input Layer of FC.
    """
    def __init__(self, n_classes):
        super(FlatAlexNetHighKernel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=17, stride=5, padding=0) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=9, stride=3, padding=0) 
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=4, stride=3, padding=0)  
        self.fc1  = nn.Linear(in_features= 1*1*384, out_features=4096) 
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
    
    
class FlatAlexNetOpt(nn.Module):
    """
    FlatAlexNet with Optimized MLP-Layer
    """
    def __init__(self, n_classes):
        super(FlatAlexNetOpt, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=6, stride=5, padding=2) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=3) 
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=2, stride=3, padding=2)  
        self.fc1  = nn.Linear(in_features= 4*4*384, out_features=2379) 
        self.fc2  = nn.Linear(in_features=2379, out_features=5592)
        self.fc3 = nn.Linear(in_features=5592 , out_features=7864)
        self.fc4 = nn.Linear(in_features=7864 , out_features=n_classes)

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
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    
class DropOutNetwork(nn.Module):
    """Implementation of the previous FlatAlexNetOpt network with additional Dropout Layer"""
    def __init__(self, n_classes, p_drop_out):
        super(DropOutNetwork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=6, stride=5, padding=2) 
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=3) 
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=2, stride=3, padding=2)  
        self.fc1  = nn.Linear(in_features= 4*4*384, out_features=2379) 
        self.fc2  = nn.Linear(in_features=2379, out_features=5592)
        self.fc3 = nn.Linear(in_features=5592 , out_features=7864)
        self.fc4 = nn.Linear(in_features=7864 , out_features=n_classes)
        self.dropout = nn.Dropout(p_drop_out)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = F.relu(self.conv3(x))
        x = self.maxpool2(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.fc4(x)
        return x
    
    
class FlatAlexNetBN(nn.Module):
    """
    FlatAlexNet with Optimized MLP-Layer and additional Batchnorm Layers before the activation layers.
    """
    def __init__(self, n_classes):
        super(FlatAlexNetBN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size=6, stride=5, padding=2) 
        self.conv1_bn=nn.BatchNorm2d(96)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=1) 
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=3) 
        self.conv2_bn=nn.BatchNorm2d(256)
        self.maxpool2 =  nn.MaxPool2d(kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=2, stride=3, padding=2)  
        self.conv3_bn=nn.BatchNorm2d(384)
        self.fc1  = nn.Linear(in_features= 4*4*384, out_features=2379) 
        self.fc1_bn=nn.BatchNorm1d(2379)
        self.fc2  = nn.Linear(in_features=2379, out_features=5592)
        self.fc2_bn=nn.BatchNorm1d(5592)
        self.fc3 = nn.Linear(in_features=5592 , out_features=7864)
        self.fc3_bn=nn.BatchNorm1d(7864)
        self.fc4 = nn.Linear(in_features=7864 , out_features=n_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.maxpool1(self.conv1_bn(x))
        x = F.relu(self.conv2(x))
        x = self.maxpool2(self.conv2_bn(x))
        x = F.relu(self.conv3(x))
        x = self.maxpool2(self.conv3_bn(x))
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(self.fc1_bn(x))
        x = self.fc2(x)
        x = F.relu(self.fc2_bn(x))
        x = self.fc3(x)
        x = F.relu(self.fc3_bn(x))
        x = self.fc4(x)
        return x
    