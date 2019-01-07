## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I

# Namish Net from the referenced paper
class NamishNet(nn.Module):

    def __init__(self, originalDropout=True, dropout=0.3, weightsInit=None, actF=F.elu):
        super(NamishNet, self).__init__()
        self.actF = actF
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ## Namish Net Layer definition
        # Conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(4,4), stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=1, padding=0)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1,1), stride=1, padding=0)
        
        # Max-Pool (Namish Net re-uses always the same layer)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        if originalDropout:
        # Dropout layers (step size .1) as described in original paper
            self.dropout1 = nn.Dropout(p=0.1)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.3)
            self.dropout4 = nn.Dropout(p=0.4)
            self.dropout5 = nn.Dropout(p=0.5)
            self.dropout6 = nn.Dropout(p=0.6)
        else:
            self.dropout1 = nn.Dropout(p=dropout)
            self.dropout2 = nn.Dropout(p=dropout)
            self.dropout3 = nn.Dropout(p=dropout)
            self.dropout4 = nn.Dropout(p=dropout)
            self.dropout5 = nn.Dropout(p=dropout)
            self.dropout6 = nn.Dropout(p=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=43264, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=1000)
        self.fc3 = nn.Linear(in_features=1000, out_features=136)
            
        # Weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if weightsInit is None:
                    pass
                elif weightsInit == "uniform":
                    nn.init.uniform_(tensor=m.weight)
                elif weightsInit == "normal":
                    nn.init.normal_(tensor=m.weight)
                elif weightsInit == "xavier":
                    nn.init.xavier_normal_(tensor=m.weight)
                else:
                    pass
      
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        #print("x.shape / input:", x.shape)
        
        ## Conv layers according to NamishNet definition
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        #print("C1.shape:", x.shape)        
            
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout2(x)
        #print("C2.shape:", x.shape)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout3(x)
        #print("C3.shape:", x.shape)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.dropout4(x)
        #print("C4.shape:", x.shape)
                
        ## Flatten
        x = x.view(x.size(0), -1)
        #print("Flatten.shape:", x.shape)
                
        ## Fully connected layers
        x = F.elu(self.fc1(x))
        x = self.dropout5(x)
                
        x = F.relu(self.fc2(x))
        x = self.dropout6(x)
                
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
