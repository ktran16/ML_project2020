import torch
import torch.nn as nn
import torch.nn.functional as F


# Simple NN networks with no hidden layer
# One Layer

# linear classifier
class OneLayer(nn.Module):
    def __init__(self, input_size, num_classes):
        super(OneLayer, self).__init__()
        self.input_size = input_size
        self.layer1 = nn.Linear(self.input_size, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        out = self.layer1(x)
        return out


# 3-layer NN with no regularization
class unRegNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(unRegNN, self).__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        out = self.layer3(x)
        return out


# 3-layer regularized with Dropout
class RegNN_dropout(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RegNN_dropout, self).__init__()
        self.clf = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        out = self.clf(x)
        return out

  
# function to define the conv layer with 3 main component: conv layer, activation function (ReLu) and MaxPooling
def conv_relu_maxp(input, output, ks):
    return [nn.Conv2d(input, output, kernel_size=ks, stride=1, padding=int((ks-1)/2), bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)]

# function to define fully connected layer with regularization method of randomly dropout
def dropout_linear_relu(dim_in, dim_out, drop_p):
    return [nn.Dropout(drop_p),
            nn.Linear(dim_in, dim_out),
            nn.ReLU(inplace=True)]

def linear_relu(dim_in, dim_out):
    return [nn.Linear(dim_in, dim_out),
           nn.ReLU(inplace=True)]


# CNN
# Simple Unregularized CNN 
class Simple_unReg_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple_unReg_CNN, self).__init__()
        self.features = nn.Sequential(*conv_relu_maxp(1, 16, 5),
                                    *conv_relu_maxp(16, 32, 5),
                                    *conv_relu_maxp(32, 64, 5))

        self.classifier = nn.Sequential(*linear_relu(3*3*64, 256),
                                        *linear_relu(256, 128),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.features(x)
        # flattening 
        x = x.view(x.size()[0], -1)
        out = self.classifier(x)
        return out


# Simple Regularized CNN 
class Simple_Reg_CNN(nn.Module):
    def __init__(self, num_classes):
        super(Simple_Reg_CNN, self).__init__()
        self.features = nn.Sequential(*conv_relu_maxp(1, 16, 5),
                                    *conv_relu_maxp(16, 32, 5),
                                    *conv_relu_maxp(32, 64, 5))

        self.classifier = nn.Sequential(*dropout_linear_relu(3*3*64, 256, 0.5),
                                        *dropout_linear_relu(256, 128, 0.5),
                                        nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.features(x)
        # flattening 
        x = x.view(x.size()[0], -1)
        out = self.classifier(x)
        return out

