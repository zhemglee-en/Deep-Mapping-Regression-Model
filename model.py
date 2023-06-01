# complex neural network
# Need ComplexPyTorch

import torch
import torch.nn as nn
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_normalize, complex_dropout

class ComplexNet(nn.Module):
    
    def __init__(self, data_size, data_slot, neuron1, neuron2, neuron3):
        super(ComplexNet, self).__init__()
        self.fc1 = ComplexLinear(data_size, neuron1)
        self.fc2 = ComplexLinear(neuron1, neuron1)
        self.fc9 = ComplexLinear(neuron1, neuron1)

        self.fc3 = ComplexLinear(data_slot, neuron2)
        self.fc4 = ComplexLinear(neuron2, neuron2)
        self.fc10 = ComplexLinear(neuron2, neuron2)

        self.fc5 = ComplexLinear(neuron1+neuron2, neuron3)
        self.fc6 = ComplexLinear(neuron3, neuron3)
        self.fc7 = ComplexLinear(neuron3, neuron3)
        self.fc11 = ComplexLinear(neuron3, neuron3)
        self.fc12 = ComplexLinear(neuron3, neuron3)
        self.fc13 = ComplexLinear(neuron3, neuron3)
        self.fc8 = ComplexLinear(neuron3, data_size)
        
             
    def forward(self,x,y):
        # input-fc-relu-fc-relu for 2 parts
        x = self.fc1(x)
        #x = complex_dropout(x)
        x = complex_relu(x) 
        '''
        x = self.fc2(x)
        #x = complex_dropout(x)
        x = complex_relu(x)
        x = self.fc9(x)
        #x = complex_dropout(x)
        x = complex_relu(x)
        '''
        
        y = self.fc3(y)
        #y = complex_dropout(y)
        y = complex_relu(y)
        '''
        y = self.fc4(y)
        #y = complex_dropout(y)
        y = complex_relu(y)
        y = self.fc10(y)
        #y = complex_dropout(y)
        y = complex_relu(y)
        '''
        
        # concat 2 parts
        z = torch.cat((x,y), dim = 1)

        # fc-relu-fc-relu-fc-relu-fc-output layers
        z = self.fc5(z)
        #z = complex_dropout(z)
        z = complex_relu(z)
        
        z = self.fc6(z)
        #z = complex_dropout(z)
        z = complex_relu(z)
        
        z = self.fc7(z)
        #z = complex_dropout(z)
        z = complex_relu(z)
        
        z = self.fc11(z)
        #z = complex_dropout(z)
        z = complex_relu(z)
        z = self.fc12(z)
        #z = complex_dropout(z)
        z = complex_relu(z)
        
        
        z = self.fc8(z)

        # take the value as output
        return z
