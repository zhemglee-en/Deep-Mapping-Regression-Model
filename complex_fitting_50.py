# complex neural network
# Need ComplexPyTorch

import numpy as np
import torch
import torch.nn as nn
import scipy.io as io
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import r2_score
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_normalize, complex_dropout
from matplotlib import pyplot as plt
import time

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
        
#         z = self.fc12(z)
#         #z = complex_dropout(z)
#         z = complex_relu(z)
        
#         z = self.fc13(z)
#         #z = complex_dropout(z)
#         z = complex_relu(z)
        
        
        z = self.fc8(z)

        # take the value as output
        return z

class My_loss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, y):
        loss = torch.mean(torch.pow((x.real - y.real), 2) + torch.pow((x.imag - y.imag), 2))
        return torch.sqrt(loss)

def calculate1(x, y):
    # loss = torch.mean(torch.abs(torch.sqrt(torch.pow(x.real, 2) + torch.pow(x.imag, 2)) - torch.sqrt(torch.pow(y.real, 2) + torch.pow(y.imag, 2)))/torch.sqrt(torch.pow(y.real, 2) + torch.pow(y.imag, 2)))
    loss = torch.mean(torch.div(torch.sum(torch.pow(x.real - torch.mean(y.real, dim = 1, keepdim=True), 2), dim = 1), torch.sum(torch.pow(y.real - torch.mean(y.real, dim = 1, keepdim=True), 2), dim = 1)))
    return loss

def calculate2(x, y):
    # loss = torch.mean(torch.abs(torch.sqrt(torch.pow(x.real, 2) + torch.pow(x.imag, 2)) - torch.sqrt(torch.pow(y.real, 2) + torch.pow(y.imag, 2)))/torch.sqrt(torch.pow(y.real, 2) + torch.pow(y.imag, 2)))
    loss = torch.mean(torch.div(torch.sum(torch.pow(x.imag - torch.mean(y.imag, dim = 1, keepdim=True), 2), dim = 1), torch.sum(torch.pow(y.imag - torch.mean(y.imag, dim = 1, keepdim=True), 2), dim = 1)))
    return loss

# set the parameter
batch_size = 200
learning_rate = 0.0005
Epoch = 400
momentum = 0.9
wd = 0.01
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ComplexNet(data_size = 673,
                   data_slot = 50,
                   neuron1 = 1000,
                   neuron2 = 100,
                   neuron3 = 1200).to(device)
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # use Adam to optimize
criterion = My_loss()

# load data here

dataFile1 = '/home/data.mat'
data_dic1 = io.loadmat(dataFile1)
train_data1  = data_dic1['input']
L_slot1 = data_dic1['L_slot']
train_label1  = data_dic1['output']

# concat
train_data = np.vstack((train_data1))
L_slot = np.vstack((L_slot1))
train_label = np.vstack((train_label1))

# normalization
min_max_scaler = preprocessing.MinMaxScaler()
L_slot = min_max_scaler.fit_transform(L_slot)
train_data = complex_normalize(train_data)

# dataset
train_x,test_x,L_slot_train,L_slot_test,train_y,test_y=train_test_split(train_data,L_slot,train_label,test_size=0.1,shuffle=True)
train_size = np.size(train_x, 0)
test_size = np.size(test_x, 0)
train_x = torch.from_numpy(train_x).type(torch.complex64)
train_y = torch.from_numpy(train_y).type(torch.complex64)
L_slot_train = torch.from_numpy(L_slot_train).type(torch.complex64)
L_slot_test = torch.from_numpy(L_slot_test).type(torch.complex64)
test_x = torch.from_numpy(test_x).type(torch.complex64)
test_y = torch.from_numpy(test_y).type(torch.complex64)
train_set = torch.utils.data.TensorDataset(train_x,L_slot_train,train_y)
test_set = torch.utils.data.TensorDataset(test_x,L_slot_test,test_y)
train_loader = torch.utils.data.DataLoader(train_set, batch_size= batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size= batch_size, shuffle=True)

# train_net
train_loss = []
test_loss = []
train_acc1 = []
train_acc2 = []
test_acc1 = []
test_acc2 = []
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_train = 0
    acc_train1 = 0
    acc_train2 = 0
    for batch_idx, (data1, data2, target) in enumerate(train_loader):
        data1, data2, target =data1.to(device).type(torch.complex64), data2.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        output = model(data1, data2)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_train = loss_train + loss
        acc_train1 = acc_train1 + calculate1(output, target)
        acc_train2 = acc_train2 + calculate2(output, target)
    train_acc1.append(acc_train1.cpu().detach().numpy()/train_size*batch_size)
    train_acc2.append(acc_train2.cpu().detach().numpy()/train_size*batch_size)
    train_loss.append(loss_train.cpu().detach().numpy()/train_size*batch_size)
    print('Train\t Epoch: {:3} \tLoss: {:.6f} \tAcc1: {:.6f} \tAcc2: {:.6f})'.format(
                epoch,
                loss_train.item()/train_size*batch_size,
                acc_train1.item()/train_size*batch_size,
                acc_train2.item()/train_size*batch_size)
            )

# test_net
def test(model, device, test_loader, optimizer, epoch):
    model.eval()
    loss_test = 0
    acc_test1 = 0
    acc_test2 = 0
    for batch_idx, (data1, data2, target) in enumerate(test_loader):
        data1, data2, target = data1.to(device).type(torch.complex64), data2.to(device).type(torch.complex64), target.to(device)
        output = model(data1, data2)
        loss = criterion(output, target)
        loss_test = loss_test + loss
        acc_test1 = acc_test1 + calculate1(output, target)
        acc_test2 = acc_test2 + calculate2(output, target)
    test_acc1.append(acc_test1.cpu().detach().numpy()/test_size*batch_size)
    test_acc2.append(acc_test2.cpu().detach().numpy()/test_size*batch_size)
    test_loss.append(loss_test.cpu().detach().numpy()/test_size*batch_size)
    print('Test\t Epoch: {:3} \tLoss: {:.6f} \tAcc1: {:.6f} \tAcc2: {:.6f}'.format(
                epoch,
                loss_test.item()/test_size*batch_size,
                acc_test1.item()/test_size*batch_size,
                acc_test2.item()/test_size*batch_size)
            )
    print("---------------------------------")

# # trainning process
# for epoch in range(Epoch):
#     train(model, device, train_loader, optimizer, epoch)
# #     T1 = time.clock()
#     test(model, device, test_loader, optimizer, epoch)
# #     T2 = time.clock()
# #     print('time:%s' % ((T2 - T1)*1000))


# plot_show
plt.title('Train Loss curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(train_loss, label = "$Trainloss$")
plt.plot(test_loss, label = "$Testloss$")
plt.legend()
plt.savefig("/home/lwc/1layer_complex_pytorch_2/fitting_data_50/curve_.png")
plt.show() 

# save net and train_predict
model.eval()
for batch_idx, (data1, data2, target) in enumerate(test_loader):
    data1, data2, target = data1.to(device).type(torch.complex64), data2.to(device).type(torch.complex64), target.to(device)
    output = model(data1, data2)

io.savemat('/home/output_predict.mat',{'output':output.cpu().detach().numpy(),'target':target.cpu().detach().numpy(),'train_loss':train_loss,'test_loss':test_loss,'train_acc1':train_acc1,'train_acc2':train_acc2,'test_acc1':test_acc1,'test_acc2':test_acc2})
torch.save(model.state_dict(), '/home/net_3_.pt')

print(train_size)