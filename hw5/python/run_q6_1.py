import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device = torch.device('cpu')

lr = 1e-2
hidden_size = 64
batch_size = 64
max_iters = 100

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

train_x_ts = torch.from_numpy(train_x).type(torch.float32)
train_y_ts = torch.from_numpy(train_y).type(torch.LongTensor)
train_loader = DataLoader(TensorDataset(train_x_ts, train_y_ts), batch_size=batch_size, shuffle=True)

test_x_ts = torch.from_numpy(test_x).type(torch.float32)
test_y_ts = torch.from_numpy(test_y).type(torch.LongTensor)
test_loader = DataLoader(TensorDataset(test_x_ts, test_y_ts), batch_size=batch_size, shuffle=False)

valid_x_ts = torch.from_numpy(valid_x).type(torch.float32)
valid_y_ts = torch.from_numpy(valid_y).type(torch.LongTensor)
valid_loader = DataLoader(TensorDataset(valid_x_ts, valid_y_ts), batch_size=batch_size, shuffle=False)


# train_data = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

# test_data = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(), download=True)
# test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

input_size = train_x.shape[1]
num_classes = train_y.shape[1]
#Defining Neural network class
class NN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
    
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(input_size, hidden_size)  
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes).to(device)

# criterion = nn.CrossEntropyLoss()
# criterion = F.cross_entropy()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
train_loss = np.zeros(max_iters)
train_acc = np.zeros(max_iters)
valid_acc = np.zeros(max_iters)
valid_loss = np.zeros(max_iters)
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    vloss = 0
    vacc = 0
    for data in train_loader:
        inputs = torch.autograd.Variable(data[0])
        labels = torch.autograd.Variable(data[1])
        targets = torch.max(labels, 1)[1]
        # data = data.reshape(data.shape[0], -1)

        # forward
        scores = model(inputs)
        loss = F.cross_entropy(scores, targets)
        total_loss += loss.item()
        predicted = torch.max(scores, 1)[1]
        total_acc += predicted.eq(targets.data).cpu().sum().item()

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent
        optimizer.step()
    
    for dataV in valid_loader:
        inputsV = torch.autograd.Variable(dataV[0])
        labelsV = torch.autograd.Variable(dataV[1])
        targetsV = torch.max(labelsV, 1)[1]
        # data = data.reshape(data.shape[0], -1)
        scores_valid = model(inputsV)
        lossV = F.cross_entropy(scores_valid, targetsV)
        vloss += loss.item()
        predictedV = torch.max(scores_valid, 1)[1]
        vacc += predictedV.eq(targetsV.data).cpu().sum().item()
   
    train_acc[itr] = (total_acc/train_y.shape[0])*100
    train_loss[itr] = total_loss
    valid_loss[itr] = vloss
    valid_acc[itr] = (vacc/(valid_x.shape[0]))*100
    
    if (itr) % 2 == 0:
        print("Epoch: {:2d} \t Training: \t loss: {:.2f} \t acc : {:.2f}".format(itr, train_loss[itr], train_acc[itr]),"\t Validation: \t loss: {:.2f} \t acc : {:.2f}".format(valid_loss[itr], valid_acc[itr]))

n = range(max_iters)
plt.figure("Loss")
plt.plot(n, train_loss, label='training')
plt.plot(n, valid_loss, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

n = range(max_iters)
plt.figure("Accuracy")
plt.plot(n, train_acc, label='training')
plt.plot(n, valid_acc, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


print('Train accuracy: {}'.format(train_acc[-1]))
print('Validation accuracy: {}'.format(valid_acc[-1]))

torch.save(model.state_dict(), "q61_weights.pkl")