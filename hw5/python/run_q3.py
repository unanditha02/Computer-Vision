import numpy as np
import scipy.io
from nn import *
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle 
from sklearn.metrics import confusion_matrix
import string

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')
train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']


print(train_x.shape)
print(valid_x.shape)
max_iters = 100
# pick a batch size, learning rate
batch_size = 32
learning_rate = 3e-3
hidden_size = 64
classes = 36
input_layer = 1024
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(input_layer, hidden_size, params, 'layer1')
initialize_weights(hidden_size, classes, params, 'output')
train_loss = np.zeros(max_iters)
train_acc = np.zeros(max_iters)
valid_acc = np.zeros(max_iters)
valid_loss = np.zeros(max_iters)
# with default settings, you should get loss < 150 and accuracy > 80%
# for epoch in range(num_epochs):
for itr in range(max_iters):
        total_loss = 0
        total_acc = 0
        
        for xb,yb in batches:
            
            # training loop can be exactly the same as q2!
            ##########################
            ##### your code here #####
            ##########################

            # forward
            h1 = forward(xb, params, 'layer1')
            probsb = forward(h1,params,'output',softmax)
            # loss
            loss, acc = compute_loss_and_acc(yb, probsb)
            # be sure to add loss and accuracy to epoch totals 
            total_loss += loss
            total_acc += acc
            total_acc = total_acc/2 
            # backward
            delta1b = probsb
            yb_idx = np.argmax(yb, axis=1)
            delta1b[np.arange(probsb.shape[0]),yb_idx] -= 1
            
            delta2b = backwards(delta1b,params,'output',linear_deriv)
            backwards(delta2b,params,'layer1',sigmoid_deriv)
            
            ##########################
            ##### your code here #####
            ##########################
            params['Wlayer1'] = params['Wlayer1'] - learning_rate*params['grad_Wlayer1']
            params['blayer1'] = params['blayer1'] - learning_rate*params['grad_blayer1']
            params['Woutput'] = params['Woutput'] - learning_rate*params['grad_Woutput']
            params['boutput'] = params['boutput'] - learning_rate*params['grad_boutput']
        # forward
        h1 = forward(valid_x, params, 'layer1')
        probs = forward(h1,params,'output',softmax)
        # loss
        vloss, vacc = compute_loss_and_acc(valid_y, probs)
        train_loss[itr] = total_loss/(train_x.shape[0])
        train_acc[itr] = total_acc
        valid_loss[itr] = vloss/(valid_x.shape[0])
        valid_acc[itr] = vacc
        if itr % 2 == 0:
                print("Epoch: {:2d} \t Training: \t loss: {:.2f} \t acc : {:.2f}".format(itr, train_loss[itr], train_acc[itr]),"\t Validation: \t loss: {:.2f} \t acc : {:.2f}".format(valid_loss[itr], valid_acc[itr]))        
        

n = range(max_iters)
plt.figure("Loss")
plt.plot(n, train_loss, label='training')
plt.plot(n, valid_loss, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.figure("Accuracy")
plt.plot(n, train_acc, label='training')
plt.plot(n, valid_acc, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

##############################################################################


# run on validation set and report accuracy! should be above 75%

#########################
#### your code here #####
#########################

# if False: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()
# import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
##########################
##### your code here #####
##########################
params_init = {}
initialize_weights(input_layer, hidden_size, params_init, 'layer1')
initialize_weights(hidden_size, classes, params_init, 'output')

with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)

initW = params_init['Wlayer1'].reshape((32,32,64))
trainedW = saved_params['Wlayer1'].reshape((32,32,64))
initW_arr = []
trainedW_arr = []
for i in range(initW.shape[2]):
    initW_arr.append(initW[:,:,i])
    trainedW_arr.append(trainedW[:,:,i])

fig = plt.figure("Initialized Weights", figsize=(20., 20.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(8, 8),
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, initW_arr):
    # Iterating over the grid returns the Axes.
    ax.imshow(im)

fig = plt.figure("Trained Weights", figsize=(20., 20.))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(8, 8),
                 axes_pad=0.1,  # pad between axes in inch.
                 )

for ax, im in zip(grid, trainedW_arr):
    ax.imshow(im)

plt.show()

# Q3.4
cf = np.zeros((train_y.shape[1],train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################
with open('q3_weights.pickle', 'rb') as handle:
    saved_params = pickle.load(handle)
totalValues = len(test_x)
h1 = forward(test_x, saved_params, 'layer1')
probs = forward(h1, saved_params, 'output',softmax)
label = np.zeros((totalValues,1))
pred = np.zeros((totalValues,1))
for i in range(totalValues):
    cf[np.argmax(probs[i,:]),np.argmax(test_y[i,:])] += 1
    label[i] = np.argmax(test_y[i,:])
    pred[i] = np.argmax(probs[i,:])

cf_fn = confusion_matrix(pred, label)
plt.imshow(cf_fn,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()