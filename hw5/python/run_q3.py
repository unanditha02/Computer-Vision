import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

print(train_x.shape)

max_iters = 50
# pick a batch size, learning rate
batch_size = 216
learning_rate = 1e-3
hidden_size = 64
classes = 36
input_layer = 1024
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)
num_epochs = 30
params = {}

# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(input_layer, hidden_size, params, 'layer1')
initialize_weights(hidden_size, classes, params, 'output')

# with default settings, you should get loss < 150 and accuracy > 80%
for epoch in range(num_epochs):
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
            # apply gradient

            ##########################
            ##### your code here #####
            ##########################
            params['Wlayer1'] = params['Wlayer1'] - learning_rate*params['grad_Wlayer1']
            params['blayer1'] = params['blayer1'] - learning_rate*params['grad_blayer1']
            params['Woutput'] = params['Woutput'] - learning_rate*params['grad_Woutput']
            params['boutput'] = params['boutput'] - learning_rate*params['grad_boutput']
            
        if itr % 2 == 0:
            print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

# # run on validation set and report accuracy! should be above 75%
# valid_acc = None
# ##########################
# ##### your code here #####
# ##########################

# print('Validation accuracy: ',valid_acc)
# if False: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()
# import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# # Q3.3
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid

# # visualize weights here
# ##########################
# ##### your code here #####
# ##########################

# # Q3.4
# confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))

# # compute comfusion matrix here
# ##########################
# ##### your code here #####
# ##########################

# import string
# plt.imshow(confusion_matrix,interpolation='nearest')
# plt.grid(True)
# plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.show()