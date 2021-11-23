import numpy as np
import scipy.io
from nn import *
from collections import Counter
import pickle 
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)
input_layer = 1024
output_layer = 1024
params = Counter()

# Q5.1 & Q5.2
# initialize layers here
##########################
##### your code here #####
##########################
initialize_weights(input_layer, hidden_size, params, 'layer1')
initialize_weights(hidden_size, hidden_size, params, 'layer2')
initialize_weights(hidden_size, hidden_size, params, 'layer3')
initialize_weights(hidden_size, output_layer, params, 'output')
params['m_Wlayer1'] = np.zeros(shape=(input_layer, hidden_size))
params['m_Wlayer2'] = np.zeros(shape=(hidden_size, hidden_size))
params['m_Wlayer3'] = np.zeros(shape=(hidden_size, hidden_size))
params['m_Woutput'] = np.zeros(shape=(hidden_size, output_layer))
params['m_blayer1'] = np.zeros(shape=(hidden_size))
params['m_blayer2'] = np.zeros(shape=(hidden_size))
params['m_blayer3'] = np.zeros(shape=(hidden_size))
params['m_boutput'] = np.zeros(shape=(output_layer))
train_loss = np.zeros(max_iters)
valid_loss = np.zeros(max_iters)
# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb, params, 'layer1',relu)
        h2 = forward(h1,params,'layer2',relu)
        h3 = forward(h2, params, 'layer3',relu)
        probsb = forward(h3,params,'output',sigmoid)
        
        loss = np.sum(np.square((probsb - xb)))
        total_loss += loss
        delta1b = 2*(probsb - xb)
        delta2b = backwards(delta1b,params,'output',sigmoid_deriv)
        delta3b = backwards(delta2b,params,'layer3',relu_deriv)
        delta4b = backwards(delta3b,params,'layer2',relu_deriv)
        backwards(delta4b,params,'layer1',relu_deriv)

        params['m_Wlayer1'] = 0.9*params['m_Wlayer1'] - learning_rate*params['grad_Wlayer1']
        params['Wlayer1'] = params['Wlayer1'] + params['m_Wlayer1']
        
        params['m_blayer1'] = 0.9*params['m_blayer1'] - learning_rate*params['grad_blayer1']
        params['blayer1'] = params['blayer1'] + params['m_blayer1']
        
        params['m_Wlayer2'] = 0.9*params['m_Wlayer2'] - learning_rate*params['grad_Wlayer2']
        params['Wlayer2'] = params['Wlayer2'] + params['m_Wlayer2']

        params['m_blayer2'] = 0.9*params['m_blayer2'] - learning_rate*params['grad_blayer2']
        params['blayer2'] = params['blayer2'] + params['m_blayer2']

        params['m_Wlayer3'] = 0.9*params['m_Wlayer3'] - learning_rate*params['grad_Wlayer3']
        params['Wlayer3'] = params['Wlayer3'] + params['m_Wlayer3']

        params['m_blayer3'] = 0.9*params['m_blayer3'] - learning_rate*params['grad_blayer3']
        params['blayer3'] = params['blayer3'] + params['m_blayer3']
        
        params['m_Woutput'] = 0.9*params['m_Woutput'] - learning_rate*params['grad_Woutput']
        params['Woutput'] = params['Woutput'] + params['m_Woutput']

        params['m_boutput'] = 0.9*params['m_boutput'] - learning_rate*params['grad_boutput']
        params['boutput'] = params['boutput'] + params['m_boutput']

    # forward
    h1 = forward(valid_x, params, 'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    h3 = forward(h2, params, 'layer3',relu)
    probs = forward(h3,params,'output',sigmoid)
        
    # loss
    vloss = np.sum((probs - valid_x)**2)
    train_loss[itr] = total_loss/len(train_x)
    valid_loss[itr] = vloss/len(valid_x)
    
    if itr % 2 == 0:
        print("Epoch: {:2d} \t Training: \t loss: {:.2f} \t ".format(itr, train_loss[itr]),"\t Validation: \t loss: {:.2f}".format(valid_loss[itr]))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
n = range(max_iters)
plt.figure("Loss")
plt.plot(n, train_loss, label='training')
plt.plot(n, valid_loss, label='validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()  
plt.show()    

with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q5.3.1
# visualize some results
#########################
#### your code here #####
#########################
with open('q5_weights.pickle', 'rb') as handle:
    params = pickle.load(handle)
index = [0, 10, 100, 130, 500, 560, 1000, 1002, 2000, 2002]
# index = (0, 3600, 100)
for i in index:
    image = valid_x[i]
    h1 = forward(image, params, 'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    h3 = forward(h2, params, 'layer3',relu)
    image_recon = forward(h3,params,'output',sigmoid)

    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image.reshape(32,32).T, cmap='gray')
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(1, 2, 2)
    plt.imshow(image_recon.reshape(32,32).T, cmap='gray')
    plt.axis('off')
    plt.title("Image Reconstructed")
    plt.show()

# Q5.3.2
# evaluate PSNR
##########################
##### your code here #####
##########################
with open('q5_weights.pickle', 'rb') as handle:
    params = pickle.load(handle)
psnr = 0
for i in range(len(valid_x)):
    image = valid_x[i]
    h1 = forward(image, params, 'layer1',relu)
    h2 = forward(h1,params,'layer2',relu)
    h3 = forward(h2, params, 'layer3',relu)
    image_recon = forward(h3,params,'output',sigmoid)

    psnr += peak_signal_noise_ratio(image, image_recon)

print(psnr/len(valid_x))
