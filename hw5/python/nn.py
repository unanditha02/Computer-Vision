import numpy as np
from util import *
# from sklearn.metrics import log_loss
# do not include any more libraries here!
# do not put any code outside of functions!

############################## Q 2.1 ##############################
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    ##########################
    ##### your code here #####
    ##########################
    np.random.seed(1)
    b = np.zeros(out_size)
    limit = np.sqrt(6/(in_size+out_size))

    W = np.random.uniform(-limit, limit, size=(in_size, out_size))
    params['W' + name] = W
    params['b' + name] = b

############################## Q 2.2.1 ##############################
# x is a matrix
# a sigmoid activation function
def sigmoid(x):

    ##########################
    ##### your code here #####
    ##########################
    res = 1 / (1 + np.exp(-x))
    return res

############################## Q 2.2.1 ##############################
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]


    ##########################
    ##### your code here #####
    ##########################
    pre_act = X @ W + b
    post_act = activation(pre_act)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

############################## Q 2.2.2  ##############################
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    # res = None

    ##########################
    ##### your code here #####
    ##########################
    examples, data = x.shape
    res = np.zeros((examples, data))
    exponential = np.exp(x)
    for e in range(examples):
        s = np.sum(exponential[e,:])
        for f in range(data):
            res[e, f] = exponential[e,f]/s
    return res

############################## Q 2.2.3 ##############################
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss, acc = None, None

    ##########################
    ##### your code here #####
    ##########################
    trueValues = 0
    totalValues = len(y)
    loss = -np.sum(np.multiply(y, np.log(probs)))
    for i in range(totalValues):
        if np.argmax(probs[i,:]) == np.argmax(y[i,:]):
            trueValues += 1
    acc = trueValues/totalValues
    return loss, acc 

############################## Q 2.3 ##############################
# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]

    # do the derivative through activation first
    # then compute the derivative W,b, and X
    ##########################
    ##### your code here #####
    ##########################
    grad_W = X.T @ (delta * activation_deriv(post_act))
    grad_b = np.sum(delta * activation_deriv(post_act), axis=0)
    grad_X = delta * activation_deriv(post_act) @ W.T
    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

############################## Q 2.4 ##############################
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []
    ##########################
    ##### your code here #####
    ##########################
    examples = len(x)
    num_batches = int(examples/batch_size)
    a = np.arange(examples)
    indices = np.zeros((num_batches, batch_size))
    for i in range(num_batches):
        indices[i,:] = np.random.choice(a, batch_size, replace=False)
        x_batch = np.zeros((batch_size, x.shape[1]))
        y_batch = np.zeros((batch_size, y.shape[1]))
        for j in range(batch_size):
            x_batch[j,:] = x[int(indices[i,j]),:]
            y_batch[j,:] = y[int(indices[i,j]),:]
            a = a[a != indices[i,j]]
        batches.append((x_batch, y_batch))

    return batches
