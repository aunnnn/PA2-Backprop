import numpy as np
import pickle
from tqdm import tqdm_notebook as tqdm

config = {}
config['layer_specs'] = [784, 100, 100, 10]  # The length of list denotes number of hidden layers; each element denotes number of neurons in that layer; first element is the size of input layer, last element is the size of output layer.
config['activation'] = 'sigmoid' # Takes values 'sigmoid', 'tanh' or 'ReLU'; denotes activation function for hidden layers
config['batch_size'] = 500  # Number of training samples per batch to be passed to network
config['epochs'] = 50  # Number of epochs to train the model
config['early_stop'] = True  # Implement early stopping or not
config['early_stop_epoch'] = 5  # Number of epochs for which validation loss increases to be counted as overfitting
config['L2_penalty'] = 0  # Regularization constant
config['momentum'] = False  # Denotes if momentum is to be applied or not
config['momentum_gamma'] = 0.9  # Denotes the constant 'gamma' in momentum expression
config['learning_rate'] = 0.001 # Learning rate of gradient descent algorithm

def softmax(x):
  """
  Write the code for softmax activation function that takes in a numpy array and returns a numpy array.
  """
  return output

def sigmoid(x):
  """
  General Sigmoid function
  """
  return 1./(1. + np.exp(-x))

def load_data(fname):
  """
  Write code to read the data and return it as 2 numpy arrays.
  Make sure to convert labels to one hot encoded format.
  """
  f = open(fname, 'rb')
  data = pickle.load(f)
  f.close()
  images, labels = data[:, :-1], data[:, -1]

  labels = labels.astype(np.int)
  onehotlabels = np.zeros((len(labels), labels.max()+1))
  onehotlabels[np.arange(len(labels)), labels] = 1
  return images, onehotlabels


class Activation:
  def __init__(self, activation_type = "sigmoid"):
    self.activation_type = activation_type
    self.x = None # Save the input 'x' for sigmoid or tanh or ReLU to this variable since it will be used later for computing gradients.
  
  def forward_pass(self, a):
    if self.activation_type == "sigmoid":
      return self.sigmoid(a)
    
    elif self.activation_type == "tanh":
      return self.tanh(a)
    
    elif self.activation_type == "ReLU":
      return self.ReLU(a)
  
  def backward_pass(self, delta):
    if self.activation_type == "sigmoid":
      grad = self.grad_sigmoid()
    
    elif self.activation_type == "tanh":
      grad = self.grad_tanh()
    
    elif self.activation_type == "ReLU":
      grad = self.grad_ReLU()
    
    return grad * delta
      
  def sigmoid(self, x):
    """
    Write the code for sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = sigmoid(x)
    return output

  def tanh(self, x):
    """
    Write the code for tanh activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = np.tanh(x)
    return output

  def ReLU(self, x):
    """
    Write the code for ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    self.x = x
    output = np.maximum(x, 0)
    return output

  def grad_sigmoid(self):
    """
    Write the code for gradient through sigmoid activation function that takes in a numpy array and returns a numpy array.
    """
    sigmoid_x = sigmoid(self.x)
    grad = sigmoid_x * (1-sigmoid_x)
    return grad

  def grad_tanh(self):
    """
    Write the code for gradient through tanh activation function that takes in a numpy array and returns a numpy array.
    """
    tanh_x = np.tanh(self.x)
    grad = 1 - (tanh_x * tanh_x)
    return grad

  def grad_ReLU(self):
    """
    Write the code for gradient through ReLU activation function that takes in a numpy array and returns a numpy array.
    """
    grad = np.where(self.x <= 0, 0, 1)
    return grad


class Layer():
  def __init__(self, in_units, out_units):
    np.random.seed(42)
    self.w = np.random.randn(in_units, out_units)  # Weight matrix
    self.b = np.zeros((1, out_units)).astype(np.float32)  # Bias
    self.x = None  # Save the input to forward_pass in this
    self.a = None  # Save the output of forward pass in this (without activation)
    self.d_x = None  # Save the gradient w.r.t x in this (AKA Delta to pass to previous layer = dE/dx)
    self.d_w = None  # Save the gradient w.r.t w in this (AKA dE/dw = Delta received . dx/dw)
    self.d_b = None  # Save the gradient w.r.t b in this (AKA dE/db = Delta received . 1)

  def forward_pass(self, x):
    """
    Write the code for forward pass through a layer. Do not apply activation function here.
    """
    self.x = x
    self.a = x @ self.w + self.b
    return self.a
  
  def backward_pass(self, delta):
    """
    Write the code for backward pass. This takes in gradient from its next layer as input,
    computes gradient for its weights and the delta to pass to its previous layers.
    """

    # This is dE/dw = delta received . dx/dw
    self.d_w = (np.swapaxes(self.x.T[:,:,np.newaxis],1,2) * np.swapaxes(delta[:,:,np.newaxis], 0, 2)).sum(axis=2)

    # This is dE/db = delta received . 1
    self.d_b = delta
                
    # This is delta to be passed to previous layer
    self.d_x = delta @ self.w.T
    return self.d_x

      
class Neuralnetwork():
  def __init__(self, config):
    self.layers = []
    self.x = None  # Save the input to forward_pass in this
    self.y = None  # Save the output vector of model in this
    self.targets = None  # Save the targets in forward_pass in this variable
    for i in range(len(config['layer_specs']) - 1):
      self.layers.append( Layer(config['layer_specs'][i], config['layer_specs'][i+1]) )
      
      # Unless it's output unit, add Activation layer on top
      if i < len(config['layer_specs']) - 2:
        self.layers.append(Activation(config['activation']))  
    
  def forward_pass(self, x, targets=None):
    """
    Write the code for forward pass through all layers of the model and return loss and predictions.
    If targets == None, loss should be None. If not, then return the loss computed.
    """
    self.x = x
    self.targets = targets
    
    # Input layer
    out = self.layers[0].forward_pass(x)
    
    # Forward...
    for layer in self.layers[1:]:
      out = layer.forward_pass(out)
      
    # Softmax
    out_exp = np.exp(out)
    sum_out_exp = np.exp(out).sum(axis=1)        
    self.y = out_exp/sum_out_exp[:,None]
        
    # Cross-entropy loss
    if targets is not None:
      return self.loss_func(self.y, targets), self.y
    else:    
      return None, self.y

  def loss_func(self, logits, targets):
    '''
    find cross entropy loss between logits and targets
    '''
    output = (targets * np.log(logits)).sum()/len(targets)
    return output
    
  def backward_pass(self):
    '''
    implement the backward pass for the whole network. 
    hint - use previously built functions.
    '''
    delta = self.targets - self.y
    
    for layer in reversed(self.layers):
      delta = layer.backward_pass(delta)
    return delta

def trainer(model, X_train, y_train, X_valid, y_valid, config):
  """
  Write the code to train the network. Use values from config to set parameters
  such as L2 penalty, number of epochs, momentum, etc.
  """
  
  BATCH_SIZE = config['batch_size']
  N_EPOCHS = config['epochs']
  LEARNING_RATE = config['learning_rate']
  
  N_BATCHES = len(X_train) // BATCH_SIZE
  
  print("N Batches: ",N_BATCHES, "Batch size", BATCH_SIZE)
  
  for i_epoch in tqdm(range(N_EPOCHS)):
    for i_minibatch in range(0, len(X_train), BATCH_SIZE):
      X_batch = X_train[i_minibatch:i_minibatch + BATCH_SIZE]
      y_batch = y_train[i_minibatch:i_minibatch + BATCH_SIZE]
      
      loss, y = model.forward_pass(X_batch, y_batch)
      delta = model.backward_pass()
            
      # Weight updates
      for l in model.layers:
        if type(l) is Layer:
          l.w += LEARNING_RATE * l.d_w
      

    print('Epoch:', i_epoch, 'loss:', loss)
      
  
  
def test(model, X_test, y_test, config):
  """
  Write code to run the model on the data passed as input and return accuracy.
  """
  _, logits = model.forward_pass(X_test)
  predictions = np.argmax(logits, axis=1) # inds that we made predictions (0,1,2,3, ...)
  
  # convert y_test from one-hot to actual target inds
  targets = y_test.nonzero()[1]
  accuracy = (predictions == targets).sum()/len(targets)
  return accuracy
      

if __name__ == "__main__":
  train_data_fname = 'data/MNIST_train.pkl'
  valid_data_fname = 'data/MNIST_valid.pkl'
  test_data_fname = 'data/MNIST_test.pkl'
  
  ### Train the network ###
  model = Neuralnetwork(config)
  X_train, y_train = load_data(train_data_fname)
  X_valid, y_valid = load_data(valid_data_fname)
  X_test, y_test = load_data(test_data_fname)
  trainer(model, X_train, y_train, X_valid, y_valid, config)
  test_acc = test(model, X_test, y_test, config)