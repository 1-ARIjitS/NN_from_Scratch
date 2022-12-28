#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import figure
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# In[2]:


df= pd.read_csv('letter-recognition.data', header=None)
print(df.shape)
df.head()


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


# label encoding the class labels
from sklearn.preprocessing import LabelEncoder
label_encoder= LabelEncoder()
df[0]= label_encoder.fit_transform(df[0])
print(df[0].unique())
df.head()


# In[6]:


df_test= df[[0]]
print(df_test.shape)
df_test.head()


# In[7]:


df_train= df.drop(0, axis=1, inplace=False)
print(df_train.shape)
df_train.head()


# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


X_train, X_rem, y_train, y_rem = train_test_split(df_train, df_test, train_size=0.8)

X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)

print("x_train dataset size: ",X_train.shape)
print("y_train dataset size: ",y_train.shape)
print("x_validation dataset size: ",X_valid.shape)
print("y_validation dataset size: ",y_valid.shape)
print("x_test dataset size: ",X_test.shape)
print("y_test dataset size: ",y_test.shape)


# In[10]:


print("number of unique classes in the dataset is- ", len(df[0].unique()))


# In[11]:


df[0].value_counts()


# In[12]:


plt.figure(figsize = (13,6))
plt.title("COUNT OF LETTERS")
sns.countplot(x=0, data= df)
plt.show()


# In[13]:


# we can observe that the datset is balanced and dont have any null values so we proceed with ot 


# # Building the NN

# In[14]:


# it has 2 layers i.e. 1 input, 1 hidden and 1 output layer
# lets assume 
# 40 neurons in the first hidden layer
# output layer is a 26 class softmax layer for multiclass classification

weight matrix between input layer and 1st hidden layer has size [16 x 40]
bias vector between input layer and 1st hidden layer is of size [40 x 1]

weight matrix between 1st hidden layer and output softmax layer is [40 x 26]
bias vector between 1st hidden layer and output softmax layer is of size [26 x 1]
# In[15]:


# converting all the datasets into arrays
x_train_arr = np.array(X_train)
y_train_arr = np.array(y_train)
x_test_arr = np.array(X_test)
y_test_arr = np.array(y_test)
x_valid_arr = np.array(X_valid)
y_valid_arr = np.array(y_valid)


# In[16]:


onehot= OneHotEncoder()


# In[17]:


y_train_arr = onehot.fit_transform(y_train_arr.reshape(-1,1))
y_train_arr = y_train_arr.toarray()
y_train_arr


# In[18]:


y_test_arr = onehot.fit_transform(y_test_arr.reshape(-1,1))
y_test_arr = y_test_arr.toarray()
y_train_arr


# In[19]:


y_valid_arr = onehot.fit_transform(y_valid_arr.reshape(-1,1))
y_valid_arr = y_valid_arr.toarray()
y_valid_arr


# In[20]:


class activation_function:
    def __init__(self):
        self.name = None
        self.output = None
        
    def tanh(self, input_data):
        return np.tanh(input_data)
    
    def grad_tanh(self, input_data):
        return 1-np.tanh(input_data)**2
    
    def relu(self, x):
        self.output = np.maximum(0,x)
        return self.output
    
    def grad_relu(self, x):
        self.output[self.output>0] = 1
        return self.output
    
    def activation(self, name, x):
        self.name = name
        if self.name == "relu":
            return self.relu(x)
        elif self.name == "tanh":
            return self.tanh(x)
        
    def grad_activation(self,x):
        if self.name == "relu":
            return self.grad_relu(x)
        elif self.name == "tanh":
            return self.grad_tanh(x)


# In[21]:


# defining all three activation functions

def activation_relu(x):
    return np.maximum(0,x)
    
def activation_tanh(x):
    return np.tanh(x)

# def activation_softmax(x):
#     exp_values= np.exp(x - np.max(x, axis=1, keepdims=True))
#     probabilities= exp_values/ np.sum(exp_values, axis= 1, keepdims=True)
#     return probabilities


# In[22]:


#defining the gradients of both the activation functions for updating the weights

def gradient_relu(x):
    x[x>0]=1
    return x

def gradient_tanh(x):
    return (1- np.tanh(x)**2)

# def gradient_softmax(x):
#     jacobian= np.diag(x)
#     for i in range(len(jacobian)):
#         for j in range(len(jacobian)):
#             if i == j:
#                 jacobian[i][j] = x[i] * (1-x[i])
#             else: 
#                 jacobian[i][j] = -x[i]*x[j]
#     return jacobian


# In[23]:


# #defining the adam optimizer for faster convergence

# def optimizer_adam(weight, alpha=0.001, beta1= 0.9, beta2= 0.999, epsilon= 1e-8):
#     t=0
    
#     m_t=0
#     v_t=0
    
#     m_t_hat= m_t
#     v_t_hat= v_t
    
#     num_steps=1000
    
#     while t in range(1,num_steps):
#         dw= compute_gradient(weight)
        
#         m_t= beta1*m_t+ (1-beta1)*dw
#         v_t= beta2*v_t+ (1-beta2)*dw*dw
        
#         m_t_hat= m_t/(1- beta1**t)
#         v_t_hat= v_t/(1-beta2**t)
        
#         weight= weight- alpha*(m_t_hat/(v_t_hat.sqrt()+ epsilon))
    
#     return weight    


# In[24]:


#defining a layer

class layer:
    def __init__(self):
        self.input = None
        self.output = None

    # computes the output Y of a layer for a given input X
    def forward_propagation(self, input):
        pass

    # computes dE/dX for a given dE/dY (and update parameters if any)
    def backward_propagation(self, output_error, learning_rate):
        pass


# In[25]:


# the fully connected layer

class fully_connected_layer:
    def __init__(self, input_size, output_size):
        # xavier weight initialization
        # input_size = fan_in
        # output_size = fan_out
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.random.randn(1,output_size)

    # returns output for a given input
    def forward_propagation(self, input_data):
        self.input = input_data.reshape(1,-1)
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backward_propagation(self, output_gradient, learning_rate, optimizer, T):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = output_gradient

        # updating the parameters
        self.weights = self.weights - (learning_rate * weights_gradient)
        self.bias = self.bias - (learning_rate * output_gradient)
        
        if optimizer=='adam':
            opt = AdamOptim(eta=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
            self.weights, self.bias = opt.update(t=T, w = self.weights, b = self.bias, dw = weights_gradient, db = output_gradient)
        
        return input_gradient


# In[26]:


# # the actiation layer

# class activation_layer:
#     def __init__(self, activation, grad_activation):
#         self.activation = activation
#         self.grad_activation = grad_activation

#     # returns the activated input
#     def forward_propagation(self, input_data):
#         self.input = input_data
#         self.output = self.activation(self.input)
#         return self.output

#     # Returns input_error=dE/dX for a given output_error=dE/dY.
#     # learning_rate is not used because there is no "learnable" parameters.
#     def backward_propagation(self, output_gradient, learning_rate, optimizer, T):
#         return self.grad_activation(self.input) * output_gradient


# In[27]:


# the activation layer

class activation_layer:
    def __init__(self, act_func):
        self.input = None
        self.output = None
        self.act_func = act_func
        self.act = activation_function()
        
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.act.activation(self.act_func, input_data)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate, optimizer, T):
        return self.act.grad_activation(self.input)*output_gradient


# In[28]:


# defining the softmax activation layer
class softmax_layer:
    def __init__(self, input_size):
        self.input = None
        self.output = None
        self.input_size = input_size
        
    def forward_propagation(self, input_data):
        self.input = input_data
        exp = np.exp(self.input - np.max(self.input, axis=1, keepdims=True))
        exp_sum = np.sum(exp, axis=1, keepdims=True)
        self.output = exp/(exp_sum)
        return self.output
    
    def backward_propagation(self, output_data, learning_rate, optimizer, T):
        dz_x = np.zeros((self.input_size,self.input_size))
        for i in range(self.input_size):
            for j in range(self.input_size):
                if i == j:
                    dz_x[i,j] = self.output[0,i]*(1-self.output[0,i])
                else:
                    dz_x[i,j] = -self.output[0,i]*self.output[0,j]
                    
        dl_x = np.dot(output_data,dz_x)
        return dl_x


# In[29]:


# defining the loss function
def cross_entropy_loss(y_true, y_pred):
    loss = np.sum(-np.log(y_pred+1e-20)*y_true)
    return loss
    
def grad_cross_entropy_loss(y_true, y_pred):
    grad_loss = -y_true/(y_pred + 1e-20)
    return grad_loss


# In[30]:


# defining the adam optimizer
class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.eta = eta
    def update(self, t, w, b, dw, db):
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        # *** biases *** #
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        # *** biases *** #
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**t)
        m_db_corr = self.m_db/(1-self.beta1**t)
        v_dw_corr = self.v_dw/(1-self.beta2**t)
        v_db_corr = self.v_db/(1-self.beta2**t)
#         print(v_db_corr)

        ## update weights and biases
        w = w - self.eta*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.eta*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        return w, b


# In[31]:


class MLP:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.grad_loss= None
        self.optimizer = None
        self.train_loss = []
        self.validation_loss = []
        self.train_accuracy= []
        self.validation_accuracy = []
        
    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, grad_loss):
        self.loss = loss
        self.grad_loss= grad_loss

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            output= (output==np.max(output))*1
            result.append(output[0])

        return result
    
    # train the network
    def fit(self, x_train, y_train, x_valid, y_valid, epochs, learning_rate, optimizer="adam"):
        # sample dimension first
        samples = len(x_train)
        self.optimizer= optimizer
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss
                err += self.loss(y_train[j], output)

                # backward propagation
                error = self.grad_loss(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate, optimizer= self.optimizer, T=i+1)

            # training loss
            err /= samples
            self.train_loss.append(err)
            print("epoch- {}/{}, loss- {}".format(i,epochs,err))
            
            # accuracy for train set
            y_pred = self.predict(x_train)
            y_pred = onehot.inverse_transform(y_pred)
            y_pred = y_pred.flatten()
            
            y_true = y_train
            y_true = onehot.inverse_transform(y_true)
            y_true = y_true.flatten()
            
            self.train_accuracy.append(accuracy_score(y_pred, y_true))
            
            #Validation loss
            valid_samples= len(x_valid)
            err_valid= 0
            for j in range(valid_samples):
                output = x_valid[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                err_valid += self.loss(y_valid[j], output)

            err_valid /= valid_samples
            self.validation_loss.append(err_valid)
            
            # accuracy for validation set
            y_valid_pred = self.predict(x_valid)
            y_valid_pred = onehot.inverse_transform(y_valid_pred)
            y_valid_pred = y_valid_pred.flatten()
            
            y_valid_true = y_valid
            y_valid_true = onehot.inverse_transform(y_valid_true)
            y_valid_true = y_valid_true.flatten()
            
            self.validation_accuracy.append(accuracy_score(y_valid_pred, y_valid_true))
    
    def plot(self):
        # plotting the graphs for loss vs epochs
        plt.figure(figsize=(15,8))
        plt.subplot(1,2,1)
        plt.plot(self.train_loss, label = "training set")
        plt.plot(self.validation_loss, label = "validation set")
        plt.title("loss vs epochs")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        
        # plotting the graphs for accuracy vs epochs
        plt.subplot(1,2,2)
        plt.plot(self.train_accuracy, label = "training set")
        plt.plot(self.validation_accuracy, label = "validation set")
        plt.title("accuracy vs epochs")
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.legend()


# # for tanh activation

# In[32]:


# neural network for tanh activation function
nn= MLP()
nn.add(fully_connected_layer(16, 40))
nn.add(activation_layer(act_func="tanh"))
#nn.add(activation_layer(activation_tanh, gradient_tanh))
nn.add(fully_connected_layer(40, 26))
nn.add(softmax_layer(26))
nn.use(cross_entropy_loss, grad_cross_entropy_loss)
nn.fit(x_train_arr, y_train_arr, x_valid_arr, y_valid_arr,  epochs=20, learning_rate=0.001, optimizer="adam")
nn.plot()

y_test_pred = np.array(nn.predict(x_test_arr))
y_test_pred = onehot.inverse_transform(y_test_pred)
y_test_true = onehot.inverse_transform(y_test_arr)
print("Accuracy on test set: ", accuracy_score(y_test_pred,y_test_true))


# # for relu activation

# In[37]:


# neural network for relu activation function
nn= MLP()
nn.add(fully_connected_layer(16, 40))
nn.add(activation_layer(act_func="relu"))
#nn.add(activation_layer(activation_relu, gradient_relu))
nn.add(fully_connected_layer(40, 26))
nn.add(softmax_layer(26))
nn.use(cross_entropy_loss, grad_cross_entropy_loss)
nn.fit(x_train_arr, y_train_arr, x_valid_arr, y_valid_arr,  epochs=20, learning_rate=1e-4, optimizer="adam")
nn.plot()

y_test_pred = np.array(nn.predict(x_test_arr))
y_test_pred = onehot.inverse_transform(y_test_pred)
y_test_true = onehot.inverse_transform(y_test_arr)
print("Accuracy on test set: ", accuracy_score(y_test_pred,y_test_true))


# # for different learning rates on tanh activation

# In[34]:


lr=[1e-5,1e-4,0.001,0.01,0.1]
for i in lr:
    print("training for learning rate",i)
    print("---------------------------------")
    nn= MLP()
    nn.add(fully_connected_layer(16, 40))
    nn.add(activation_layer(act_func="tanh"))
    #nn.add(activation_layer(activation_relu, gradient_relu))
    nn.add(fully_connected_layer(40, 26))
    nn.add(softmax_layer(26))
    nn.use(cross_entropy_loss, grad_cross_entropy_loss)
    nn.fit(x_train_arr, y_train_arr, x_valid_arr, y_valid_arr,  epochs=20, learning_rate=i, optimizer="adam")
    nn.plot()
    
    y_test_pred = np.array(nn.predict(x_test_arr))
    y_test_pred = onehot.inverse_transform(y_test_pred)
    y_test_true = onehot.inverse_transform(y_test_arr)
    print("Accuracy on test set: ", accuracy_score(y_test_pred,y_test_true))
    print("\n")


# # for different learning rates on relu activation

# In[35]:


lr=[1e-5,1e-4,0.001,0.01,0.1]
for i in lr:
    print("training for learning rate",i)
    print("---------------------------------")
    nn= MLP()
    nn.add(fully_connected_layer(16, 40))
    nn.add(activation_layer(act_func="relu"))
    #nn.add(activation_layer(activation_relu, gradient_relu))
    nn.add(fully_connected_layer(40, 26))
    nn.add(softmax_layer(26))
    nn.use(cross_entropy_loss, grad_cross_entropy_loss)
    nn.fit(x_train_arr, y_train_arr, x_valid_arr, y_valid_arr,  epochs=20, learning_rate=i, optimizer="adam")
    nn.plot()
    
    y_test_pred = np.array(nn.predict(x_test_arr))
    y_test_pred = onehot.inverse_transform(y_test_pred)
    y_test_true = onehot.inverse_transform(y_test_arr)
    print("Accuracy on test set: ", accuracy_score(y_test_pred,y_test_true))
    print("\n")


# # for different epochs on tanh activation

# In[38]:


# lets run it for 100 epochs so that we can observe what is happening for all the epochs between 20 and 100
# neural network for tanh activation function
nn= MLP()
nn.add(fully_connected_layer(16, 40))
nn.add(activation_layer(act_func="tanh"))
#nn.add(activation_layer(activation_tanh, gradient_tanh))
nn.add(fully_connected_layer(40, 26))
nn.add(softmax_layer(26))
nn.use(cross_entropy_loss, grad_cross_entropy_loss)
nn.fit(x_train_arr, y_train_arr, x_valid_arr, y_valid_arr,  epochs=100, learning_rate=0.001, optimizer="adam")
nn.plot()

y_test_pred = np.array(nn.predict(x_test_arr))
y_test_pred = onehot.inverse_transform(y_test_pred)
y_test_true = onehot.inverse_transform(y_test_arr)
print("Accuracy on test set: ", accuracy_score(y_test_pred,y_test_true))


# # for different epochs on relu activation

# In[39]:


# lets run it for 100 epochs so that we can observe what is happening for all the epochs between 20 and 100
# neural network for tanh activation function
nn= MLP()
nn.add(fully_connected_layer(16, 40))
nn.add(activation_layer(act_func="relu"))
#nn.add(activation_layer(activation_tanh, gradient_tanh))
nn.add(fully_connected_layer(40, 26))
nn.add(softmax_layer(26))
nn.use(cross_entropy_loss, grad_cross_entropy_loss)
nn.fit(x_train_arr, y_train_arr, x_valid_arr, y_valid_arr,  epochs=100, learning_rate=1e-4, optimizer="adam")
nn.plot()

y_test_pred = np.array(nn.predict(x_test_arr))
y_test_pred = onehot.inverse_transform(y_test_pred)
y_test_true = onehot.inverse_transform(y_test_arr)
print("Accuracy on test set: ", accuracy_score(y_test_pred,y_test_true))

so from the above analysis we ge that the best parameters for tanh are as follows-
1. learning rate- 0.001 or 1e-3
2. epoch- 100 epochs can be taken as test accuracy and generalization gap does not change significantly
we get accuracy on test set- 68.3%so from the above analysis we ge that the best parameters for relu are as follows-
1. learning rate- 0.0001 or 1e-4
2. epoch- we mus take 20 epochs because after that the loss increases and the accuracy decreases rapidly
we get accuracy on test set- 63.5%
# # finally we prefer tanh activation with learning rate 0.001 or 1e-3, number of epochs 100 and get test accuracy of 68.3%

# In[40]:


# code ended for Q.1.

