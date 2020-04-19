import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import pylab as pl 
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

digits = load_digits()

# Sample dataset
images_and_labels = list(zip(digits.images, digits.target))
plt.figure(figsize=(5,5))
for index, (image, label) in enumerate(images_and_labels[:15]):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % label)
    

# Normalising
for i in range(len(digits.images)):
  digits.images[i] = np.true_divide(digits.images[i], 255)


x = digits.images.reshape((len(digits.images), -1))
y = digits.target


train_indexes=random.sample(range(len(x)),int(len(x)/5)) #20-80 %
test_indexes=[i for i in range(len(x)) if i not in train_indexes]


train_images=[x[i] for i in train_indexes]
test_images=[x[i] for i in test_indexes]
train_labels=[y[i]/10 for i in train_indexes]
test_labels=[y[i]/10 for i in test_indexes]


# Neural Network class
class Neural_Network():
  def __init__(self):
    self.inputSize = 64
    self.outputSize = 1
    self.hiddenSize = 32
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # (64x32) 
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # (32x1) 

  def forward(self, X):
    self.input_x_weights = np.dot(X, self.W1) 
    self.hidden_layer_activation = self.sigmoid(self.input_x_weights) 
    self.hidden_layer_x_weights = np.dot(self.hidden_layer_activation, self.W2) 
    o = self.sigmoid(self.hidden_layer_x_weights) 
    return o 

  def sigmoid(self, s):
    return 1/(1+np.exp(-s))

  def sigmoidPrime(self, s):
    #derivative of sigmoid, s is already the sigmoid value
    return s * (1 - s)

  def backward(self, X, y, o):
    self.o_error = (y - o) ** 2 
    self.o_delta = self.o_error*self.sigmoidPrime(o) 

    self.hidden_layer_activation_error = self.o_delta.dot(self.W2.T) 
    self.hidden_layer_activation_delta = self.hidden_layer_activation_error*self.sigmoidPrime(self.hidden_layer_activation)

    self.W1 += X.T.dot(self.hidden_layer_activation_delta) 
    self.W2 += self.hidden_layer_activation.T.dot(self.o_delta) 

  def train (self, X, y):
    o = self.forward(X)
    self.backward(X, y, o)


# Training
NN = Neural_Network()
for i in range(100000):
  NN.train(np.asarray(train_images),np.asarray(train_labels).reshape(359,1))


# Evaluating
output = NN.forward(test_images)

plt.figure(figsize=(5,5))
for index in range(0,10):
    plt.subplot(3, 5, index + 1)
    plt.axis('off')
    plt.imshow(test_images[index].reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('%i' % (output[index]*10))
plt.show()


# Confusion matrix
matrix = confusion_matrix(np.asarray(test_labels), output)
df_cm = pd.DataFrame(matrix)
sn.set(font_scale=1.4)
sn.heatmap(df_cm,
                   annot=True,
                   annot_kws={"size": 16},
                   cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()    