#Calulates the output weights ...


import numpy as np

# reference for normal equation: http://www4.ncsu.edu/eos/users/w/white/www/white/mamac/Lecture%2011.pdf

X_temp = np.genfromtxt('data.txt', delimiter = ',')

N = 3               # number of inputs
Nh = 2              # number hidden layers
No = 1              # number output layers

##Variables## 

# y_col = output values
# X matrix = data
# weights = inner weights
# output_weights 
# inputs =  x(x,y,z) input values
# net = net weighted input to final layer
# O = result of activation function
# y_hat = output of processor
# cost = difference between system output and desired
# E = MSE
# R = left side of "normal equation" solution to minimize MSE
# C = right side
# Woh = modified output weights
# train_data = first 800 patterns
# test_data  = last 200 patterns 
# setup

(M_temp, N) = np.shape(X_temp)       # M = number patterns
                                # N = number columns
                                
M = 500                       # test train split - should split more randomly...

y_col =  X_temp[:,N-1]          
X = np.ones((M, N+1))           # create an array of ones 
X[:M,:N-1] = X_temp[:M,:N-1]         
weights = np.random.rand(M, N)          
output_weights = np.random.rand(M, No)

train_data = X[:M,:N]
print(train_data.shape)
test_data = X[M:,:N]
print (test_data.shape)

#send inputs through system
net = np.transpose(np.dot(weights,train_data.T))# Push inputs through the weights/matrix multiply
O = np.tanh(net)                            # Push output through activation function 

#calculate cost of original system output
y_hat = np.dot(O,output_weights)    #*****switched order here, I think this amtches the equation in the reference doc
y_hat = np.transpose(y_hat)         # change column to vector
y_hat = np.squeeze(y_hat)           # remove empty dimension
cost = y_col[:M] - y_hat[:M]              
E = 1./M*np.dot(cost,cost)          # error function MSE
print ("The original error is : ", E)

#solve for best weights 
R = np.dot(O.T,O)               # left side linear equation "normal equation"
C = np.dot(O.T,y_col[:M])           # right side linear equation "normal equation"
Woh = np.linalg.solve(R,C)      # matrix solve to find optimal output weights
output_weights = Woh

#calculate training error
y_hat = np.dot(O,output_weights)   
y_hat = np.transpose(y_hat)       
y_hat = np.squeeze(y_hat)           
cost = y_col[:M] - y_hat               
E = 1./M*np.dot(cost,cost)       
print ("The training error is : ", E)

#Re calculate cost with test data
net = np.transpose(np.dot(weights,train_data.T)) # Push inputs through the weights/matrix multiply
O = np.tanh(net)                                # Push output through activation function 

y_hat = np.dot(O,output_weights)   
y_hat = np.transpose(y_hat)       
y_hat = np.squeeze(y_hat)           
cost = y_col[M:] - y_hat               
E = 1./M*np.dot(cost,cost)       
print ("The test error is : ", E)


