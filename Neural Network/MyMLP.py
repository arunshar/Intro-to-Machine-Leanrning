import numpy as np

def process_data(data,mean=None,std=None):
    # normalize the data to have zero mean and unit variance (add 1e-15 to std to avoid numerical issue)
    if mean is not None:
        data = np.divide(data-mean,std+1e-15)
        # directly use the mean and std precomputed from the training data
        return data
    else:
        # compute the mean and std based on the training data
        # mean = std = 0 # placeholder
        mean = data.mean(axis=0,keepdims=True)
        std = data.std(axis=0,keepdims=True)
        data = np.divide(data-mean,std+1e-15)
        return data, mean, std

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label),10])
    count = 0
    for num in label:
        one_hot[count][num] = 1.0
        count += 1
    return one_hot

def tanh(x):
    f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    # implement the hyperbolic tangent activation function for hidden layer
    # You may receive some warning messages from Numpy. No worries, they should not affect your final results
    # f_x = x # placeholder

    # print (f_x)
    return f_x

def softmax(x):
    # f_x = softmax(x)
    max = np.max(x, axis=1, keepdims=True)
    e_x = np.exp(x - max)
    sum = np.sum(e_x, axis=1, keepdims=True)
    f_x = e_x / sum
    # implement the softmax activation function for output layer
    # f_x = x # placeholder
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvment over the best validation accuracy for more than 50 iterations
        """
        while count<=50:

            z = tanh(np.dot(train_x, self.weight_1) + self.bias_1)
            # print ("hidden layer dimensions ",z.shape)
            o = softmax(np.dot(z,self.weight_2)+self.bias_2)
            # print ("outer layer dimension",o.shape)
            dk = (train_y - o)
            # print (dk.shape)
            # print (z.shape)
            # print (self.weight_2.T.shape)
            dj = np.multiply((1.0-np.power(z,2)),(np.dot(dk,self.weight_2.T)))
            delta_w1 = np.dot(train_x.T,dj)
            delta_w2 = np.dot(z.T,dk)
            self.weight_1 = self.weight_1 + lr * delta_w1
            self.weight_2 = self.weight_2 + lr * delta_w2
            # print (train_x.shape)
            # print (z.shape)
            # print (self.weight_1.shape)
            # print (z.shape)
            # print (dk.shape)
            # print (dj.shape)


            # z = self.get_hidden(train_x)
            # y = self.predict(train_x)
            # print ("The z shape is : ", z.shape)
            # print ("The y shape is : ", y.shape)
            # print (1-z**2)
            # self.weight_1 = lr*(train_y - )
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            # print (train_x.shape)
            # print (train_y.shape)
            # z = self.get_hidden(train_x)
            # r = self.predict(train_x)
            # print (r)
            # print ("The shape of z is :", z.shape)
            # print ("The shape of y is :", train_y.shape)
            # r = self.predict(train_x)
            # print (r)
            # print (train_y.shape)
            # print (r.shape)
            # print (count)
            # print ("This is training : ",train_x.shape)
            # print ("This is testing : ",train_y.shape)

            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters


            #update the parameters based on sum of gradients for all training samples


            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc>best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self,x):
        # generate the predicted probability of different classes
        # print (x.shape)

        # convert class probability to predicted labels
        # print ("This is predict !!!! ")
        # z = self.get_hidden(x)
        # o = softmax(np.dot(np.dot(x,self.weight_1) + self.bias_1,self.weight_2)+self.bias_2)

        o = softmax(np.dot(self.get_hidden(x),self.weight_2)+self.bias_2)
        # print (o.shape)
        # y = np.amax(o, axis=1)
        y = np.argmax(o, axis=1)
        # print (y)
        # print (y.shape)
        # print (np.amax(o))
        # print ("the dimensions of y is : ",len(y[0]))
        # y = np.zeros([len(x),]).astype('int') # placeholder
        # y = softmax(np.dot(x,self.weight_2))
        # print ("Haain ",y.shape)
        return y

    def get_hidden(self,x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        # print (self.weight_1.shape)
        # print ("This is hidden !!! ")
        # print ("Shape of X : ",x.shape)
        # print ("Shape of weight_1 : ", self.weight_1.shape)
        # print (np.dot(x,self.weight_1))
        z = tanh(np.dot(x,self.weight_1) + self.bias_1)
        # print (z)
        # z = x # placeholder
        # print (z.shape)
        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
