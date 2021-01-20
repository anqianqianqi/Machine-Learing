#@Author: Anqi Luo
import csv
import numpy as np
from matplotlib import pyplot as plt
import time

#read data line by line and fold the bias term by adding constant -1 into every entry
#constant -1 will work because we inspected that labels for attributes are all positive integers  
def read_data(infile:str):
    master = []
    with open(infile,'r') as f:
        lines = csv.reader(f,delimiter='\t')
        for line in lines:
            item = [int(j.split( sep=':' )[0]) for j in [a for a in line[1:]]]
            item.append(-1)
            item.insert( 0,float( line[0] ) ) #add the true label of the item at the initial position 0
            master.append(item)
    return master

#helper function to get the dot product of two sparse matrices
def sparse_dot(X:list, theta):
    product = 0.0
    X = X[1:]
    for item in X:
        product += theta.get(item,0.0)
    return product
    
#helper function to calculate gradient with one training sample and update theta
def get_gradient(data:list,theta:dict,learning_rate:float):
    for word in data:
        dot_product = sparse_dot(word,theta)
        increment = learning_rate * (float(word[0])-np.exp(dot_product)/(1+np.exp((dot_product))))
        for feature in word[1:]:
            theta[feature] = theta.get(feature,0.0) + increment
    return theta

#predict label using the trained parameter: theta and report error rate
def get_error(data:list,theta):
    sum = 0
    for word in data:
        dot_product = sparse_dot( word,theta )
        if dot_product >= 0:
            prediction = float(1)
        else:
            prediction = float(0)
        if word[0] != prediction:
            sum+=1
    return float(sum/len(data))

def get_log_error(data:list,theta):
    loss = 0.0
    for word in data:
        dot_product = sparse_dot( word,theta )
        loss += np.log(1+np.exp(dot_product)) - word[0]*(dot_product)
    return loss/len(data)



if __name__ == '__main__':
    start_time = time.time()
    train_data = read_data('model1_formatted_train.tsv')
    valid_data = read_data('model1_formatted_valid.tsv')
    test_data = read_data('model1_formatted_test.tsv')
    theta = {}
    train_log_loss = []
    valid_log_loss = []
    epoch = 60
    #train theta by running 60 epochs of the whole training dataset
    for _ in range(epoch):
        theta = get_gradient(train_data,theta,0.1)
        train_log_loss.append(get_log_error(train_data,theta))
        valid_log_loss.append(get_log_error(valid_data,theta ))
    print( "training error is {}".format(get_error( train_data,theta ) ))
    print( "testing error is {}".format( get_error( test_data,theta ) ) )
    plt.xlabel("Epoch")
    plt.ylabel("Average Negative Log Likelihood")
    x_axis = np.linspace(0, epcho - 1, epcho)
    plt.plot(x_axis, train_log_loss,  linewidth=2.0, label='Training')
    plt.plot(x_axis, valid_log_loss,  linewidth=2.0, label='Validation')
    plt.title('Average Negative Log Likelihood versus Training Epoch')
    plt.legend(loc='upper right')
    plt.savefig('model1-60epcho.png')
    print( "--- %s minutes ---" % ((time.time() - start_time)/60.0))

