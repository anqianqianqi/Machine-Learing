import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import time

def linearAlphaForward(X,alpha):
    #input: x.shape = (M+1,1)
    #linear parameter: alpha.shape = (D,M+1)
    #linear output: x*alpha.shape = (D,1)
    X = X.reshape(-1,1)
    return np.matmul(alpha,X)

def sigmoidForward(A):
    # we fold the bias feature 1 as z0
    # activation output: Z.shape = (D+1,1)
    return np.append(1/(1+np.exp(-A)),[1])

def linearBetaForward(Z,beta):
    # Z.shape = (D+1,1)
    # Beta.shape = (K, D+1)
    # linear output: B.shape = (K,1)
    return np.matmul(beta,Z)

def softmax(B):
    # softmax output: YGama.shape = (K,1)
    expB = np.exp(B)
    sum = np.sum(expB)
    return expB/sum

def softmaxBackward(Y:int,YGama):
    #softmaxBackward output  d(loss)/d(B).shape = (K,1)
    YGama[Y] -= 1
    return YGama

def linearBetaBackward(Z,beta,softmaxBackward):
    # d(B)/d(Z) = beta,  d(B)/d(Z).shape = (K, D+1)
    # d(B)/d(beta) = Z,  d(B)/d(beta).shape = (K, D+1)
    # linearBetaBackward output[0] d(loss)/d(Z).shape = (D+1,1)
    # linearBetaBackward output[1] d(loss)/d(beta).shape = (K,D+1)
    Z = Z.reshape( -1,1 )
    softmaxBackward = softmaxBackward.reshape(-1,1)
    return np.matmul(np.transpose(beta),softmaxBackward),np.matmul(softmaxBackward,np.transpose(Z))

def sigmoidBackward(Z,linearBetaBackward):
    # d(Z)/d(A) = z(1-z)
    # drop the bias term z0
    # sigmoidBackward output d(loss)/d(A).shape = (D,1)
    linearBetaBackward = linearBetaBackward.reshape(1,-1)
    inter = Z*(1-Z).reshape(1,-1)
    return (linearBetaBackward*inter).reshape(-1,1)[:-1]

def linearAlphaBackward(X,sigmoidBackward):
    # d(A)/d(alpha) = X, X.shape = (M+1,1)
    # linearAlphaBackward output d(loss)/d(alpha).shape = (D,M+1)
    X = X.reshape( 1,-1 )
    return np.matmul(sigmoidBackward,X)

#check backpop gradient with finite different method
def finite_diff(x,y,theta):
    epsilon = 1e-5
    grad = np.zeros(len(theta))
    for m in range(1,len(theta)+1):
        d = np.zeros(m)
        d[m] = 1
        #v = forward(x,y,theta + epsilon * d
        #v -= forward(x,y,theta + epsilon * d)
        #v /= 2*epsilon
        #grad[m] = v
    return grad

def read_data(infile: str):
    contents = pd.read_csv(infile,header=None, index_col=False)
    # fold bias feature 1 as x[-1] -> to the last column
    contents[contents.shape[1] + 1] = [1] * contents.shape[0]
    return np.array(contents)

def NNForward (x,Y,alpha,beta):
    A = linearAlphaForward( x,alpha )
    Z = sigmoidForward( A )
    B = linearBetaForward( Z,beta )
    YGama = softmax( B )
    return A,Z,B,YGama

def NNBackward(A,Z,B,YGama,Y,alpha,beta,x):
    dB = softmaxBackward( Y,YGama)
    dZ,dbeta = linearBetaBackward( Z,beta,dB )
    dA = sigmoidBackward( Z,dZ )
    dalpha = linearAlphaBackward( x,dA )
    return dalpha,dbeta

def get_prediction(alpha,beta,contents):
    pred = np.array([])
    true = np.array([])
    entropy = np.array([])
    for entry in contents:
        x = entry[1:]
        Y = int( entry[0] )
        A,Z,B,YGama = NNForward (x,Y,alpha,beta)
        prediction = np.argmax(YGama)
        pred = np.append(pred,[prediction])
        true = np.append(true,[Y])
        entropy = np.append(entropy,[-np.log(YGama[Y])])
    cross_entropy = np.average(entropy)
    error_rate = float(np.sum(pred - true != 0)/len(true))
    return error_rate,cross_entropy

def train(contents,test_contents,epoch,D,initial,gama):
    M = contents.shape[1] - 2
    K = 10
    errors = np.array([])
    cross_entropies =  np.array([])
    errors_test =  np.array([])
    cross_entropies_test =  np.array([])
    # set up initial alpha, beta with 2 strategies
    # initital strategy 0: zeros for all alpha and beta
    # initial strategy 1: randomly sample from uniform distribution range from -0.1 to 0.1
    if initial == 1:
        alpha = np.array( [np.zeros( M+1 ) ]* D )
        beta = np.array( [np.zeros( D+1 ) ] * K )
    elif initial == 0:
        alpha = np.random.rand( D,M+1 )*0.2-0.1
        beta = np.random.rand( K, D + 1 )*0.2-0.1
    for ep in range(epoch):
        #train alpha, beta by SGD
        for entry in contents:
            x = entry[1:]
            Y = int( entry[0] )
            A,Z,B,YGama = NNForward (x,Y,alpha,beta)
            dalpha,dbeta = NNBackward( A,Z,B,YGama,Y,alpha,beta,x )
            alpha -= gama*dalpha
            beta -= gama*dbeta
        #predict with trained alpha, beta
        error_rate, cross_entropy = get_prediction( alpha,beta,contents )
        print( "Epoch {} training cross entropy is {}".format( ep+1,cross_entropy ) )
        error_rate_test,cross_entropy_test = get_prediction( alpha,beta,test_contents)
        print( "Epoch {} testing cross entropy is {}".format( ep+1,cross_entropy_test ) )
        errors = np.append( errors,[error_rate] )
        cross_entropies = np.append( cross_entropies, [cross_entropy] )
        errors_test = np.append( errors_test,[error_rate_test] )
        cross_entropies_test = np.append( cross_entropies_test,[cross_entropy_test] )
    print( "training error rate is {}".format( errors[-1] ) )
    print( "testing error rate is {}".format(  errors_test[-1] ) )
    return errors, cross_entropies,errors_test,cross_entropies_test


if __name__ == "__main__":
    start_time = time.time()
    initial = 0
    D = 50
    gama = 0.1
    train_contents = read_data('largeTrain.csv')
    test_contents = read_data('largeTest.csv')
    epoch = 100
    train_error,train_cross_entrpy,test_error,test_cross_entrpy = train(train_contents,test_contents,epoch,D,initial,gama)
    plt.xlabel("Epoch")
    plt.ylabel("Average Entropy")
    x_axis = np.linspace(0, epoch - 1, epoch)
    plt.plot(x_axis, train_cross_entrpy,  linewidth=2.0, label='Train')
    plt.plot(x_axis, test_cross_entrpy,  linewidth=2.0, label='Test')
    plt.title('Average Entropy versus Training Epoch (learning rate = {})'.format(gama))
    plt.legend(loc='upper right')
    #plt.show()
    plt.savefig('large-{}epoch-{}gama.png'.format(epcho,gama))
    print( "--- %s minutes ---" % ((time.time() - start_time)/60.0))
