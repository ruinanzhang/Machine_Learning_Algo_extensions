'''
Description: Script to train the network and measure its performance on the test set.

Author: Alejandro Escontrela
Version: V.1.
Date: June 12th, 2018
'''
from CNN.network import *
from CNN.utils import *

from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
parser = argparse.ArgumentParser(description='Train a convolutional neural network.')
parser.add_argument('save_path', metavar = 'Save Path', help='name of file to save parameters in.')

if __name__ == '__main__':
    
    digits = load_digits()
    img_dim = 28
    m =200
    X_train = extract_data('t10k-images-idx3-ubyte.gz', m, img_dim)
    y_train = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
    args = parser.parse_args()
    save_path = args.save_path
    print("save path is: " + save_path)
    cost = train(X = X_train, y_dash = y_train, save_path = save_path,img_dim = img_dim)

    params, cost = pickle.load(open(save_path, 'rb'))
    [f1, f2, w3, w4, b1, b2, b3, b4] = params
    
    # Plot cost
    """
    plt.plot(cost, 'r')
    plt.xlabel('# Iterations')
    plt.ylabel('Cost')
    plt.legend('Loss', loc='upper right')
    plt.show()
    """
    # Get test data
    m =200
    X_test = extract_data('t10k-images-idx3-ubyte.gz', m, img_dim)
    y_test = extract_labels('t10k-labels-idx1-ubyte.gz', m).reshape(m,1)
    # Normalize the data
    X = X_test
    y_dash = y_test
    X-= int(np.mean(X)) # subtract mean
    X/= int(np.std(X)) # divide by standard deviation
    test_data = np.hstack((X,y_dash))
    
    X = test_data[:,0:-1]
    X = X.reshape(len(test_data), 1,img_dim, img_dim)
    y = test_data[:,-1]

    corr = 0
    digit_count = [0 for i in range(10)]
    digit_correct = [0 for i in range(10)]
   
    print()
    print("Computing accuracy over test set:")

    t = tqdm(range(len(X)), leave=True)

    for i in t:
        x = X[i]
        pred, prob = predict(x, f1, f2, w3, w4, b1, b2, b3, b4)
        digit_count[int(y[i])]+=1
        if pred==y[i]:
            corr+=1
            digit_correct[pred]+=1

        t.set_description("Acc:%0.2f%%" % (float(corr/(i+1))*100))
        
    print("Overall Accuracy: %.2f" % (float(corr/len(test_data)*100)))
    x = np.arange(10)
    print(digit_correct)
    print(digit_count)
    digit_recall = [x/y for x,y in zip(digit_correct, digit_count)]
    plt.xlabel('Digits')
    plt.ylabel('Recall')
    plt.title("Recall on Test Set")
    plt.bar(x,digit_recall)
    plt.show()
