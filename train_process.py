import numpy as np
import pandas as pd

data = pd.read_csv("train/train.csv")
# print(data)
data = np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_train = data[1000:m].T
Y_train = data_train[1]
Y_train = Y_train.astype(int)
X_train = data_train[2:n]
X_train = X_train/255
_, m_train = X_train.shape


def init_params():
    character_set_size = 91
    W1 = np.random.rand(character_set_size, 288) - 0.5
    b1 = np.random.rand(character_set_size, 1) - 0.5
    W2 = np.random.rand(character_set_size, character_set_size) - 0.5
    b2 = np.random.rand(character_set_size, 1) - 0.5
    return W1, b1, W2, b2
def load_params():
    W1 = pd.read_csv("train/W1.csv")
    W1 = np.array(W1)
    b1 = pd.read_csv("train/b1.csv")
    b1 = np.array(b1)
    W2 = pd.read_csv("train/W2.csv")
    W2 = np.array(W2)
    b2 = pd.read_csv("train/b2.csv")
    b2 = np.array(b2)
    return W1[:, 1:], b1[:, 1:], W2[:, 1:], b2[:, 1:]
def batter_init_params():
    character_set_size = 91-48
    W1 = np.random.rand(character_set_size, 288) - 0.5
    b1 = np.random.rand(character_set_size, 1) - 0.5
    W2 = np.random.rand(character_set_size, character_set_size) - 0.5
    b2 = np.random.rand(character_set_size, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    # one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y = np.zeros((Y.size, 43))
    one_hot_Y[np.arange(Y.size), Y-48] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0) #二為矩陣中axis_0的每個最大值位置
#e.g. np.argmax([[1, 5, 3], [4, 2, 6]], 0) >> [1, 0, 1]; 說明: 1:(1<4, 所以回傳4的位置)/ 0:(5>2, 所以回傳5的位置)/ 1:(3<6, 所以回傳6的位置)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    # W1, b1, W2, b2 = init_params()
    W1, b1, W2, b2 = load_params()
    # W1, b1, W2, b2 = batter_init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if i % 50 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            # accuracy = get_accuracy(predictions, Y)
            accuracy = get_accuracy(predictions+48, Y)
            print("Accuracy:", accuracy)
            if(accuracy>0.9): break
    return W1, b1, W2, b2


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    # current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", chr(int(prediction)+48))
    print("Label: ", chr(int(label)))

# run function
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.05, 500)

pd.DataFrame(W1).to_csv("train/W1.csv")
pd.DataFrame(b1).to_csv("train/b1.csv")
pd.DataFrame(W2).to_csv("train/W2.csv")
pd.DataFrame(b2).to_csv("train/b2.csv")


test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)

# dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
# get_accuracy(dev_predictions, Y_dev)

