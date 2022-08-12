import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import copy
from datetime import datetime
np.random.seed(6)

# AKTIVACIJSKE FUNKCIJE IN GRADIENTI --------------------------------

def tanh(x):
    return np.tanh(x);

def tanh_prime(x):
    return np.diag((1-np.tanh(x)**2).reshape(-1));

def softmax(x):
    z = x - np.tile(np.max(x,axis = 1),(x.shape[1],1)).transpose() #numerična stabilnost
    E = np.exp(z)
    e =np.sum(E,axis = 1).reshape(-1,1)
    return E / (e + 1e-10)

def softmax_prime(x):
    _,n = x.shape
    kronecker_delta = np.eye(n)
    
    s_i = softmax(x) 
    s_j = np.tile(s_i,(n,1)).transpose()
    return s_j*(kronecker_delta-s_i)


# LOSS FUNKCIJE IN ODVODI --------------------------------

def mse(real, predictions):
    return np.mean(np.power(real - predictions, 2));

def mse_prime(real, predictions):
    return 2*(predictions-real)/real.size;

def log_loss(real, predictions):

    eksp = np.log(predictions)
    eksp = eksp[np.arange(len(real)),real]
    final_cost = np.sum(eksp) 
    return -final_cost/len(predictions)

def log_loss_prime(real, predictions):
    grad = np.zeros(predictions.shape)
    grad[np.arange(predictions.shape[0]),real] = -1/(predictions[np.arange(predictions.shape[0]),real]+1e-10)
    return grad #/predictions.shape[1]
    #ne pozabi spremeniti tega
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        raise NotImplementedError
   
    def backward_propagation(self, output_error, learning_rate,lambda_):
        raise NotImplementedError 

class LinearLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate,lambda_):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * (weights_error + lambda_ * self.weights)
        self.bias -= learning_rate * output_error
        return input_error

class ActivationLayer(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward_propagation(self, output_error, learning_rate,lambda_):
        return np.dot( output_error,self.activation_prime(self.input) )

class Model:
    def __init__(self, input_sizes, output_sizes, lambda_):
        self.loss = None
        self.loss_prime = None
        self.lambda_ = None
        self.all_layers = None
    
    def predict(self, X_test):
        output = X_test
        for layer in self.all_layers:
            output = layer.forward_propagation(output)
        if(output.shape[1] == 1):
            output = output.reshape(-1)
        return output

    def forward(self, X_train_sample):
        output = X_train_sample.reshape(1,-1)
        for layer in self.all_layers:
            output = layer.forward_propagation(output)
        return output
        
    def backward(self, y_train_sample,output,learning_rate):
        error = self.loss_prime(y_train_sample, output)
        for layer in reversed(self.all_layers):
            error = layer.backward_propagation(error, learning_rate,self.lambda_)

    def weights(self):
        weights = []
        n = len(self.all_layers)
        for i in range(0,n,2):
            layer = self.all_layers[i]
            weights.append(np.vstack((layer.weights,layer.bias)))
        return weights

class RegressionModel(Model):
    def __init__(self, input_sizes, output_sizes, lambda_):
        self.loss = mse
        self.loss_prime = mse_prime
        self.lambda_ = lambda_

        #layers
        self.all_layers = []
        for in_,out_ in zip(input_sizes[:-1], output_sizes[:-1]):
            self.all_layers.append(LinearLayer(in_,out_))
            self.all_layers.append(ActivationLayer(tanh,tanh_prime))
        self.all_layers.append(LinearLayer(input_sizes[-1],output_sizes[-1]))
        
class ClassificationModel(Model):
    def __init__(self, input_sizes, output_sizes, lambda_):
        self.loss = log_loss
        self.loss_prime = log_loss_prime
        self.lambda_ = lambda_
       
        #layers
        self.all_layers = []
        for in_,out_ in zip(input_sizes[:-1], output_sizes[:-1]):
            self.all_layers.append(LinearLayer(in_,out_))
            self.all_layers.append(ActivationLayer(tanh,tanh_prime))

        self.all_layers.append(LinearLayer(input_sizes[-1],output_sizes[-1]))
        self.all_layers.append(ActivationLayer(softmax,softmax_prime))    

class ANNRegression:
    def __init__(self, units, lambda_ ):
        self.lambda_ = lambda_
        self.units = units
    
    def fit(self, X, y, epoch = 10000, learning_rate = 0.001, batch_size = 32,verbose = False,early_stopping = False):     
                                                                                                    #TODO ce bo cas dodaj da dela za bache
                                                                                                    #TODO early stoping z validation
        m,n= X.shape
        input_sizes = [n] + self.units
        output_sizes = self.units + [1]

        if early_stopping:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        best_score = 100000
        prev_error = 100000
        model = RegressionModel(input_sizes, output_sizes, self.lambda_)
        samples = len(X)
        
        for i in range(epoch):
            err = 0
            for j in range(samples):
                output = model.forward(X[j])
                err += mse([y[j]], output)
                model.backward(y[j], output, learning_rate)
            
            err /= samples
            if(verbose):
                print('epoch %d/%d   error=%f' % (i+1, epoch, err))

            if early_stopping:
                tren_score = mse(y_val, model.predict(X_val))
                if(tren_score < best_score):
                    best_score = tren_score
                    best_model = copy.deepcopy(model)
                  

                if(err + 1e-5 > prev_error):
                    break
            prev_error = err


        if early_stopping:
            return best_model, best_score            
        else:
            return model
    
class ANNClassification:
    def __init__(self, units, lambda_ ):
        self.lambda_ = lambda_
        self.units = units
    
    def fit(self, X, y, epoch = 100000, learning_rate = 0.009, batch_size = 32,verbose = False, early_stopping = False):      
                                                                                                    #TODO ce bo cas dodaj da dela za bache
                                                                                                    #TODO early stoping z validation
        c = len(set(y))
        m,n= X.shape

        input_sizes = [n] + self.units
        output_sizes = self.units + [c]

        if early_stopping:
            X, X_val, y, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
        best_score = 100000
        prev_error = 100000


        model = ClassificationModel(input_sizes, output_sizes, self.lambda_)     
        samples = len(X)
        for i in range(epoch):
            err = 0
            for j in range(samples):
                output = model.forward(X[j])
                err += log_loss([y[j]], output)
                model.backward(y[j], output, learning_rate)
            err /= samples
            
            if(verbose):
                print('epoch %d/%d   error=%f' % (i+1, epoch, err))

            if early_stopping:
                tren_score = log_loss(y_val, model.predict(X_val))
                if(tren_score < best_score):
                    best_score = tren_score
                    best_model = copy.deepcopy(model)
                    print("validation: " + str(tren_score))
                
                if(err + 1e-5 > prev_error):
                    break
            prev_error = err

        if early_stopping:
            return best_model, best_score            
        else:
            return model

def test_on_hausing_data_Classification():
    df = pd.read_csv("data/housing3.csv")
    y = np.int_(np.array(df.iloc[:,13].str.split("C").str[1]))
    y = y-1

    X = np.array(df.iloc[:,1:13])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    layer_configs = [[],[20],[20,10],[50,20,10],[70,50,20,10]]
    for units in layer_configs:
        fitter = ANNClassification(units=units, lambda_=0.0001)
        m ,s= fitter.fit(X_train, y_train, epoch = 300, learning_rate = 0.0001, verbose = False, early_stopping = True)
        pred = m.predict(X_test)
        print(log_loss(y_test,pred))
    

def test_on_hausing_data_Regression():
    df = pd.read_csv("data/housing2r.csv")
    y = np.array(df.iloc[:,5])

    X = np.array(df.iloc[:,1:5])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    layer_configs = [[],[20],[20,10],[50,20,10],[70,50,20,10]]
    for units in layer_configs:
        fitter = ANNRegression(units=units, lambda_=0.0001)
        m ,s= fitter.fit(X_train, y_train, epoch = 300, learning_rate = 0.0001, verbose = False, early_stopping = True)
        pred = m.predict(X_test)
        print(mse(y_test,pred))


def transform_data(train):
    y = []
    if(train):
        df = pd.read_csv('data/train.csv')
        y = np.int_(np.array(df.iloc[:,94].str.split("_").str[1]))
        y = y-1

    else:
        df = pd.read_csv('data/test.csv')






    X = np.array(df.iloc[:,1:94])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X,y


def create_final_file(y_test):

    y_test = pd.DataFrame(y_test).reset_index()
    y_test.columns = ['id',"Class_1","Class_2","Class_3","Class_4","Class_5","Class_6","Class_7","Class_8","Class_9",]
    y_test["id"] = y_test["id"]+1
    y_test.to_csv("final.txt", header=True, index=False, sep=',')


def create_final_predictions():
    X_train, y_train  = transform_data(train = True)
    X_test,y_test = transform_data(train = False)
    """
    lambdas = [0.01,0.001,0.0001]
    learning_rates = [0.01,0.001,0.0001]
    layer_configs = [[150,100,50,20],[100,50,20],[50,20],[20]]
    """
    best_lambda = 0.0001, 
    best_learning_rate = 0.01
    layer_config = [100,50,20]
   
                
    start_time = datetime.now()
    fitter = ANNClassification(units=layer_config, lambda_=best_lambda)
    m,score_on_val = fitter.fit(X_train, y_train, epoch = 150, learning_rate = best_learning_rate, verbose = True, early_stopping = True)
    
    print(score_on_val)
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

    y_test= m.predict(X_test)
    create_final_file(y_test)


def check_compatibility_of_loss_functions():
    return 0


if __name__ == "__main__":
    
    test_on_hausing_data_Classification()
    #test_on_hausing_data_Regression()
    #create_final_predictions()
