import numpy as np
import math
from numpy.core.fromnumeric import argmax, shape
from numpy import asarray
from numpy import save
from numpy import load

class BaseNeuron():
    
    #sigmoid activation function
    def ActivationFunction(self,x):
        return (1.0/(1.0+np.exp(-x)))

    #derivative of the sigmoid func
    def ActivationFunctionDerivative(self,x):
        return x*(1-x)

class Neuron(BaseNeuron):
    def __init__(self,nWeights,bias,layer,id):
        self.layer = layer
        self.id = id
        self.weights = nWeights
        self.bias = bias
        self.summations = []
        self.output = 0.0
        self.gradient = 0.0
        self.delta = 0.0

    #Can be referred to as the weighted sum of the weights, plus activations of the previous layer and then plus a bias.
    #use the dot product to speed up the calculation.
    def Integrate(self, input):
        dotProd = np.dot(self.weights,input) + self.bias
        self.summations = dotProd
        return dotProd
    
    #run the value through an activation function that squishes the value into the integer range we care about.
    def Activate(self,act):
        self.output = super().ActivationFunction(self.summations)
        return self.output
    
    #runs the value through the derivative of the action function.
    def ActivationDerivative(self,x):
        return super().ActivationFunctionDerivative(x)
    
    #Update the weight of this conneciton based on our learning rate, the delta which is the descent slope, and the input
    def UpdateWeight(self,i,lr,input):
        #weight = weight - learning_rate * error * input
        self.weights[0][i] = self.weights[0][i] - lr * self.delta * input

class ANN():
    def __init__(self,inputData,epoch,features,networkLayout,networkLocation,learningRate=0.5):
        self.rootData = inputData
        self.networkLayout = networkLayout
        self.classes = inputData[:,0]
        self.classes = self.classes.reshape(self.classes.size,1)
        self.data = inputData[:,1:]
        self.learningRate = learningRate
        self.epoch = epoch
        #how many times we want to train our ANN
        self.epoch = epoch
        self.features = features
        self.layers = []
        self.networkLocation = networkLocation
        if networkLocation == "":
            self.CreateLayers(networkLayout)
        else:
            #create
            self.LoadNetwork()
        #we have 54 independent variables, start off with random values and 2 neurons
        self.iWeights0 = np.reshape(np.random.rand(self.features,1),(1,54))
        self.iWeights1 = np.reshape(np.random.rand(self.features,1),(1,54))
        #8 neurons
        self.hWeights0 = np.random.rand(2,1)
        self.hBiases = np.random.rand(2,1)
        #self.hWeights1 = np.random.rand(self.data.shape[1],1)
        #output weights, 2 neurons coming in.
        self.oWeights = np.reshape(np.random.rand(2,1),(1,2))
        self.oBias = np.random.rand(1,1)
        #start with a random value for bias
        self.biash = np.random.rand(1)
        #start with a random value for bias
        self.biaso = np.random.rand(1)
    
    def SaveNetwork(self):
        asarrays = []
        for l in self.layers:
            lWeights = []
            for n in l:
                lWeights.append([*n.weights,n.bias])
            asarrays.append(lWeights)
        save("data.npy",asarrays)

    def LoadNetwork(self):
        data = load("data.npy",allow_pickle=True)
        di = 0
        l = 0
        neurons = []
        for n in self.networkLayout:
            if l == 0:
                inputs = []
            else:
                inputs = self.networkLayout[l-1]
            dr = data[l]
            for i in range(n):
                neuron = Neuron(np.reshape(dr[i][0],(1,len(dr[i][0]))),dr[i][1],l,f"N{i}")
                di += 1
                #neuron = Neuron(np.reshape(np.random.rand(inputs,1),(1,inputs)),np.random.rand(1),1,f"N{i}")
                neurons.append(neuron)
            self.layers.append(neurons.copy())
            neurons.clear()
            l+=1

    def CreateLayers(self,networkLayout):
        #inputs
        #networkLayout is an array of tuples containing Layer,neurons
        #hidden layer 1
        neurons = []
        l = 0
        for n in networkLayout:
            if l == 0:
                inputs = self.features
            else:
                inputs = networkLayout[l-1]
            for i in range(n):
                neuron = Neuron(np.reshape(np.random.rand(inputs,1),(1,inputs)),np.random.rand(1),1,f"N{i}")
                neurons.append(neuron)
            self.layers.append(neurons.copy())
            neurons.clear()
            l+=1
    
    def MeanSquaredError(self,classes,predicted):
        sumSquareError = 0.0
        for i in range(len(classes)):
            sumSquareError += (classes[i] - predicted[i])**2.0
        meanSError = 1.0/len(classes)*sumSquareError
        return meanSError

    def FeedForward(self,inData):
        data = inData
        for layer in self.layers:
            oData = []
            for neuron in layer:
                output = neuron.Integrate(data)
                neuronActivation = neuron.Activate(output)
                oData.append(*neuronActivation)
            data = oData
        return data

    def BackPropagation(self,expected):
        #derivative tells us the sensitivity of the weight to change 
        #The gradient tells us how much the parameter needs to change, and in which direction, to minimise loss
        #can compute the gradient by using the chain rule.
        i = len(self.layers)-1
        #go layer by layer
        while(i >= 0):
            errors = []
            layer = self.layers[i]
            
            #check for the output layer
            if i == len(self.layers)-1:
                #for each neuron in this layer, which is the output, we look at the neurons that feed into it.
                for y in range(len(layer)):
                    neuron = layer[y]
                    #looking at the error in relation to the expected output!
                    c0 = (neuron.output - expected)*2
                    errors.append(c0)
            else:
                #for each neuron in this lear
                for y in range(len(layer)):
                    error = 0.0
                    #for each neuron in the previous layer
                    for n in self.layers[i+1]:
                        #update the error for each neuron
                        error += n.weights[0][y] * n.delta
                    
                    errors.append(error)

            #update the delta for each neuron based on the errors of the previous layer and the activation function derivative.
            for y in range(len(layer)):
                neuron = layer[y]
                neuron.delta = errors[y] * neuron.ActivationDerivative(*neuron.output)
            i -= 1

    def UpdateNetworkWeights(self,row,lr):
        for l in range(len(self.layers)):
            inputs = row
            #i == 0 would be the input layer which does not have a previous!
            if l != 0:
                #inputs are the outputs of the layer before
                inputs = [n.output for n in self.layers[l-1]]
                #for each neuron
            for n in self.layers[l]:
                #for each input to the network
                for i in range(len(inputs)):
                    #update the weight of the neuron based on the output of the neuron and the input that caused it and the learning rate
                    n.UpdateWeight(i,lr,inputs[i])
                n.bias = n.bias - lr * 1 * n.delta
                    
    def Train(self):
        print(f"network layout is {self.networkLayout} with learning rate {self.learningRate}")
        for e in range(self.epoch):
            i = 0
            total = 0
            correct = 0
            mse = 0
            predictions = []
            for row in self.data:
                predicted = self.FeedForward(row)
                #The error between the expected outputs and the outputs forward propagated from the network.
                total += 1
                if(predicted[0] > 0.5):
                    predicted = 1
                else:
                    predicted = 0
                if predicted  == self.classes[i]:
                    correct += 1
                predictions.append(predicted)
                expected = self.classes[i]
                self.BackPropagation(*self.classes[i])
                self.UpdateNetworkWeights(row,self.learningRate)
                i += 1
            accuracy = (correct/total)*100.0
            mse = self.MeanSquaredError(self.classes,predictions)
            print('epoch>%d,MSE=%.3f,Accuracy=%.3f' % (e, mse,accuracy))
        self.SaveNetwork()
        
    def Predict(self,data):
        classPredictions = []
        for row in data:
            prediction = self.FeedForward(row)
            if(prediction[0] > 0.5):
                prediction = 1
            else:
                prediction = 0
            classPredictions.append(prediction)
        return classPredictions

def MainTraining():
    training_spam = np.loadtxt(open("training_spam.csv"), delimiter=",").astype(np.int)
    trainingDataProportion = int(training_spam.shape[0] *0.7)
    testingDataProportion = int(training_spam.shape[0]*0.3)
    trainingData = training_spam[:trainingDataProportion]
    testingData = training_spam[trainingDataProportion:]

    classifier = ANN(trainingData,100,54,(12,12,1),"",0.8)


    classifier.Train()
    classifier.SaveNetwork()

    true_classes = testingData[:,0]
    predictions = classifier.Predict(testingData[:,1:])
    print(predictions)
    testingAccuracy = np.mean(np.equal(predictions,true_classes))
    print(f"Accuracy on the testing set {testingAccuracy}")
