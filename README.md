<h1> ANN-Spam-Classification </h1>
<h2> An ANN approach to the classification of spam email messages.</h2>

<p> Given a set of testing data, with a bag of words encoding on each email message. We end up with a matrix of messages and their features, as binary encoded values. That is, for each email message, a feature x will be encoded to a 1 if it is present in an email message. Each email has 54 features, the first column of data being the class (0,1) indicating that it is spam or not. </p>

The approach was to create an Artificial Neural Network, to handle the classification on messages. To do this, the ANN was coded from scratch in Python.
A few key functions:
<ol>
<li>Feed Foward: This function feeds the input values through each layer of the network, calculation weighted sums, and running the activation functions to squish our output values into a smaller integer space.</li>
<li>Loss Function: a integral as it tells us the average error rate from all of our processing runs so far. As part of training we must reduce our error rate using Gradient Descent. Mean Squared Error was the function used to calculate.</li>
<li>Backpropagation: a process in which we calculate the error of each neuron to find out the gradient we must nudge our weights and bias in, to reduce the MSE of our network.</li>
<li>Update Network Weights: integral part which allows us to update the weights and biases of every neuron in the network.</li>
</ol>


<h1>Results</h1>
![ANN Results](https://github.com/the1ullneversee/ANN-Spam-Classification/blob/main/Results.png.jpg)
