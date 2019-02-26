# Tensorflow Dense DNN on MNIST Data

## Generalized ML process
1. Read training & validation datasets
2. Define the neural net architecture and configs
3. Define the loss function
4. Define the optimizer algorithm
5. Train the neural net on batches of data
6. Evaluate the net's performance on the validation dataset

## Techniques
1. `dense.py` includes a simple one output layer network.
2. `hidden.py` adds a reLU-activated dense layer with its own weights and biases.
3. `convolutional.py` adds another layer of complexity.

## Concepts

### Dense Layer
A regular layer of neurons in a neural network. Each neuron receives input from *all the neurons* in the previous layer, thus **densely connected**. The layer has weight matrix **W**, a bias vector **b**, and the activations of previous layer **a**.

### Dropout Layer
A regularization technique used to tackle the problem of overfitting. The Dropout method takes in a random float between 0 and 1, **`keep_prob`**, which is the fraction of the neurons to drop during training time. A dropout layer does not have any traininable parameters; nothing gets updated during backward pass of backpropagation.

To ensure that expected sum of vectors fed to this layer remains the same if no dropout was applied, the remaining dimensions which are not set to zero are scaled by `1 / keep_prob`.