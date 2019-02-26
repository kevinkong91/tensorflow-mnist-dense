# Tensorflow Dense DNN on MNIST Data

## Generalized ML process
1. Read training & validation datasets
2. Define the neural net architecture and configs
3. Define the loss function
4. Define the optimizer algorithm
5. Train the neural net on batches of data
6. Evaluate the net's performance on the validation dataset

## Skills Applied
- Create dense, convolutional, max pooling hidden layers
- Connect the layers with flattening and reshaping techniques
- Apply dropout

## Techniques

### Dense Layer
`dense.py` includes a simple one output layer network.

### Hidden Layer
`hidden.py` adds a reLU-activated dense layer with its own weights and biases.

### Convolutional Layer
`convolutional.py` adds more layers of complexity. This method yields especially accurate results with 2- and more dimensions of input. However, the improved performance comes at the cost of speed: complexity of the network slows down the speed of training for improved accuracy.

![Model of the Convolutional Layer](https://katacoda.com/basiafusinska/courses/tensorflow-getting-started/tensorflow-mnist-expert/assets/convolution.png "Model of the Convolutional Layer")

The parameters of thelayer are the size of the convolution window and the strides.

*Padding* set as `'SAME'` indicates that the resulting layer is of the same size.

After this step, we apply *max pooling*. We will build two convolutional layers, connect it to the dense hidden layer. The above diagram visualizes this architecture.

## Concepts

### Dense Layer
A regular layer of neurons in a neural network. Each neuron receives input from *all the neurons* in the previous layer, thus **densely connected**. The layer has weight matrix **W**, a bias vector **b**, and the activations of previous layer **a**.

### Dropout Layer
A regularization technique used to tackle the problem of overfitting. The Dropout method takes in a random float between 0 and 1, **`keep_prob`**, which is the fraction of the neurons to drop during training time. It's very important to dropout the neurons *only in the training phase*, not when evaluating the model; thus we define an additional placeholder to keep the dropout probability.

A dropout layer does not have any traininable parameters; nothing gets updated during backward pass of backpropagation.

To ensure that expected sum of vectors fed to this layer remains the same if no dropout was applied, the remaining dimensions which are not set to zero are scaled by `1 / keep_prob`.