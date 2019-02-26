import training
import tensorflow as tf

image_size = 28
labels_size = 10
hidden_size = 1024  # 1024 "neurons"

# Define placeholders
# Input tensor - flatten the matrix of pixels from the digit image
training_data = tf.placeholder(tf.float32, [None, image_size * image_size])
# Output tensor
labels = tf.placeholder(tf.float32, [None, labels_size])

# Define another dense, hidden layer
# (dense = each receives inputs from all neurons from previous layer)
W_h = tf.Variable(tf.truncated_normal(
    [image_size * image_size, hidden_size], stddev=0.1))
b_h = tf.Variable(tf.constant(0.1, shape=[hidden_size]))

# Hidden layer with reLU activation function
hidden = tf.nn.relu(tf.matmul(training_data, W_h) + b_h)

# Variables for the output layer
W = tf.Variable(tf.truncated_normal([hidden_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Connect hidden to output layer
output = tf.matmul(hidden, W) + b

# Train & test the network
training.train_network(training_data, labels, output)
