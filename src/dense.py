import training
import tensorflow as tf

image_size = 28
labels_size = 10

# Define placeholders
training_data = tf.placeholder(tf.float32, [None, image_size * image_size])
labels = tf.placeholder(tf.float32, [None, labels_size])

# Variables to be "fitted"
W = tf.Variable(tf.truncated_normal(
    [image_size * image_size, labels_size], stddev=0.1))
b = tf.Variable(tf.constant(0.1, shape=[labels_size]))

# Build the network (only output layer)
output = tf.matmul(training_data, W) + b

# Train and test the network
training.train_network(training_data, labels, output)
