import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def train_network(training_data, labels, output, keep_prob=tf.placeholder(tf.float32)):
    # Define the net architecture
    learning_rate = 1e-4
    steps_number = 1000
    batch_size = 100

    # Read the data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # Define the objective/loss function
    loss = tf.reduce(tf.nn.softmax_cross_entropy_with_logits(
        labels=labels, logits=output))

    # Training step
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Accuracy calculation
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
    predictions_as_float = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(predictions_as_float)

    # Run the training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for i in range(steps_number):
        # Poll next batch
        input_batch, labels_batch = mnist.train.next_batch(batch_size)

        # Construct inputs
        training_data = {
            training_data: input_batch,
            labels: labels_batch,
            keep_prob: 1.0
        }

        # Print accuracy progress
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict=training_data)
            print("STEP %d - Training batch accuracy: %g %%" %
                  (i, train_accuracy * 100))

        # Run the training step
        training_data['keep_prob'] = 0.5
        train_step.run(feed_dict=training_data)

    print("FINISHED TRAINING")

    # Evaluate the test set
    test_data = {
        training_data: mnist.test.images,
        labels: mnist.test.labels,
        keep_prob: 1.0
    }
    test_accuracy = accuracy.eval(feed_dict=test_data)
    print("TEST ACCURACY: %g %%" % test_accuracy * 100)
