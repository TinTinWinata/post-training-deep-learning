import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

(train_data, _), (test_data, _) =  tf.keras.datasets.mnist.load_data()
train_data = train_data/np.float32(255)
train_data = np.reshape(train_data, (train_data.shape[0], 784))

test_data = test_data/np.float32(255)
test_data = np.reshape(test_data, (test_data.shape[0], 784))

class Restricted_Boltzmann_Machines(object):
    def __init__(self, input_size, output_size, learning_rate=1.0, batch_size=100):
        self._input_size = input_size
        self._output_size = output_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.w = tf.zeros([input_size, output_size], np.float32)
        self.hb = tf.zeros([output_size], np.float32)
        self.vb = tf.zeros([input_size], np.float32)

    # Forward Pass
    def prob_h_given_v(self, visible, w, hb):
        return tf.nn.sigmoid(tf.matmul(visible, w) + hb)

    # Backward Pass
    def prob_v_given_h(self, hidden, w, vb):
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(w)) + vb)

    # Generate the sample probability
    def sample_prob(self, probs):
        return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))

    # Training method for the model
    def train(self, X, epochs=10):

        loss = []
        for epoch in range(epochs):
            # For each step or batch
            for start, end in zip(range(0, len(X), self.batch_size), range(self.batch_size,len(X), self.batch_size)):
                batch = X[start:end]

                # Initialize with sample probabilities
                h0 = self.sample_prob(self.prob_h_given_v(batch, self.w, self.hb))
                v1 = self.sample_prob(self.prob_v_given_h(h0, self.w, self.vb))
                h1 = self.prob_h_given_v(v1, self.w, self.hb)

                # Create the Gradients
                positive_grad = tf.matmul(tf.transpose(batch), h0)
                negative_grad = tf.matmul(tf.transpose(v1), h1)

                # Update learning rates
                self.w = self.w + self.learning_rate *(positive_grad - negative_grad) / tf.dtypes.cast(tf.shape(batch)[0],tf.float32)
                self.vb = self.vb +  self.learning_rate * tf.reduce_mean(batch - v1, 0)
                self.hb = self.hb +  self.learning_rate * tf.reduce_mean(h0 - h1, 0)

            # Find the error rate
            err = tf.reduce_mean(tf.square(batch - v1))
            print ('Epoch: %d' % epoch,'Reconstruction error / loss: %f' % err)
            loss.append(err)

        return loss

    # Create expected output for our DBN
    def rbm_output(self, X):
        out = tf.nn.sigmoid(tf.matmul(X, self.w) + self.hb)
        return out

    def rbm_reconstruct(self,X):
        h = tf.nn.sigmoid(tf.matmul(X, self.w) + self.hb)
        reconstruct = tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.w)) + self.vb)
        return reconstruct

# Size of inputs is the number of inputs in the training set
input_size = train_data.shape[1]

rbm_model = Restricted_Boltzmann_Machines(input_size, 200)
err = rbm_model.train(train_data, 50)

plt.plot(err)
plt.xlabel('Epochs')
plt.ylabel('Reconstruction Cost')
plt.savefig("Reconstruction Cost of RBM Model.png")
plt.show()

out = rbm_model.rbm_reconstruct(test_data)

# Plotting original and reconstructed images
row, col = 2, 8
idx = np.random.randint(0, 100, row * col // 2)
f, axarr = plt.subplots(row, col, sharex=True, sharey=True, figsize=(20,4))
for fig, row in zip([test_data,out], axarr):
    for i, ax in zip(idx,row):
        ax.imshow(tf.reshape(fig[i],[28, 28]), cmap='Greys_r')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

plt.tight_layout()
plt.savefig("Original and Reconstructed Image.png")
plt.show()