import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

batch_size = 128 # Number of images processed at once
nb_classes = 10  # 10 Digits from 0 to 9

# Dimensionen of the input images (28x28 pixel)
img_rows, img_cols = 28, 28

# Load image data with labels, split into test and training set 
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# reshape images in 4D tensor (N images, 28 rows, 28 columns, 1 channel) 
# rescale pixels range from [0, 255] to [0, 1]
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1).astype("float32")
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1).astype("float32")

X_train /= 255
X_test /= 255

print('X_train shape: ', X_train.shape)
print(X_train.shape[0], "training samples")
print(X_test.shape[0], "test samples")

# convert digit labels (0-9) in one-hot encoded binary vectors. 
# These correspond to the training/test labels at the output of the net. 
Y_train = tf.keras.utils.to_categorical(y_train, nb_classes)
Y_test = tf.keras.utils.to_categorical(y_test, nb_classes)
print("One-hot encoding: {}".format(Y_train[0, :]))

# Show bitmaps of the first 9 trainings images
for i in range(9):
    plt.subplot(3,3, i+1)
    plt.imshow(X_train[i, :, :, 0], cmap='gray')
    plt.axis('off')

# Define LeNet-5 model
model = tf.keras.Sequential()

# Conv2D(number_filters, kernel_size, input_shape=(number_channels, img_col), padding, activation)
# model.add(tf.keras.layers.Conv2D(6, (5, 5), input_shape=[img_rows, img_cols, 1], padding='same', activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu'))
# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# model.add(tf.keras.layers.Conv2D(120, (5, 5), activation='relu'))
# #model.add(tf.keras.layers.Dropout(0.25))

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(84, activation='relu'))
# model.add(tf.keras.layers.Dropout(0.5))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))

# model.compile(loss='categorical_crossentropy', optimizer='SGD', metrics=['accuracy']) #adadelta

model.add(tf.keras.layers.Conv2D(6, (5, 5), input_shape=[img_rows, img_cols, 1], padding='same', activation='tanh'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(tf.keras.layers.Dropout(0.25)) # Add dropout after pooling
model.add(tf.keras.layers.Conv2D(16, (5, 5), strides=(1, 1), activation='tanh', padding='valid'))
model.add(tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(tf.keras.layers.Dropout(0.25)) # Add dropout after pooling
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(120, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5)) # Add dropout after fully connected layer
model.add(tf.keras.layers.Dense(84, activation='tanh'))
model.add(tf.keras.layers.Dropout(0.5)) # Add dropout after fully connected layer
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

nb_epoch = 30 # Number of passes over all pictures of the training set

hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, 
          verbose=1, validation_data=(X_test, Y_test))

score =  model.evaluate(X_test, Y_test, verbose=0)
print('Test score', score[0])
print('Test accuracy', score[1])

predictions = model.predict(X_test[:9])
res = np.argmax(predictions, axis=1)
plt.figure(figsize=(10, 10))

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[i, :, :, 0], cmap='gray')
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    plt.ylabel("Prediction = {}".format(res[i]), fontsize=18)

plt.savefig("Predictions.png")
plt.show()

test_score = model.evaluate(X_test, Y_test, verbose=0)
print("Test loss {:.4f}, accuracy {:.2f}%".format(test_score[0], test_score[1] * 100))

f, ax = plt.subplots()
ax.plot([None] + hist.history["accuracy"], "o-")
ax.plot([None] + hist.history["val_accuracy"], "x-")
# Plot legend and use the best location automatically: loc = 0.
ax.legend(["Train acc", "Validation acc"], loc = 0)
ax.set_title("Training / Validation Accuracy per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("acc")
plt.savefig("Training and Validation Accuracy per Epoch.png")

f, ax = plt.subplots()
ax.plot([None] + hist.history["loss"], "o-")
ax.plot([None] + hist.history["val_loss"], "x-")
# Plot legend and use the best location automatically: loc = 0.
ax.legend(["Train Loss", "Validation Loss"], loc = 0)
ax.set_title("Training / Validation Loss per Epoch")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.savefig("Training and Validation Loss per Epoch.png")
plt.show()