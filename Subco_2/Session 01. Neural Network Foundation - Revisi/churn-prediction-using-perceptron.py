import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read data from CSV
data = pd.read_csv('churn.csv')

# Dropping non-numeric columns for simplicity
data = data.drop(columns=['customer_id', 'country', 'gender'])

# Splitting data into features and labels
X = data.drop(columns='churn')
y = data['churn']

# Standardizing the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Calculate the total number of samples in the original dataset
total_samples = len(data)

# Calculate proportions
train_proportion = len(X_train) / total_samples * 100
val_proportion = len(X_val) / total_samples * 100
test_proportion = len(X_test) / total_samples * 100

# Print the proportions
print(f"Training set: {train_proportion:.2f}% of the original dataset")
print(f"Validation set: {val_proportion:.2f}% of the original dataset")
print(f"Test set: {test_proportion:.2f}% of the original dataset")

# Single-layer Perceptron using Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=(X_train.shape[1],))
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)

# Making predictions on 5 samples from the test set
predictions = model.predict(X_test[:5])

# Thresholding the predictions to get class labels
predicted_labels = [1 if p >= 0.5 else 0 for p in predictions]

print("Predicted labels:", predicted_labels)
print("True labels:", list(y_test[:5]))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Plotting accuracy and loss
plt.figure(figsize=(10, 5))

# Plotting accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plotting loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('Single Layer Perceptron - Model Loss and Accuracy Graph.png')
plt.show()