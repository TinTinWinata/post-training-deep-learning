import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pandas as pd 
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Load data from CSV
df = pd.read_csv('train.csv')
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataframe

# Splitting the data into train, validation, and test
train_size = int(0.8 * len(df))
val_size = int(0.1 * len(df))

train_df = df[:train_size]
val_df = df[train_size:train_size + val_size]
test_df = df[train_size + val_size:]

text_col = 'description'
label_col = 'label'

# Other information about the data
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']

def subtract_one_label(text, label):
    return text, label - 1

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Convert dataframes to tf.data.Datasets
train_data = tf.data.Dataset.from_tensor_slices((train_df[text_col].values, train_df[label_col].values))
train_data = train_data.map(subtract_one_label)

val_data = tf.data.Dataset.from_tensor_slices((val_df[text_col].values, val_df[label_col].values))
val_data = val_data.map(subtract_one_label)

test_data = tf.data.Dataset.from_tensor_slices((test_df[text_col].values, test_df[label_col].values))
test_data = test_data.map(subtract_one_label)

# Preprocess the dataset
BUFFER_SIZE = 10000
BATCH_SIZE = 64
MAX_LENGTH = 256

# Define a function to encode text using the BERT tokenizer
def encode_text(text, label):
    encoded_text = tokenizer.encode(text.numpy().decode('utf-8'), add_special_tokens=True,
                                    max_length=MAX_LENGTH, truncation=True, padding='max_length')
    return encoded_text, label

# Update the tf.data.Dataset map functions
train_data = train_data.map(lambda text, label: tf.py_function(encode_text, [text, label], [tf.int64, tf.int64]))
val_data = val_data.map(lambda text, label: tf.py_function(encode_text, [text, label], [tf.int64, tf.int64]))
test_data = test_data.map(lambda text, label: tf.py_function(encode_text, [text, label], [tf.int64, tf.int64]))

# Pad and batch datasets
train_data = train_data.padded_batch(64, padded_shapes=([-1], []))
val_data = val_data.padded_batch(64, padded_shapes=([-1], []))
test_data = test_data.padded_batch(64, padded_shapes=([-1], []))

# for i, (text, label) in enumerate(train_data.take(4)):
#     print(f"Sample news {i}\n \
#     Label: {label.numpy()} {class_names[label.numpy()]}\n \
#     Description: {text.numpy().decode('utf-8')}\n----------\n")

# Transformer block
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# Building the model
embed_dim = 128
num_heads = 2
ff_dim = 32

# Note: The VOCAB_SIZE should now be the size of the BERT vocabulary. 
VOCAB_SIZE = tokenizer.vocab_size

inputs = layers.Input(shape=(MAX_LENGTH,))
embedding_layer = layers.Embedding(input_dim=VOCAB_SIZE, output_dim=embed_dim)(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(embedding_layer)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Training
history = model.fit(train_data, validation_data=val_data, epochs=10)

# Plot training & validation accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.savefig("Loss and Accuracy Graph Plot.png")
plt.tight_layout()
plt.show()

# Evaluation
test_loss, test_accuracy = model.evaluate(test_data)
print("Test Accuracy: ", test_accuracy)

def predict(model, sample_news, tokenizer, max_length, class_names):
    # Tokenize the sample news
    encoded_sample = tokenizer.encode(sample_news[0], add_special_tokens=True, max_length=max_length, truncation=True, padding='max_length')
    encoded_sample = np.array([encoded_sample])  # Convert the encoded sample to a 2D array

    # Predict the news type
    preds = model.predict(encoded_sample)
    pred_class = np.argmax(preds[0])

    print(f'Predicted class: {pred_class} \nPredicted Class name: {class_names[pred_class]}')

sample_news = ['Tesla, a self driving car company is also planning to make a humanoid robot. This humanoid robot appeared dancing in the latest Tesla AI day']
predict(model, sample_news, tokenizer, MAX_LENGTH, class_names)