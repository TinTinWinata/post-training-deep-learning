import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import BertTokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

data = pd.read_csv('train.csv')
data = data.drop(columns=['label', 'title'])
sentences = data['description']

# # Create a toy dataset
# sentences = [
#     "The capital of Canada is Ottawa.",
#     "The capital of France is Paris.",
#     "The capital of Japan is Tokyo.",
#     "The capital of Indonesia is Jakarta.",
#     "The capital of Germany is Berlin.",
#     "The capital of India is New Delhi."
# ]

# Tokenize and preprocess using BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
input_sequences = [tokenizer.encode(s, add_special_tokens=True) for s in sentences]

# Find the max length for padding
max_sequence_len = max([len(seq) for seq in input_sequences])
# input_sequences = tokenizer(sentences.tolist(), add_special_tokens=True, padding='max_length', truncation=True, max_length=max_sequence_len, return_tensors='np')['input_ids']

# Prepare X and y
X = [seq[:-1] for seq in input_sequences]
y = [seq[1:] for seq in input_sequences]

# Padding sequences
X = pad_sequences(X, maxlen=max_sequence_len-1, padding='post', truncating='post')
y = pad_sequences(y, maxlen=max_sequence_len-1, padding='post', truncating='post')

# y_categorical = [to_categorical(i, num_classes=len(tokenizer.vocab)) for i in y]
# y_categorical = np.array(y_categorical)

def scaled_dot_product_attention(query, key, value, mask):
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    d_k = tf.cast(tf.shape(key)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(d_k)
    
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    
    return output, attention_weights

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, query, key, value, mask):
        batch_size = tf.shape(query)[0]

        query = self.split_heads(self.wq(query), batch_size)
        key = self.split_heads(self.wk(key), batch_size)
        value = self.split_heads(self.wv(value), batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        
        return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])

def positional_encoding(position, d_model):
    angle_rads = 1 / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / np.float32(d_model))
    angle_rads[:, 0::2] = np.sin(position * angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.sin(position * angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return tf.cast(pos_encoding, dtype=tf.float32)

# Parameters
num_layers = 2
d_model = 128
num_heads = 8
dff = 512
input_vocab_size = len(tokenizer.vocab)
maximum_position_encoding = max_sequence_len
dropout_rate = 0.1

# The transformer model
inputs = tf.keras.layers.Input(shape=(max_sequence_len-1,))
x = tf.keras.layers.Embedding(input_vocab_size, d_model)(inputs)
x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
x += positional_encoding(maximum_position_encoding, d_model)

x = tf.keras.layers.Dropout(dropout_rate)(x)

for _ in range(num_layers):
    attn_output, _ = MultiHeadAttention(d_model, num_heads)(x, x, x, None)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ff_output = point_wise_feed_forward_network(d_model, dff)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_output)

outputs = tf.keras.layers.Dense(input_vocab_size, activation='softmax')(x)
transformer = tf.keras.Model(inputs=inputs, outputs=outputs)
# transformer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
transformer.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
transformer.summary()

# Train the model
history = transformer.fit(X, y, epochs=100, verbose=1)

# Plot training accuracy values
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

# Plot training loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')

plt.tight_layout()
plt.show()

# Make predictions
def predict_next_token(model, tokenizer, text):
    sequence = tokenizer.encode(text, add_special_tokens=True)
    sequence = pad_sequences([sequence], maxlen=max_sequence_len-1, padding='post', truncating='post')
    prediction = model.predict(sequence, verbose=0)
    predicted_index = np.argmax(prediction[0], axis=-1)
    return tokenizer.decode(predicted_index)

print(predict_next_token(transformer, tokenizer, "The capital of Indonesia is"))