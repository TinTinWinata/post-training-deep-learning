import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_text as text
import pandas as pd
import os
import warnings

os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'UNCOMPRESSED'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def subtract_one_label(text, label):
    return text, label - 1

# Read the CSV files
train_df = pd.read_csv('train.csv')
val_df = pd.read_csv('test.csv')

text_col = 'description'
label_col = 'label'

train_data = tf.data.Dataset.from_tensor_slices((train_df[text_col].values, train_df[label_col].values))
train_data = train_data.map(subtract_one_label)

val_data = tf.data.Dataset.from_tensor_slices((val_df[text_col].values, val_df[label_col].values))
val_data = val_data.map(subtract_one_label)

# Other information about the data
class_names = ['World', 'Sports', 'Business', 'Sci/Tech']
num_classes = len(class_names)
num_train = len(train_df)
num_val = len(val_df)

print(f'The news are grouped into {num_classes} classes that are :{class_names}')
print(f'The number of training samples: {num_train} \nThe number of validation samples: {num_val}')

for i, (text, label) in enumerate(train_data.take(4)):
    print(f"Sample news {i}\n \
    Label: {label.numpy()} {class_names[label.numpy()]}\n \
    Description: {text.numpy().decode('utf-8')}\n----------\n")

buffer_size = 1000
batch_size = 32

train_data = train_data.shuffle(buffer_size)
train_data = train_data.batch(batch_size).prefetch(1)
val_data = val_data.batch(batch_size).prefetch(1)

for news, label in train_data.take(1):
    print(f'Sample news\n----\n {news.numpy()[:4]} \n----\nCorresponding labels: {label.numpy()[:4]}')
  
bert_handle = 'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2'
preprocessing_model = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
preprocess_layer = hub.KerasLayer(preprocessing_model)

sample_news = ['Tech rumors: The tech giant Apple is working on its self driving car']
preprocessed_news = preprocess_layer(sample_news)

print(f'Keys       : {list(preprocessed_news.keys())}')
print(f'Shape      : {preprocessed_news["input_word_ids"].shape}')
print(f'Word Ids   : {preprocessed_news["input_word_ids"][0, :5]}')
print(f'Input Mask : {preprocessed_news["input_mask"][0, :5]}')
print(f'Type Ids   : {preprocessed_news["input_type_ids"][0, :5]}')

bert_model = hub.KerasLayer(bert_handle)
bert_outputs = bert_model(preprocessed_news)

print(f'Pooled output shape:{bert_outputs["pooled_output"].shape}')
print(f'Pooled output values:{bert_outputs["pooled_output"][0, :5]}')
print(f'Sequence output shape:{bert_outputs["sequence_output"].shape}')
print(f'Sequence output values:{bert_outputs["sequence_output"][0, :5]}')

input_text = tf.keras.layers.Input(shape=(), dtype=tf.string, name='Input')

# A preprocesing model before layer BERT
preprocessing_layer = hub.KerasLayer(preprocessing_model, name='preprocessing_layer')
bert_input = preprocessing_layer(input_text)

# Getting a Bert model, set trainable to True
bert_encoder = hub.KerasLayer(bert_handle, trainable=True, name='bert_encoder')
bert_outputs = bert_encoder(bert_input)

# For finetuning, we take pooled_output
pooled_bert_output = bert_outputs['pooled_output']

# Adding a dense layer 
dense_net = tf.keras.layers.Dense(16, activation='relu', name='fully_connected_layer')(pooled_bert_output)

# Add dropout layer for regularization
dense_net = tf.keras.layers.Dropout(0.2)(dense_net)

# Last dense layer for classification purpose
final_output = tf.keras.layers.Dense(4, activation='softmax', name='classifier')(dense_net)

# Combine input and output
news_classifier = tf.keras.Model(input_text, final_output)

print(news_classifier.summary())

news_classifier.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), 
                        loss='sparse_categorical_crossentropy', 
                        metrics=['accuracy'])

# Train the model
batch_size = 32

# Compute the length of the train_data and val_data
num_train = sum(1 for _ in train_data)
num_val = sum(1 for _ in val_data)

train_steps = num_train // batch_size
val_steps = num_val // batch_size

history = news_classifier.fit(train_data, epochs=15, validation_data=val_data, steps_per_epoch=train_steps, validation_steps=val_steps)

# function to plot accuracy and loss
def plot_acc_loss(history):
    model_history = history.history
    acc = model_history['accuracy']
    val_acc = model_history['val_accuracy']
    loss = model_history['loss']
    val_loss = model_history['val_loss']
    
    epochs = history.epoch

    plt.figure(figsize=(10,5))
    plt.plot(epochs, acc, 'r', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'g', label='Validation Accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig("Training and Validation Accuracy.png")
    plt.legend(loc=0)

    # Create a new figure with plt.figure()
    plt.figure()

    plt.figure(figsize=(10,5))
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'y', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc=0)
    plt.savefig("Training and Validation Loss.png")
    plt.show()
  
plot_acc_loss(history)

def predict(model, sample_news, class_names):
    # Convert sample news into array
    sample_news = np.array(sample_news)

    # Predict the news type
    preds = model.predict(sample_news)
    pred_class = np.argmax(preds[0])

    print(f'Predicted class: {pred_class} \nPredicted Class name: {class_names[pred_class]}')

sample_news = ['Tesla, a self driving car company is also planning to make a humanoid robot. This humanoid robot appeared dancing in the latest Tesla AI day']
predict(news_classifier, sample_news, class_names)