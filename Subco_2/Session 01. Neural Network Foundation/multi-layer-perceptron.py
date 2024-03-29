import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, auc, roc_auc_score, roc_curve

def load_data(keras_datasets, first_layer="dense", channels=1, plot_images=False, class_names=[]):
    (x_train, y_train), (x_test, y_test) = keras_datasets.load_data()
    print('Before reshape - X_train.shape:', x_train.shape)
    print('Before reshape - X_test.shape:', x_test.shape)
    height=x_train.shape[1]
    width=x_train.shape[2]        
    # flatten 28*28 images to a 784 vector for each image
    num_pixels = height * width
    
    # plot images
    if plot_images:
        plot_images_with_labels(x_train, y_train, height, width, class_names, 25)
    
    if first_layer == "dense":
        # convert shape of x_train from (60000, 28, 28) to (60000, 784) - 784 columns per row        
        X_train = x_train.reshape((x_train.shape[0], num_pixels)).astype('float32')
        X_test = x_test.reshape((x_test.shape[0], num_pixels)).astype('float32')        
        
    print('After reshape - X_train.shape:', X_train.shape)
    print('After reshape - X_test.shape:', X_test.shape)
    print('Before rescaling:', X_train[0])
    # normalize the values between 0 and 1
    X_train = (X_train.astype(np.float32))/255
    X_test = (X_test.astype(np.float32))/255
    print('After rescaling:', X_train[0])
              
    # convert labels to categorical/dummy encoding so that we can use simple "categorical_crossentropy" as loss.
    print('Class label of first image before converting to categorical:', y_train[0])
    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    print('Total number of classes:', num_classes)
    print('Class label of first image after converting to categorical:', y_train[0])
              
    return (X_train, y_train, X_test, y_test, height, width)

def plot_images_with_labels(X, y, img_height, img_width, class_names, nb_count=25):
    plt.figure(figsize=(10, 10))
    for i in range(nb_count):
        plt.subplot(int(np.sqrt(nb_count)), int(np.sqrt(nb_count)), i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(X[i].reshape((img_height,img_width)), cmap=plt.get_cmap('gray'))        
        label_index = int(y[i])
        plt.title(class_names[label_index])
    plt.savefig('image-with-labels.png')
    plt.show()

def train_model(model, X_train, y_train, X_valid=None, y_valid=None, validation_split=0.20, data_aug = False, best_model_name='best_model.h5', epochs=50, batch_size=512, verbose=1):
    er = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=verbose)
    cp = ModelCheckpoint(filepath = best_model_name, save_best_only = True,verbose=verbose)
    callbacks = [cp, er]
    
    if not data_aug and X_valid is not None:  
        print('Training without data augmentation...')
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,verbose=verbose, callbacks=callbacks, validation_data=(X_valid,y_valid))
        return history
    elif not data_aug and X_valid is None:
        print('Training without data augmentation...')
        history = model.fit(X_train, y_train, batch_size=batch_size,epochs=epochs, verbose=verbose, shuffle=True, callbacks=callbacks, validation_split=validation_split)
        return history
    else:
        print('Training with data augmentation...')
        train_datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        train_set_ae = train_datagen.flow(X_train, y_train, batch_size=batch_size)

        validation_datagen = ImageDataGenerator()
        validation_set_ae = validation_datagen.flow(X_valid, y_valid, batch_size=batch_size)
        
        history = model.fit_generator(train_set_ae,
                                           epochs=epochs,
                                           steps_per_epoch=np.ceil(X_train.shape[0]/batch_size),
                                           verbose=verbose, callbacks=callbacks,
                                           validation_data=(validation_set_ae),
                                           validation_steps=np.ceil(X_valid.shape[0]/batch_size))
        
        return history
    
def plot_loss_and_metrics(history, plot_loss_only= False, metrics=['acc']):
    fig, axes = plt.subplots(nrows=1, ncols=len(metrics)+1, figsize=(20, 4))
    axes[0].plot(history.history['loss'])
    axes[0].plot(history.history['val_loss'])
    axes[0].set_title('Model Loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(['Train', 'Val'], loc='lower right')    
        
    if not plot_loss_only:
        axes[1].plot(history.history['acc'])
        axes[1].plot(history.history['val_acc'])
        axes[1].set_title('Model Accuracy')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].legend(['Train', 'Val'], loc='lower right')  
        
        if 'mae' in metrics:
            axes[2].plot(history.history['mae'])
            axes[2].plot(history.history['val_mae'])
            axes[2].set_title('Model Mean Absolute Error')
            axes[2].set_ylabel('Mean Absolute Error')
            axes[2].set_xlabel('Epoch')
            axes[2].legend(['Train', 'Val'], loc='lower right') 
        if 'mse' in metrics:
            axes[3].plot(history.history['mse'])
            axes[3].plot(history.history['val_mse'])
            axes[3].set_title('Model Mean Squared Error')
            axes[3].set_ylabel('Mean Squared Error')
            axes[3].set_xlabel('Epoch')
            axes[3].legend(['Train', 'Val'], loc='lower right')
            
    plt.savefig('mlp-loss-and-accuracy-metrics.png')
    plt.show()
    
def plot_roc_curve(fpr,tpr): 
  import matplotlib.pyplot as plt
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate')
  plt.savefig('mlp-roc-curve.png') 
  plt.show()  
    
def load_evaluate_predict(fileName, X_test, y_test, nb_round=0, print_first=1, metrics=['acc']):
    #load best model, evaluate and predict on unseen data    
    best_model = load_model(fileName)
    results = best_model.evaluate(X_test, y_test)    
    print('Test loss = {}'.format(np.round(results[0], 2)))
    print('Test accuracy = {}'.format(np.round(results[1], 2)))
    if len(metrics)>1:
        print('Test ' + metrics[1] + '= {}'.format(np.round(results[2], 2)))
        print('Test ' + metrics[2] + '= {}'.format(np.round(results[3], 2)))

    y_pred_proba = best_model.predict(X_test)
    for i in range(print_first):
        print('')
        print("   Actual:", y_test[i])
        print('Predicted:', np.round(y_pred_proba[i], nb_round))
    
    return best_model, y_pred_proba

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    import itertools
    
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=12)
    plt.xlabel('Predicted label', fontsize=12)
    
def report_metrics(y_test, y_pred, y_pred_proba, classes, multiclass=False):
    # confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    plot_confusion_matrix(cnf_matrix, classes=classes, title="Confusion matrix")
    plt.savefig('mlp-confusion-matrix.png')
    plt.show()

    # classification report
    print('Classification Report:\n', classification_report(y_test, y_pred))
    
    if not multiclass:
        # Calculate the ROC AUC score
        auc = roc_auc_score(y_test, y_pred_proba)
        print('AUC: %.3f' % auc)
        
        # Plot the ROC curve
        fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred_proba)
        print('ROC curve:\n')
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()
        
NUM_CLASSES=10
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

EPOCHS=50
BATCH_SIZE=1000
CHANNELS=1
VERBOSE=1
METRICS=['acc']

X_train, y_train, X_test, y_test, IMG_HEIGHT, IMG_WIDTH = load_data(keras.datasets.fashion_mnist, first_layer="dense", channels=1, plot_images=True, class_names=CLASS_NAMES)

X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

print("Training dataset shape:", X_train.shape)
print("Validation dataset shape:", X_valid.shape)
print("Testing dataset shape:", X_test.shape)

total_samples = X_train.shape[0] + X_valid.shape[0] + X_test.shape[0]

train_percentage = (X_train.shape[0] / total_samples) * 100
valid_percentage = (X_valid.shape[0] / total_samples) * 100
test_percentage = (X_test.shape[0] / total_samples) * 100

print("Training dataset percentage:", train_percentage, "%")
print("Validation dataset percentage:", valid_percentage, "%")
print("Testing dataset percentage:", test_percentage, "%")

# create MLP
def build_mlp_model(height, width, nb_classes, metrics):
    # create the multilayer preceptron model with 4 hidden layers
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(256, input_dim=(height* width), activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(nb_classes, activation='softmax'))
    model.compile(loss = 'categorical_crossentropy',
              optimizer='rmsprop',                           #optimizer=keras.optimizers.SGD(lr=.001),#optimizer='sgd',              
              metrics=metrics)
    return model

model_mlp = build_mlp_model(IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES, METRICS)

print(model_mlp.summary())

hidden1 = model_mlp.layers[1]
weights, biases = hidden1.get_weights()
print(weights.shape)
print(biases.shape)

history_mlp = train_model(model_mlp, X_train, y_train, X_valid=X_valid, y_valid=y_valid, data_aug = False, 
            best_model_name='best_model_mlp.h5', epochs=EPOCHS, batch_size=BATCH_SIZE,verbose=VERBOSE)

# print the loss and accuracy
plot_loss_and_metrics(history_mlp)
_, y_pred_proba_mlp = load_evaluate_predict('best_model_mlp.h5', X_test, y_test, nb_round=0, print_first=1, metrics=METRICS)

print(y_pred_proba_mlp[0])
print(y_pred_proba_mlp[0].shape)
## Get most likely class
y_pred_mlp = np.argmax(y_pred_proba_mlp, axis=1)
print(y_pred_mlp)

# Confusion Matrix, Classification report, ROC curve
report_metrics(np.argmax(y_test, axis=1), y_pred_mlp, y_pred_proba_mlp, CLASS_NAMES, multiclass=True)

# predictions on unseen test data
X_new = X_test[:3]
y_proba = model_mlp.predict(X_new)
print(y_proba.round(2))

y_pred = np.argmax(model_mlp.predict(X_new), axis=1)

print(f"Prediction  : {y_pred}")
print(f"Class Names : {np.array(CLASS_NAMES)[y_pred]}")