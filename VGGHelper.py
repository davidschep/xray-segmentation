import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from keras.models import Model
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
import cv2
import os
from keras.applications.vgg16 import VGG16



# Define the neural network architecture
class SimpleNN(tf.keras.Model):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')  # Adjust input dimensions if needed
        self.fc2 = layers.Dense(3)  # Adjust output dimensions based on the number of classes

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

def train_simple_nn(epochs, X, Y, model_filename):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU")
        except RuntimeError as e:
            print(e)

    # Convert the data to TensorFlow tensors
    X_train_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    Y_train_tensor = tf.convert_to_tensor(Y, dtype=tf.int64)

    # Create a TensorFlow Dataset and a DataLoader
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, Y_train_tensor))
    train_loader = train_dataset.shuffle(buffer_size=10000).batch(32)

    # Initialize the model
    model = SimpleNN()

    # Define loss function and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Metrics to track loss
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    # Training function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    # Train the model
    best_loss = float('inf')

    for epoch in range(epochs):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        
        for inputs, labels in train_loader:
            train_step(inputs, labels)

        current_loss = train_loss.result()
        if current_loss < best_loss:
            best_loss = current_loss
            model.save_weights(f"./Models/{model_filename}.h5")
            print(f'New best model saved with loss: {best_loss:.4f}')

        print(f'Epoch {epoch + 1}, Loss: {current_loss:.4f}')

def extract_raw_data(dirname, size):
    train_images = []
    for fname in os.listdir(dirname):
        img = cv2.imread(os.path.join(dirname, fname), cv2.IMREAD_GRAYSCALE) 
        if img is None:
            print(f"Warning: Could not read image {fname} from {dirname}")
            continue 
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        train_stack = np.stack((img,)*3, axis=-1)
        train_images.append(train_stack)   
    return np.array(train_images)

def extract_label_data(dirname, size):
    train_masks = [] 
    for fname in os.listdir(dirname):
        img = cv2.imread(os.path.join(dirname, fname), cv2.IMREAD_GRAYSCALE)    
        if img is None:
            print(f"Warning: Could not read image {fname} from {dirname}")
            continue     
        img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        train_masks.append(img)
        
    return np.array(train_masks)
def split_data_set_gray_scale(dirname_data, dirname_label, test_size, picture_size):

    X_train, X_temp, y_train, y_temp = train_test_split(extract_raw_data(dirname_data,picture_size), extract_label_data(dirname_label,picture_size), test_size=test_size, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.947, random_state=42)

    #X_train, X_test, y_train, y_test = train_test_split(extract_raw_data(dirname_data,picture_size), extract_label_data(dirname_label,picture_size), test_size=test_size, random_state=42)
    
    y_train = np.expand_dims(y_train, axis=3) 
    y_test = np.expand_dims(y_test, axis=3)
    
    return X_train, X_test, y_train, y_test,X_val,y_val

def features_2_dataframe(features,y_train):

    X=features
    X = X.reshape(-1, X.shape[3]) 
    Y = y_train.reshape(-1)
    dataframe = pd.DataFrame(X)
    dataframe['Label'] = Y
    return dataframe

def get_VGG_model_features(train_data_X,y_train, size):
    VGG_model = VGG16(weights='imagenet', include_top=False, input_shape=(size[0], size[1], 3))
    #Make loaded layers as non-trainable. This is important as we want to work with pre-trained weights
    for layer in VGG_model.layers:
        layer.trainable = False
    VGG_model.summary()  #Trainable parameters will be 0
    #After the first 2 convolutional layers the image dimension changes so we take the features vectors after first two layers
    #Load VGG16 model wothout classifier/fully connected layers
    #Talked to the professor and as it seems the input shape makes sure that we can actually run pooling and kernal operations
    VGG_model = Model(inputs=VGG_model.input, outputs=VGG_model.get_layer('block1_conv2').output)
    VGG_model.summary()
    features = VGG_model.predict(train_data_X)
    
    return features, features_2_dataframe(features,y_train), VGG_model
