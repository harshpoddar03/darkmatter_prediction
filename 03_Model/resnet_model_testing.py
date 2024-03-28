

print("Importing Libraries...")
import numpy as np
import gc
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import os

import tensorflow as tf
from tensorflow.keras import layers, models

############################################
print("Loading Data...")
#load the data
augmented_no = np.load('../00_data/augumented/augmented_no.npy')
augmented_sphere = np.load('../00_data/augumented/augmented_sphere.npy')
augmented_vort = np.load('../00_data/augumented/augmented_vort.npy')


#make y classes for the data
y_no = np.zeros((augmented_no.shape[0],1))
y_sphere = np.ones((augmented_sphere.shape[0],1))
y_vort = np.full((augmented_vort.shape[0],1),2)

y_no = to_categorical(y_no, num_classes=3)
y_sphere = to_categorical(y_sphere, num_classes=3)
y_vort = to_categorical(y_vort, num_classes=3)

print("Data Loaded...")

############################################

print("Preparing Data...")

x_data = np.concatenate((augmented_no, augmented_sphere, augmented_vort), axis=0)
y_data = np.concatenate((y_no, y_sphere, y_vort), axis=0)
del(augmented_no, augmented_sphere, augmented_vort, y_no, y_sphere, y_vort)
gc.collect()

x_data = x_data.astype('float32')
gc.collect()

print("Data Prepared...")


############################################

print("Splitting Data...")

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
del(x_data, y_data)
gc.collect()

print("Data Splitted...")
############################################

print("Building Model...")



# Define the number of classes in your dataset
num_classes = 3  # Update this to the number of classes in your dataset

def build_resnet50_model(input_shape=(150, 150, 1), num_classes=num_classes):
    # Initialize the ResNet50 model without pre-trained weights and with the specified input shape
    base_model = tf.keras.applications.ResNet101V2(include_top=False,
                                                 weights=None,  # No pre-trained weights
                                                 input_shape=input_shape)

    # Since we're training from scratch, all layers can be trainable
    base_model.trainable = True

    # Create a new model on top
    inputs = tf.keras.Input(shape=input_shape)
    # Use the generated model
    x = base_model(inputs, training=True)  # Set training=True to ensure BN layers adapt
    
    # Add pooling layer or flatten layer
    x = layers.GlobalAveragePooling2D()(x)
    # Add a dropout layer for some regularization
    x = layers.Dense(128, activation='relu')(x)
    # Add final dense layer with softmax for classification
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Compile the model
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model 

# Build the model
model = build_resnet50_model(input_shape=(150, 150, 1), num_classes=num_classes)


# Print the model summary to check the final architecture
print(model.summary())

print("Model Built...")


############################################

print("Training Model...")


model_path = '../04_Data_Models/doubleresnet/resnet50_dual_model'
status_file_path = '../04_Data_Models/doubleresnet/status.txt'

for i in range(10):
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=4)
    # model.fit(x_train, y_train, epochs=10, batch_size=4, validation_data=(x_test, y_test))
    
    # Save the model in the recommended format, checking if it exists and deleting it if necessary.
    # We're using a directory format recommended by TensorFlow for the model save
    if os.path.exists(model_path):
        # This is a directory, so we need to remove it as such
        os.system(f'rm -rf {model_path}')
    model.save(model_path)
    
    # Append the status after each epoch to a file
    with open(status_file_path, 'a') as f:
        f.write(f'Epoch {i} done\n')
    
    # Attempt to clear unused memory
    gc.collect()

    # Optional: If you need to continue training from the last checkpoint, you can load the model
    # This is optional and only necessary if your training loop is interrupted or you plan to do further training
    # model = load_model(model_path)
    

print("Model Trained...")

############################################