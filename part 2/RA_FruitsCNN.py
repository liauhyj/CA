#!/usr/bin/env python
# coding: utf-8

# In[370]:


import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


# In[371]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[372]:


training_data_generator = ImageDataGenerator(rescale = 1.0/255)


# In[373]:


training_iterator= training_data_generator.flow_from_directory("Downloads/CA_Part2_Dataset/train", batch_size = 32, class_mode="categorical", color_mode="rgb", target_size=(150,150))


# In[374]:


testing_data_generator = ImageDataGenerator(rescale = 1.0/255)


# In[375]:


testing_iterator= testing_data_generator.flow_from_directory("Downloads/CA_Part2_Dataset/test", batch_size = 64, class_mode="categorical", color_mode="rgb", target_size=(150,50))


# In[376]:


validation_data_generator = ImageDataGenerator(rescale = 1/255)


# In[377]:


validation_generator = validation_data_generator.flow_from_directory("Downloads/CA_Part2_Dataset/test", batch_size = 64, class_mode="categorical", color_mode="rgb", target_size=(150,150))


##### In[378]: THE CNN MODEL #######


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (5,5), activation='relu',input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(32, (5,5), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
        
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation = 'softmax')
])

model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['acc'])

history = model.fit(training_iterator,
                    epochs = 15,
                    verbose = 1,
                   validation_data = validation_generator)


# In[381]:


acc = history.history['acc']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']

epochs = range(len(acc))


plt.plot(epochs, val_acc, 'b', label='Accuracy')
plt.plot(epochs, val_loss, 'r', label='Loss')
plt.title('Accuracy and Loss')
plt.legend()
plt.figure()

plt.show()


# In[ ]:




