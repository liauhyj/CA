import numpy as np
import pandas as pd
import sys
import tensorflow as tf
import matplotlib.pyplot as plot
from PIL import Image
import os
import shutil


def resizepictures():
    files=[]
    for file in os.listdir('C:/Users/Xu Zhenli/Desktop/Python ML/test'):
        if file.endswith(".jpg"):
            files.append("C:/Users/Xu Zhenli/Desktop/Python ML/test/"+file)
    
    for f in files:
        shutil.copy(f,'C:/Users/Xu Zhenli/Desktop/Tr')
    
    for f in os.listdir('C:/Users/Xu Zhenli/Desktop/Tr'):
        img = Image.open("C:/Users/Xu Zhenli/Desktop/Tr/"+f)
        rgb_img=img.convert('RGB')
        out = rgb_img.resize((224, 224),Image.ANTIALIAS) 
        out.save("C:/Users/Xu Zhenli/Desktop/Test/"+f)
    
def turntoarray():
    x_train=[]
    y_train=[]
    
    for f in os.listdir('C:/Users/Xu Zhenli/Desktop/Train'):
        if f.startswith("apple"):
            y_train.append(1)
        elif f.startswith("banana"):
            y_train.append(2)
        elif f.startswith("orange"):
            y_train.append(3)
        else:
            y_train.append(4)
        
        img=Image.open('C:/Users/Xu Zhenli/Desktop/Train/'+f)
#        img=img.resize((100,100),Image.ANTIALIAS)
        img_arr=np.asarray(img,dtype='float64')/255.
        x_train.append(img_arr.tolist())
    
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    
    x_test=[]
    y_test=[]
    
    for f in os.listdir('C:/Users/Xu Zhenli/Desktop/Test'):
        if f.startswith("apple"):
            y_test.append(1)
        elif f.startswith("banana"):
            y_test.append(2)
        elif f.startswith("orange"):
            y_test.append(3)
        else:
            y_test.append(4)
        
        img=Image.open('C:/Users/Xu Zhenli/Desktop/Test/'+f)
#        img=img.resize((100,100),Image.ANTIALIAS)
        img_arr=np.asarray(img,dtype='float64')/255.
        x_test.append(img_arr.tolist())
    
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    return (x_train,y_train,x_test,y_test)


def preprocess(y_train, y_test):

	y_train = tf.keras.utils.to_categorical(y_train, 10)
	y_test = tf.keras.utils.to_categorical(y_test, 10)

	return (y_train, y_test)
    
def run_cnn(_x_train, _y_train, _x_test, _y_test):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3),activation='relu', input_shape=(224, 224, 3)))
    model.add(tf.keras.layers.Conv2D(64, (4, 4),strides=2,activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(tf.keras.layers.Conv2D(64, (3, 3),activation='relu'))
    model.add(tf.keras.layers.Conv2D(128, (5, 5),activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(tf.keras.layers.Dropout(0.25))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    
    
    model.fit(_x_train, _y_train, 
		batch_size=64, epochs=10, verbose=1,
		validation_data=(_x_test, _y_test))
    
    score = model.evaluate(_x_test, _y_test)
    print("score =", score)    


#resizepictures()
    
(x_train, _y_train, x_test, _y_test) = turntoarray()
(y_train, y_test) = preprocess(_y_train, _y_test)
model = run_cnn(x_train, y_train, x_test, y_test)



