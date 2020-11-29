import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# index_labels is used for ease comparing the actual image vs predicted image
index_labels = ["Apple", "Banana", "Mixed Fruits", "Orange"]

# accuracy_data, loss_data and training_duration_data will be used to evaluate our models
accuracy_data = []
loss_data = []
training_duration_data = []
predict_duration_data= []

def read_resize_image_from_path(path):
    img_data = []
    label_data = []
    for image in glob.glob(path):
        # classify the image and append classification to label_data
        if "apple" in image:
            label_data.append(0)
        elif "banana" in image:
            label_data.append(1)
        elif "mixed" in image:
            label_data.append(2)
        elif "orange" in image:
            label_data.append(3)
        # read image and use RGB color space
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        # resize the image to a 34 x 34 pixel image
        img_resized = cv2.resize(img, dsize=(34, 34), interpolation=cv2.INTER_CUBIC)
        img_data.append(img_resized)
    img_data = np.array(img_data)
    label_data = np.array(label_data)
    return img_data, label_data


def render_img(data):
    plt.imshow(data)
    plt.show()


def preprocess(_x_train, _y_train, _x_test, _y_test):
    # establish the depth of our images; here we have color images, the depth is 3.
    x_train = np.reshape(_x_train, (_x_train.shape[0], 34, 34, 3))
    x_test = np.reshape(_x_test, (_x_test.shape[0], 34, 34, 3))

    # scale the values by 255 since each channel takes a value between 0 to 255
    x_train = x_train / 255
    x_test = x_test / 255

    # one-hot encode y_train and y_test to have values of a length 4 array
    y_train = tf.keras.utils.to_categorical(_y_train, 4)
    y_test = tf.keras.utils.to_categorical(_y_test, 4)

    return (x_train, y_train, x_test, y_test)


def run_cnn1(_x_train, _y_train, _x_test, _y_test):
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 32 filters, each of size 3x3 with activation function relu
    # the input shape is specified as (34, 34, 3) which is exactly the same shape of our input data
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(34, 34, 3)))
    # in this Pooling layer we are using a pooling window of size 3x3
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))
    # in this Dropout layer we are specifying a dropout rate of 0.5
    model.add(tf.keras.layers.Dropout(0.5))
    # before feeding the data into the dense/fully-connected layer, we flatten the data into
    # a 1-dimensional vector
    model.add(tf.keras.layers.Flatten())
    # in this dense layer, we specifying units as 64 with activation function relu
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    # in this Dropout layer we are specifying a dropout rate of 0.25
    model.add(tf.keras.layers.Dropout(0.25))
    # in this dense layer, we specifying units as 4 with activation function softmax
    # because we have exactly 4 possible categories of images, namely, apple, banana
    # mixed fruits and orange
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_start_time = time.time()
    hist=model.fit(_x_train, _y_train,
              batch_size=240, epochs=100, verbose=1,
              validation_data=(_x_test, _y_test))
    # plot of train_loss, validation_loss, train_accuracy, validation_accuracy against epoch
    plt.plot(hist.history['loss'],color='r')
    plt.plot(hist.history['val_loss'], color='g')
    plt.plot(hist.history['accuracy'], color='b')
    plt.plot(hist.history['val_accuracy'], color='k')
    plt.title('training_loss, validation_loss, training_accuracy, validation_accuracy against epoch')
    plt.ylabel('values')
    plt.xlabel('epoch')
    plt.legend(['training_loss', 'validation_loss', 'training_accuracy', 'validation_accuracy'], loc='upper right')
    plt.show()
    training_time_taken = time.time() - training_start_time
    training_duration_data.append(training_time_taken)
    print("\nDuration of training\n--- %s seconds ---\n" % training_time_taken)
    print("Predicting...")
    predict_start_time = time.time()
    score = model.evaluate(_x_test, _y_test)
    print("score =", score)
    predict_time_taken = time.time() - predict_start_time
    loss_data.append(score[0])
    accuracy_data.append(score[1])
    predict_duration_data.append(predict_time_taken)
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())


def run_cnn2(_x_train, _y_train, _x_test, _y_test):
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 32 filters, each of size 3x3 with activation function relu
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(34, 34, 3)))
    # in this Pooling layer we are using a pooling window of size 6x6
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(6, 6)))
    # in this Dropout layer we are specifying a dropout rate of 0.5
    model.add(tf.keras.layers.Dropout(0.5))
    # before feeding the data into the dense/fully-connected layer, we flatten the data into
    # a 1-dimensional vector
    model.add(tf.keras.layers.Flatten())
    # in this dense layer, we specifying units as 4 with activation function softmax
    # because we have exactly 4 possible categories of images, namely, apple, banana
    # mixed fruits and orange
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_start_time = time.time()
    model.fit(_x_train, _y_train,
              batch_size=240, epochs=300, verbose=1,
              validation_data=(_x_test, _y_test))
    training_time_taken = time.time() - training_start_time
    training_duration_data.append(training_time_taken)
    print("\nDuration of training\n--- %s seconds ---\n" % training_time_taken)

    print("Predicting...")
    predict_start_time = time.time()
    score = model.evaluate(_x_test, _y_test)
    print("score =", score)
    predict_time_taken = time.time() - predict_start_time
    loss_data.append(score[0])
    accuracy_data.append(score[1])
    predict_duration_data.append(predict_time_taken)
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())


def run_cnn3(_x_train, _y_train, _x_test, _y_test):
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 16 filters, each of size 5x5 with activation function relu
    model.add(tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(34, 34, 3)))
    # in this Pooling layer we are using a pooling window of size 2x2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # in this Conv2D layer we are using 32 filters, each of size 3x3 with activation function relu
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
    # in this Pooling layer we are using a pooling window of size 2x2
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # before feeding the data into the dense/fully-connected layer, we flatten the data into
    # a 1-dimensional vector
    model.add(tf.keras.layers.Flatten())
    # in this dense layer, we specifying units as 256 with activation function relu
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    # in this Dropout layer we are specifying a dropout rate of 0.5
    model.add(tf.keras.layers.Dropout(0.5))
    # in this dense layer, we specifying units as 4 with activation function softmax
    # because we have exactly 4 possible categories of images, namely, apple, banana
    # mixed fruits and orange
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_start_time = time.time()
    model.fit(_x_train, _y_train,
              batch_size=240, epochs=100, verbose=1,
              validation_data=(_x_test, _y_test))
    training_time_taken = time.time() - training_start_time
    training_duration_data.append(training_time_taken)
    print("\nDuration of training\n--- %s seconds ---\n" % training_time_taken)

    print("Predicting...")
    predict_start_time = time.time()
    score = model.evaluate(_x_test, _y_test)
    print("score =", score)
    predict_time_taken = time.time() - predict_start_time
    loss_data.append(score[0])
    accuracy_data.append(score[1])
    predict_duration_data.append(predict_time_taken)
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())


def run_cnn4(_x_train, _y_train, _x_test, _y_test):
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 32 filters, each of size 3x3 with activation function relu
    # the input shape is specified as (34, 34, 3) which is exactly the same shape of our input data
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(34, 34, 3)))
    # in this Conv2D layer we are using 64 filters, each of size 4x4 with activation function relu
    # each filter has a stride of 2
    model.add(tf.keras.layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
    # in this Pooling layer we are using a pooling window of size 2x2 with padding as same
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    # in this Conv2D layer we are using 64 filters, each of size 3x3 with activation function relu
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    # in this Conv2D layer we are using 128 filters, each of size 5x5 with activation function relu
    model.add(tf.keras.layers.Conv2D(128, (5, 5), activation='relu'))
    # in this Pooling layer we are using a pooling window of size 2x2 with padding as same
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
    # in this Dropout layer we are specifying a dropout rate of 0.25
    model.add(tf.keras.layers.Dropout(0.25))
    # before feeding the data into the dense/fully-connected layer, we flatten the data into
    # a 1-dimensional vector
    model.add(tf.keras.layers.Flatten())
    # in this dense layer, we specifying units as 32 with activation function relu
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    # in this Dropout layer we are specifying a dropout rate of 0.5
    model.add(tf.keras.layers.Dropout(0.5))
    # in this dense layer, we specifying units as 4 with activation function softmax
    # because we have exactly 4 possible categories of images, namely, apple, banana
    # mixed fruits and orange
    model.add(tf.keras.layers.Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    training_start_time = time.time()
    model.fit(_x_train, _y_train,
              batch_size=240, epochs=100, verbose=1,
              validation_data=(_x_test, _y_test))
    training_time_taken = time.time() - training_start_time
    training_duration_data.append(training_time_taken)
    print("\nDuration of training\n--- %s seconds ---\n" % training_time_taken)

    print("Predicting...")
    predict_start_time = time.time()
    score = model.evaluate(_x_test, _y_test)
    print("score =", score)
    predict_time_taken = time.time() - predict_start_time
    loss_data.append(score[0])
    accuracy_data.append(score[1])
    predict_duration_data.append(predict_time_taken)
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())


def plot_y_vs_x(x, y, x_label, y_label):
    plt.plot(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title("Plot of %s vs %s" % (x_label, y_label))
    plt.show()


_x_train, _y_train = read_resize_image_from_path("data/train/*.jpg")

_x_test, _y_test = read_resize_image_from_path("data/test/*.jpg")

(x_train, y_train, x_test, y_test) = preprocess(_x_train, _y_train, _x_test, _y_test)

# uncomment the line below to see an image of the trained data
# render_img(x_train[76])

run_cnn1(x_train, y_train, x_test, y_test)

#run_cnn2(x_train, y_train, x_test, y_test)
#
#run_cnn3(x_train, y_train, x_test, y_test)
#
#run_cnn4(x_train, y_train, x_test, y_test)


#this two line is where I am not sure
#plot_y_vs_x(loss_data, accuracy_data, "loss", "accuracy")
#
#plot_y_vs_x(training_duration_data, accuracy_data, "training duration", "accuracy")
#
#print(loss_data)
#print(accuracy_data)
#print(training_duration_data)
#print(predict_duration_data)