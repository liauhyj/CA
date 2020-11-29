import glob
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import time

# index_labels is used for ease of comparing the actual image vs predicted image
index_labels = ["Apple", "Banana", "Mixed Fruits", "Orange"]

# the variables below will be used to evaluate our models
model_data = []
img_size_data = []
accuracy_data = []
training_duration_data = []


def read_resize_image_from_path(path, size):
    img_data = []
    for image in glob.glob(path):
        # read image and use RGB color space
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        # resize the image to a size x size pixel image
        img_resized = cv2.resize(img, dsize=(size, size), interpolation=cv2.INTER_CUBIC)
        img_data.append(img_resized)
    img_data = np.array(img_data)
    return img_data


def classify_images_from_path(path):
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
    label_data = np.array(label_data)
    return label_data


def render_img(data):
    plt.imshow(data)
    plt.show()


def preprocess(_x_train, _y_train, _x_test, _y_test, size):
    # establish the depth of our images; here we have color images, the depth is 3.
    x_train = np.reshape(_x_train, (_x_train.shape[0], size, size, 3))
    x_test = np.reshape(_x_test, (_x_test.shape[0], size, size, 3))

    # scale the values by 255 since each channel takes a value between 0 to 255
    x_train = x_train / 255
    x_test = x_test / 255

    # one-hot encode y_train and y_test to have values of a length 4 array
    y_train = tf.keras.utils.to_categorical(_y_train, 4)
    y_test = tf.keras.utils.to_categorical(_y_test, 4)

    return (x_train, y_train, x_test, y_test)


# function to plot loss, val_loss, accuracy, val_accuracy against epoch
def plot_accuracy_loss(data):
    plt.figure(figsize=(12, 8), dpi=80)
    plt.plot(data.history['loss'], color='r')
    plt.plot(data.history['val_loss'], color='g')
    plt.plot(data.history['accuracy'], color='b')
    plt.plot(data.history['val_accuracy'], color='k')
    plt.title('Plot of Loss and Accuracy against Epoch run')
    plt.ylabel('Values')
    plt.xlabel('Epoch run')
    plt.legend(['Training set loss', 'Validation set loss', 'Training set accuracy', 'Validation set accuracy'],
               loc='upper right')
    plt.show()


def run_cnn1(_x_train, _y_train, _x_test, _y_test, size):
    model_data.append("cnn1")
    img_size_data.append("%s x %s" % (size, size))
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 32 filters, each of size 3x3 with activation function relu
    # the input shape is specified as (size, size, 3) which is exactly the same shape of our input data
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))
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
    plot_data = model.fit(_x_train, _y_train,
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
    accuracy_data.append(score[1])
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(_x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())

    plot_accuracy_loss(plot_data)


def run_cnn2(_x_train, _y_train, _x_test, _y_test, size):
    model_data.append("cnn2")
    img_size_data.append("%s x %s" % (size, size))
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 32 filters, each of size 3x3 with activation function relu
    # the input shape is specified as (size, size, 3) which is exactly the same shape of our input data
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))
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
    plot_data = model.fit(_x_train, _y_train,
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
    accuracy_data.append(score[1])
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(_x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())

    plot_accuracy_loss(plot_data)


def run_cnn3(_x_train, _y_train, _x_test, _y_test, size):
    model_data.append("cnn3")
    img_size_data.append("%s x %s" % (size, size))
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 16 filters, each of size 5x5 with activation function relu
    # the input shape is specified as (size, size, 3) which is exactly the same shape of our input data
    model.add(tf.keras.layers.Conv2D(16, (5, 5), activation='relu', input_shape=(size, size, 3)))
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
    plot_data = model.fit(_x_train, _y_train,
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
    accuracy_data.append(score[1])
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(_x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())

    plot_accuracy_loss(plot_data)


def run_cnn4(_x_train, _y_train, _x_test, _y_test, size):
    model_data.append("cnn4")
    img_size_data.append("%s x %s" % (size, size))
    model = tf.keras.Sequential()
    # in this Conv2D layer we are using 32 filters, each of size 3x3 with activation function relu
    # the input shape is specified as (size, size, 3) which is exactly the same shape of our input data
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(size, size, 3)))
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
    plot_data = model.fit(_x_train, _y_train,
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
    accuracy_data.append(score[1])
    print("\nDuration of prediction\n--- %s seconds ---" % predict_time_taken)

    print("\n--- Comparison of actual vs predicted ---")
    predictions = model.predict(_x_test)
    for i in np.arange(len(predictions)):
        print('Actual: ', pd.Series(y_test[i], index=index_labels).idxmax(),
              'Predicted: ', pd.Series(predictions[i], index=index_labels).idxmax())

    plot_accuracy_loss(plot_data)


def print_df_for_evaluation(model_data, img_size_data, accuracy_data, training_duration_data):
    df = pd.DataFrame({
        "Model": model_data,
        "Image Size": img_size_data,
        "Accuracy": accuracy_data,
        "Training Duration": training_duration_data
    })
    print(df)


_x_train_34 = read_resize_image_from_path("data/train/*.jpg", 34)

_x_test_34 = read_resize_image_from_path("data/test/*.jpg", 34)

_y_train = classify_images_from_path("data/train/*.jpg")

_y_test = classify_images_from_path("data/test/*.jpg")

(x_train_34, y_train, x_test_34, y_test) = preprocess(_x_train_34, _y_train, _x_test_34, _y_test, 34)

# uncomment the line below to see an image of the trained data
# render_img(x_train_34[156])

run_cnn1(x_train_34, y_train, x_test_34, y_test, 34)

# run_cnn2(x_train_34, y_train, x_test_34, y_test, 34)
#
# run_cnn3(x_train_34, y_train, x_test_34, y_test, 34)
#
# run_cnn4(x_train_34, y_train, x_test_34, y_test, 34)
#
# _x_train_128 = read_resize_image_from_path("data/train/*.jpg", 128)
#
# _x_test_128 = read_resize_image_from_path("data/test/*.jpg", 128)

# (x_train_128, y_train, x_test_128, y_test) = preprocess(_x_train_128, _y_train, _x_test_128, _y_test, 128)
#
# # uncomment the line below to see an image of the trained data
# # render_img(x_train_128[156])
#
# run_cnn1(x_train_128, y_train, x_test_128, y_test, 128)
#
# run_cnn2(x_train_128, y_train, x_test_128, y_test, 128)
#
# run_cnn3(x_train_128, y_train, x_test_128, y_test, 128)
#
# run_cnn4(x_train_128, y_train, x_test_128, y_test, 128)

print("\n Summary of runs\n")
print_df_for_evaluation(model_data, img_size_data, accuracy_data, training_duration_data)

