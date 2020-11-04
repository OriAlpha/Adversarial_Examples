import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

# Function to create adversarial pattern
def adversarial_pattern(image, label):
    image = tf.cast(image, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.MSE(label, prediction)

    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    return signed_grad

# Adversarial data generator
def generate_adversarials(batch_size):
    while True:
        x = []
        y = []
        for batch in range(batch_size):
            if batch_size > 10000 and batch % 1000 == 0:
                print(batch / batch_size)
            N = random.randint(0, 100)

            label = y_train[N]
            perturbations = adversarial_pattern(x_train[N].reshape((1, 28, 28, 1)), label).numpy()
            image = x_train[N]
            epsilon = 0.1
            adversarial = image + perturbations * epsilon
            x.append(adversarial)
            y.append(y_train[N])

        x = np.asarray(x).reshape((batch_size, 28, 28, 1))
        y = np.asarray(y)

        yield x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# MNIST
labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

x_train = x_train / 255
x_test = x_test / 255

# data reshape
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

# Converts to binary class matrix
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Create model
def model_generate():
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(32))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    return model

model = model_generate()
model.summary()

epochs = 1
model.fit(x_train, y_train,batch_size=32,epochs=epochs,validation_data=(x_test, y_test))

# Generate adversarial data
adversarial_xtest, adversarial_ytest = next(generate_adversarials(x_test.shape[0]))

orig_model = model.evaluate(x=x_test, y=y_test)
print("Accuracy of original model on original data:", orig_model[1]*100)
adver_model = model.evaluate(x=adversarial_xtest, y=adversarial_ytest)
print("Accuracy of original model on adversarial data:", adver_model[1]*100)

# Generate adversarial data for training a model
adversarial_xtrain, adversarial_ytrain = next(generate_adversarials(x_train.shape[0]))
# Train a model on adversarial examples
model.fit(adversarial_xtrain, adversarial_ytrain,batch_size=32,epochs=epochs,validation_data=(x_test, y_test))

adver_model_trained = model.evaluate(x=x_test, y=y_test)
print("Accuracy of adversarial model on original data:",adver_model_trained[1]*100)
model.evaluate(x=adversarial_xtest, y=adversarial_ytest)
print("Accuracy of adversarial model on original data:",adver_model_trained[1]*100)


