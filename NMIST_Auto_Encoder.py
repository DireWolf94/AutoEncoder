from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

encoding_dim=32
in_img = Input(shape=(784,))
encoded = Dense(encoding_dim, activation='relu')(in_img)
decoded = Dense(784, activation='sigmoid')(encoded)

autoencoder=Model(inputs=in_img,outputs=decoded)
autoencoder.compile(optimizer='rmsprop', loss='mse')

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(len(x_train),x_train.shape[1]*x_train.shape[2])
x_test=x_test.reshape(len(x_test),x_test.shape[1]*x_test.shape[2])

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

decoded_img=autoencoder.predict(x_test)

n = 10  #number of image to display
plt.figure(figsize=(30, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    
plt.show()