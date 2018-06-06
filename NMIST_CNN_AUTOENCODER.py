import keras
from keras.layers import Input,Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#preprocessing
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1)) 


#model

inputs = Input((28, 28, 1))
conv1 = Conv2D(16, (3, 3),  padding='same')(inputs)
act1=PReLU()(conv1)
batch1=BatchNormalization(axis=3)(act1) 
pool1 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch1)

conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
act2=PReLU()(conv2)
batch2=BatchNormalization(axis=3)(act2)
pool2 = MaxPooling2D(pool_size=(2, 2),padding='same')(batch2)

deconv2 = Conv2D(16, (3, 3), padding='same')(pool2)
dbatch2=BatchNormalization(axis=3)(deconv2) 
dact2 = Activation('relu')(dbatch2)
dact2=UpSampling2D((2,2))(dact2)

deconv3 = Conv2D(16, (3, 3), padding='same')(dact2)
dbatch3=BatchNormalization(axis=3)(deconv3) 
dact3 = Activation('relu')(dbatch3)
dact3=UpSampling2D((2,2))(dact3)

decoded=Conv2D(1,(3,3),padding='same',activation='sigmoid')(dact3)

autoencoder=Model(inputs=inputs,outputs=decoded)
autoencoder.compile(optimizer='rmsprop', loss='mse')

#for plotting loss graph
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

#training
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[plot_losses])

autoencoder.save("CNN_ENCODER_NMIST.h5")

#teesing on test data
decoded=autoencoder.predict(x_test)

n = 10  #number of image to display
plt.figure(figsize=(30, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded[i].reshape(28, 28))
    
plt.show()