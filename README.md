# EXPERIMENT - 07 Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The task is to classify a given image of a handwritten digit into one of 10 classes representing integer values from 0 to 9, inclusively.Autoencoders uses MaxPooling, convolutional and upsampling layers to denoise the image. It will first encode the image into a lower-dimensional representation, then decodes the representation back to the image.

 We are using MNIST Dataset for this experiment. The dataset has a collection of 60,000 handwrittend digits of size 28 X 28. 

 ![image](https://github.com/Shavedha/convolutional-denoising-autoencoder/assets/93427376/9c78d274-3729-43ac-bc78-951155c34ffe)


## Convolution Autoencoder Network Model

![image](https://github.com/Shavedha/convolutional-denoising-autoencoder/assets/93427376/2798c154-8506-48fc-afd5-9d7896ae12c9)


## DESIGN STEPS

1. Import the necessary libraries and dataset.
2. Load the dataset and scale the values for easier computation.
3. Add noise to the images randomly for both the train and test sets.
4. Build the Neural Model using Convolutional Layer,Pooling Layer,Up Sampling Layer. 
5. Make sure the input shape and output shape of the model are identical.
6. Pass test data for validating manually.
7. Plot the predictions for visualization.

## PROGRAM
```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import utils
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = mnist.load_data()

x_train.shape

x_train_scaled = x_train.astype('float32') / 255.  # to bring the images to the range btw 0 and 1
x_test_scaled = x_test.astype('float32') / 255.
x_train_scaled = np.reshape(x_train_scaled, (len(x_train_scaled), 28, 28, 1))
x_test_scaled = np.reshape(x_test_scaled, (len(x_test_scaled), 28, 28, 1))

noise_factor = 0.5
x_train_noisy = x_train_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train_scaled.shape)
x_test_noisy = x_test_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test_scaled.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

print("Shavedha Y - 212221230095")
n = 10
plt.figure(figsize=(20, 2))
for i in range(1, n + 1):
    ax = plt.subplot(1, n, i)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

input_img = keras.Input(shape=(28, 28, 1))

x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(input_img)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.MaxPooling2D((2,2),padding='same')(x)
x=layers.Conv2D(8,(7,7),activation='relu',padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(encoded)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(4,(3,3),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(8,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu',padding='same')(x)
x=layers.UpSampling2D((2,2))(x)
x=layers.Conv2D(16,(5,5),activation='relu')(x)
x=layers.UpSampling2D((1,1))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded)

autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.fit(x_train_noisy, x_train_scaled,
                epochs=3,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_noisy, x_test_scaled))

print('Y Shavedha - 212221230095')
metrics = pd.DataFrame(autoencoder.history.history)
metrics[['loss','val_loss']].plot()

decoded_imgs = autoencoder.predict(x_test_noisy)

print("Y Shavedha - 212221230095")
n = 10
plt.figure(figsize=(20, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(3, n, i)
    plt.imshow(x_test_scaled[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display noisy
    ax = plt.subplot(3, n, i+n)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)    

    # Display reconstruction
    ax = plt.subplot(3, n, i + 2*n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/Shavedha/convolutional-denoising-autoencoder/assets/93427376/fd8192b1-0284-49b0-8260-a4f640e583bb)

### Original vs Noisy Vs Reconstructed Image

![image](https://github.com/Shavedha/convolutional-denoising-autoencoder/assets/93427376/e075ec94-3693-4b2a-918b-6074a7aa85ba)


## RESULT
Thus a convolutional autoencoder for image denoising application is developed successfully. 
