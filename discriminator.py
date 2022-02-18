
from keras import Sequential, Model, Input
from keras.layers import Conv2D, LeakyReLU, Dropout, ZeroPadding2D, BatchNormalization, Flatten, Dense


def get_discriminator():
    img_shape = (28, 28, 1)
    
    # initialize model
    discriminator = Sequential()

    # add first convolutional layer
    discriminator.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding='same'))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    # add second convolutional layer
    discriminator.add(Conv2D(64, kernel_size=3, strides=2, padding='same'))
    discriminator.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    # add third convolutional layer
    discriminator.add(Conv2D(128, kernel_size=3, strides=2, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    # add fourth convolutional layer
    discriminator.add(Conv2D(256, kernel_size=3, strides=1, padding='same'))
    discriminator.add(BatchNormalization(momentum=0.8))
    discriminator.add(LeakyReLU(alpha=0.2))
    discriminator.add(Dropout(0.25))

    # add classification layer
    discriminator.add(Flatten())
    discriminator.add(Dense(1, activation='sigmoid'))

    # print discriminator summary
    discriminator.summary()

    # create model that takes in input and returns probability
    img = Input(shape=img_shape)
    probability = discriminator(img)

    return Model(img, probability)


if __name__ == '__main__':
    model = get_discriminator()
