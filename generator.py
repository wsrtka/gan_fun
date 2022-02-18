
from keras import Sequential, Input, Model
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, BatchNormalization, Activation


def get_generator():
    generator = Sequential()

    # add upsampling layer
    generator.add(Dense(128 * 7 * 7, activation='relu', input_dim=100))
    generator.add(Reshape((7, 7, 128)))
    generator.add(UpSampling2D(size=(2, 2)))

    # add first convolutional layer
    generator.add(Conv2D(128, kernel_size=3, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation('relu'))
    generator.add(UpSampling2D(size=(2, 2)))

    # add second convolutional layer
    generator.add(Conv2D(1, kernel_size=3, padding='same'))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Activation('relu'))

    # add third convolutional layer
    generator.add(Conv2D(1, kernel_size=3, padding='same'))
    generator.add(Activation('tanh'))

    # print generator summary
    generator.summary()

    # create model with noise input and fake image output
    noise = Input(shape=(100,))
    fake_image = generator(noise)

    return Model(noise, fake_image)


if __name__ == '__main__':
    get_generator()
