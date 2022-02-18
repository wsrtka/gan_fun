
from keras import Input, Model
from keras.optimizers import Adam

from dataset import get_dataset
from discriminator import get_discriminator
from generator import get_generator


def get_combined():
    # create optimizer
    optimizer = Adam(learning_rate=0.0002, beta=0.5)

    # build discriminator
    discriminator = get_discriminator()
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    discriminator.trainable = False

    # build generator
    generator = get_generator()

    # build combined
    z = Input(shape=(100,))
    img = generator(z)

    valid = discriminator(img)

    combined = Model(z, valid)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    return combined


if __name__ == '__main__':
    ...
