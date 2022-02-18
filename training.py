
import numpy as np
import matplotlib.pyplot as plt

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


def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(0, 1, size=[examples, dim])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(examples, 28, 28)

    plt.figure(figsize=figsize)

    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'results/gan_generated_image_epoch_{epoch}.png')


def train(epochs: int, X_train, generator: Model, discriminator: Model, combined: Model, batch_size=128, save_interval=50):
    # ground truth vectors
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        # train discriminator
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)

        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # train combined network
        g_loss = combined.train_on_batch(noise, valid)

        print(f'Epoch {epoch+1}/{epochs} [D loss: {d_loss[0]}, acc: {d_loss[1]}] [G loss: {g_loss}]')

        if epoch % save_interval == 0:
            plot_generated_images(epoch, generator)


if __name__ == '__main__':
    ...
