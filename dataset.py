
import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import fashion_mnist


def get_dataset():
    (training_data, _), (_, _) = fashion_mnist.load_data()

    X_train = (training_data / 127.5) - 1
    X_train = np.expand_dims(X_train, axis=3)

    return X_train


def visualize_input(img, ax):
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max() / 2.5

    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y], 2)), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img [x][y] < thresh else 'black')


if __name__ == '__main__':
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    (training_data, _), (_, _) = fashion_mnist.load_data()
    visualize_input(training_data[3343], ax)
    plt.show()
