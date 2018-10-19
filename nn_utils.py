import numpy as np
from time import time
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time()
        func(*args, **kwargs)
        end = time()
        take = end - start
        print("elapsed:", round(take, 2))
    return wrapper


def vectorize_sequences(sequences, dimesion=10000):
    results = np.zeros((len(sequences), dimesion))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def plot_loss(loss_values, val_loss_values):
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("train_val_loss_text.png", dpi=300)
    plt.show()


def plot_acc(acc_values, val_acc_values):
    epochs = range(len(acc_values))
    plt.clf()
    plt.plot(epochs, acc_values, 'bo', label='Training acc')
    plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("train_val_acc_text.png", dpi=300)
    plt.show()


def get_img_generators(train_dir, validation_dir):
    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    return train_generator, validation_generator
