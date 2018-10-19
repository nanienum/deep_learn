import os
import shutil
from keras import layers
from keras import models
from keras import optimizers
import image
from nn_utils import get_img_generators, plot_acc, plot_loss, timeit


def prepare_cats_dogs_dataset():
    original_dir = '/Users/o/data/cats_vs_dogs/train'
    base_dir = '/Users/o/data/cats_vs_dogs/cats_and_dogs_small/'
    os.makedirs(base_dir, exist_ok=True)

    train_dir = os.path.join(base_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    validation_dir = os.path.join(base_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)

    test_dir = os.path.join(base_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    train_cats_dir = os.path.join(train_dir, 'cats')
    os.makedirs(train_cats_dir, exist_ok=True)

    train_dogs_dir = os.path.join(train_dir, 'dogs')
    os.makedirs(train_dogs_dir, exist_ok=True)

    validation_cats_dir = os.path.join(validation_dir, 'cats')
    os.makedirs(validation_cats_dir, exist_ok=True)

    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    os.makedirs(validation_dogs_dir, exist_ok=True)

    test_cats_dir = os.path.join(test_dir, 'cats')
    os.makedirs(test_cats_dir, exist_ok=True)

    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.makedirs(test_dogs_dir, exist_ok=True)

    fnames = [f'cat.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'cat.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'cat.{i}.jpg' for i in range(1500, 2000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'dog.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'dog.{i}.jpg' for i in range(1000, 1500)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    fnames = [f'dog.{i}.jpg' for i in range(1000)]
    for fname in fnames:
        src = os.path.join(original_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

    print('total training cat images:', len(os.listdir(train_cats_dir)))
    print('total validation cat images:', len(os.listdir(validation_cats_dir)))
    print('total test cat images:', len(os.listdir(test_cats_dir)))

    print('total training dog images:', len(os.listdir(train_dogs_dir)))
    print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
    print('total test dog images:', len(os.listdir(test_dogs_dir)))


@timeit
def main():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])
    train_dir = "/Users/o/data/cats_vs_dogs/cats_and_dogs_small/train/"
    validation_dir = "/Users/o/data/cats_vs_dogs/cats_and_dogs_small/validation/"
    assert os.path.exists(train_dir)
    assert os.path.exists(validation_dir)

    train_generator, validation_generator = get_img_generators(train_dir, validation_dir)
    history = model.fit_generator(train_generator,
                        steps_per_epoch=100, epochs=30,
                        validation_data=validation_generator,
                        validation_steps=50, verbose=0)

    model.save("cats_dogs_small_1.h5")
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plot_acc(acc, val_acc)
    plot_loss(loss, val_loss)


if __name__ == '__main__':
    main()
