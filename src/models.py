"""
functions for loading specific data sets and models
"""

import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, BatchNormalization, Dense
import shutil
import numpy as np
import copy


def make_model(model_type):
    """
    creating models based on pre-defined architectures
    :param model_type: str, type of model ["MLP", "CNN_SHALLOW", "CNN_DEEP", "CNN_DEEP_FMNIST", "VGG16"]
    :return: compiled keras model
    """
    if model_type == 'MLP':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
            tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                  name='layer_2'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(10)
        ])
    elif model_type == 'CNN_SHALLOW':
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(128, 3, 2,
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001),
                                   input_shape=(28, 28, 1)),
            tf.keras.layers.SpatialDropout2D(0.5),
            tf.keras.layers.Conv2D(128, 3, 2,
                                   activation='relu',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.SpatialDropout2D(0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10)
        ])
    elif model_type == 'CNN_DEEP' or model_type == 'CNN_DEEP_FMNIST':
        model = tf.keras.models.Sequential([
            Conv2D(input_shape=(32, 32, 3) if model_type == 'CNN_DEEP' else (28, 28, 1),
                   filters=96, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=96, kernel_size=(3, 3), strides=2, activation='relu'),
            Dropout(0.2),
            Conv2D(filters=192, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=192, kernel_size=(3, 3), strides=2, activation='relu'),
            Dropout(0.5),
            Flatten(),
            BatchNormalization(),
            Dense(256, activation='relu'),
            Dense(10)
        ])
    elif model_type == 'VGG16':
        model = keras.applications.vgg16.VGG16()
    else:
        print('Please only input one of the following:\n1. MLP\n2. CNN_SHALLOW\n3. CNN_DEEP\n4. VGG16')
        return None

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def spoil_labels(train_lbls, labels, to_labels, how_many=10):
    """
    introducing annotation errors for generating toy examples
    :param train_lbls: list/ndarray of data labels
    :param labels: label id that shall be altered
    :param to_labels: label id that shall be assigned
    :param how_many: number of labels to change
    :return: data labels with introduced noise
    """
    noisy_lbls = copy.copy(train_lbls)
    n_data = len(train_lbls)
    random_ind = np.random.choice(n_data, n_data, replace=False)

    for label in labels:
        for label_to in to_labels:
            c = 0
            for i in random_ind:
                if c == how_many:
                    break
                if noisy_lbls[i] == label and noisy_lbls[i] == train_lbls[i]:
                    noisy_lbls[i] = label_to
                    c += 1
    return noisy_lbls


def from_tfdata_to_ndarray(dataset):
    """
    converting tf data batches into ndarray to obtain a common format
    """
    images = []
    labels = []
    for st, (img, lbl) in enumerate(dataset):
        if st > 100:
            break
        images.append(img)
        labels.append(lbl)
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def add_random_labels(test_labels):
    """
    add random groups as control groups by introducing new labels
    :param test_labels: original labels
    :return: labels with introduced random groups
    """
    num_classes = len(np.unique(test_labels))

    for class_id in range(num_classes):
        idx_cid = [i for i, x in enumerate(test_labels) if x == class_id]  # np.argwhere(test_lbl==class_id)
        for count, idx in enumerate(idx_cid):
            if count > len(idx_cid) / 2:
                break
            test_labels[idx] = np.random.randint(num_classes, num_classes + num_classes)
    return test_labels


def load_fairface(data_path, random_lbls=False):
    if data_path is None:
        raise ValueError('FairFace data set path must be given in load_data()')
    elif not os.path.isdir(data_path):
        raise ValueError('FairFace data set not available at: ' + data_path)
    train_dataset = tf.data.experimental.load(os.path.join(data_path, 'train/'))
    test_dataset = tf.data.experimental.load(os.path.join(data_path, 'val/'))

    train_images, train_labels = from_tfdata_to_ndarray(train_dataset)
    test_images, test_labels = from_tfdata_to_ndarray(test_dataset)

    if random_lbls:
        test_labels = add_random_labels(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def load_data(dataset, data_path=None):
    """
    loading and preprocessing selected data sets
    :param dataset: name of the data set
    :param data_path: path to the FairFace data set
    :return: data set as nested dict
    """
    if dataset == 'MNIST' or dataset == 'DEF_MNIST':
        loader_function = tf.keras.datasets.mnist.load_data()
        image_shape = [28, 28, 1]
    elif dataset == 'FMNIST' or dataset == 'DEF_FMNIST' or dataset == 'DEF_FMNIST2':
        loader_function = tf.keras.datasets.fashion_mnist.load_data()
        image_shape = [28, 28, 1]
    elif dataset == 'CIFAR10':
        loader_function = tf.keras.datasets.cifar10.load_data()
        image_shape = [32, 32, 3]
    elif dataset == 'FAIRFACE':
        loader_function = load_fairface(data_path)
        image_shape = [224, 224, 3]
    else:
        print(
            'Please only input one of the following:\n1. MNIST\n2. FMNIST\n3. CIFAR\n4. DEF_MNIST\n5. DEF_FMNIST\n6. DEF_FMNIST2\n7. FAIRFACE')
        return None

    # load files from npz
    (train_images, train_labels), (test_images, test_labels) = loader_function

    if dataset == 'DEF_MNIST' or dataset == 'DEF_FMNIST':
        train_labels = spoil_labels(train_labels, [0], [1],
                                    int(len(train_labels[train_labels == 0]) * 0.9))
        test_labels = spoil_labels(test_labels, [0], [1],
                                   int(len(test_labels[test_labels == 0]) * 0.9))
    elif dataset == 'DEF_FMNIST2':
        train_labels = spoil_labels(train_labels, [7], [9],
                                    int(len(train_labels[train_labels == 7]) * 0.9))
        test_labels = spoil_labels(test_labels, [7], [9],
                                   int(len(test_labels[test_labels == 7]) * 0.9))

    # reshape to 4 dim to be used for model training
    if dataset != 'FAIRFACE':
        train_images = train_images.reshape([train_images.shape[0]] + image_shape)
        test_images = test_images.reshape([test_images.shape[0]] + image_shape)

        # convert images to float in range [0,1]
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    out_dict = {'train': {'inputs': train_images,
                          'labels': train_labels},
                'test': {'inputs': test_images,
                         'labels': test_labels}}
    return out_dict


def train_or_load_model(dataset, model_type, overwrite_model=False, data_path=None):
    """
    training or loading a model
    :param dataset: data set name to use for training
    :param model_type: name of model type
    :param overwrite_model: whether to overwrite existing models
    :param data_path: path to the FairFace data set
    :return: keras model with per-layer activation output
    """
    model_path = 'models/' + dataset + '_' + model_type

    if overwrite_model and os.path.isdir(model_path):
        shutil.rmtree(model_path)

    if os.path.isdir(model_path):
        model = keras.models.load_model(model_path)

    else:
        data = load_data(dataset, data_path=data_path)

        model = make_model(model_type)

        if model_type != 'VGG16':
            model.fit(
                x=data['train']['inputs'],
                y=data['train']['labels'],
                epochs=50 if dataset == 'CIFAR10' else 20,
                validation_data=(data['test']['inputs'], data['test']['labels'])
            )

            os.makedirs(model_path)
            model.save(model_path)

    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])

    return extractor
