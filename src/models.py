import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, BatchNormalization, Dense
import shutil
import numpy as np
import copy
import pickle

def make_model(model_type):
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
            #                     Activation('relu'),
            Conv2D(filters=96, kernel_size=(3, 3), strides=2, activation='relu'),
            #                     Activation('relu'),
            Dropout(0.2),
            Conv2D(filters=192, kernel_size=(3, 3), activation='relu'),
            #                     Activation('relu'),
            Conv2D(filters=192, kernel_size=(3, 3), strides=2, activation='relu'),
            #                     Activation('relu'),
            Dropout(0.5),
            Flatten(),
            BatchNormalization(),
            Dense(256, activation='relu'),
            #                     Activation('relu'),
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
    images = []
    labels = []
    for st, (img, lbl) in enumerate(dataset):
        if st > 100:
            break
        images.append(img)
        labels.append(lbl)
    return np.concatenate(images, axis=0), np.concatenate(labels, axis=0)


def add_random_labels(test_labels):
    num_classes = len(np.unique(test_labels))

    for class_id in range(num_classes):
        idx_cid = [i for i, x in enumerate(test_labels) if x == class_id]  # np.argwhere[test_lbl==class_id]
        for count, idx in enumerate(idx_cid):
            if count > len(idx_cid) / 2:
                break
            test_labels[idx] = np.random.randint(num_classes, num_classes + num_classes)
    return test_labels


def load_fairface(random_lbls=False):
    train_dataset = tf.data.experimental.load(
        '/project/ratul/awl/temp_post_hoc_topomap/face_rec_stuff/VGGFace2-ResNet50-tf2/data/tf_data/train/')
    test_dataset = tf.data.experimental.load(
        '/project/ratul/awl/temp_post_hoc_topomap/face_rec_stuff/VGGFace2-ResNet50-tf2/data/tf_data/val/')

    train_images, train_labels = from_tfdata_to_ndarray(train_dataset)
    test_images, test_labels = from_tfdata_to_ndarray(test_dataset)

    if random_lbls:
        test_labels = add_random_labels(test_labels)

    return (train_images, train_labels), (test_images, test_labels)


def load_data(dataset):
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
        loader_function = load_fairface()
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

class CustomCallback(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, save_freq):
        self.model_name = filepath
        self.save_freq = save_freq
        super().__init__(self.model_name, save_freq=self.save_freq)
            
    def on_train_batch_end(self, batch, logs=None):
        if self._should_save_on_batch(batch):
            print("trying to save checkpoint")
            filename = self.model_name + "epoch_" + str(self._current_epoch+1) + "_batch_" + str(f'{(batch+1):05}') + '.tf'
            self.model.save(filename)
            print("\nsaved checkpoint: " + filename + "\n")
            

class EvaluateBatchEnd(tf.keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data 
        x, y = self.test_data
        self.unique_classes = np.unique(y)
        self.amount_classes = len(self.unique_classes)
        print(x.shape)
        print(x[0].shape)
        
    def on_train_batch_end(self, epoch, logs={}):
        x, y = self.test_data
        acc_of_classes = []
        if(os.path.exists('accuracies.pkl')):
            with open('accuracies.pkl', 'rb') as f: 
                acc_of_classes = pickle.load(f)
            
        this_batch_acc_of_classes = []
        for i in range(self.amount_classes):
            #CIFAR STUFF
            #all_indices, _ = np.where(y==self.unique_classes[i])
            all_indices = np.where(y==self.unique_classes[i])
            class_x = x[all_indices]
            class_y = y[all_indices]
            #print('\nFound {} datapoints in class {}'.format(len(class_x), i))
            scores = self.model.evaluate(class_x, class_y, verbose=False)
            this_batch_acc_of_classes.append(scores[1])
            #print('\nTesting loss: {}, accuracy: {} of class: {}\n'.format(scores[0], scores[1], i))
        
        acc_of_classes.append(this_batch_acc_of_classes)
        with open('accuracies.pkl', 'wb') as f:
                pickle.dump(acc_of_classes, f)
                
def train_or_load_model(dataset, model_type, overwrite_model=False, create_video_frames=False, checkpoint_path=None):
    #model_path = '/project/ankrug/posthoc_topomaps/ANNScan/models/' + dataset + '_' + model_type
    model_path = 'models/' + dataset + '_' + model_type

    if overwrite_model and os.path.isdir(model_path):
        shutil.rmtree(model_path)

    if os.path.isdir(model_path): 
        model = keras.models.load_model(model_path)

    else:
        #         if dataset=='MNIST':
        data = load_data(dataset)

        model = make_model(model_type)


        # raise ValueError(
        #     'In this workshop we do not want to retrain models.
        #     Make sure you created the shortcut to the CogXAI_models directory properly.')
        if model_type != 'VGG16':
            callback_list = []
            if create_video_frames:
                SAVE_FREQ = 1 # number of batches 
                custom_callback = CustomCallback(filepath=checkpoint_path, save_freq=SAVE_FREQ)
                callback_list.append(custom_callback)
                accuracy_saver = EvaluateBatchEnd((data['test']['inputs'], data['test']['labels']))
                callback_list.append(accuracy_saver)
            model.fit(
                x=data['train']['inputs'],
                y=data['train']['labels'],
                epochs=5 if dataset == 'CIFAR10' else 1,
                callbacks=callback_list,
                validation_data=(data['test']['inputs'], data['test']['labels'])
            )

            os.makedirs(model_path)
            
            model.save(model_path)

    extractor = keras.Model(inputs=model.inputs,
                            outputs=[layer.output for layer in model.layers])
    #                             outputs=outputs)
    #                             outputs=[layer.output for layer in model.layers if layer.output.name[:5] == 'conv2' or layer.output.name[:5] == 'dense'])

    return extractor

def load_data_and_model(dataset, model_type, overwrite_model=False):
    data = load_data(dataset)

    model = train_or_load_model(dataset, model_type, overwrite_model)
    return data, model
