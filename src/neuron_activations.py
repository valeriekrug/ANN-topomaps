from src.models import train_or_load_model, load_data
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def get_random_inputs(data, n, example_ids, subset='test'):
    example_idx = example_ids[np.random.permutation(len(example_ids))[:n]]
    example_inputs = data[subset]['inputs'][example_idx]
    return example_inputs


def get_random_inputs_per_class(data, n, class_ids, subset='test'):
    example_inputs = dict()
    for c_id in class_ids:
        class_examples = np.argwhere(data[subset]['labels'] == c_id)[:, 0]
        example_idx = class_examples[np.random.permutation(len(class_examples))[:n]]
        example_inputs[str(c_id)] = data[subset]['inputs'][example_idx]
    return example_inputs


def get_activations(model, layer, inputs):
    if type(inputs) == dict:
        activations = dict()
        for c_id in inputs.keys():
            acts = model(inputs[c_id])[layer].numpy()
            activations[c_id] = acts
    else:
        acts = model(inputs)[layer].numpy()
        activations = acts
    return activations


def compute_NAPs(activations, normalize=True):
    NAPs = []
    for c_id in activations.keys():
        nap = np.mean(activations[c_id], 0)
        NAPs.append(nap)
    NAPs = np.stack(NAPs)

    if normalize:
        normalizer = np.expand_dims(np.mean(NAPs, 0), 0)
        NAPs = NAPs - normalizer

        # add eps to avoid inf for distance computation for zero-only neurons/feature maps
    NAPs = NAPs + 1e-15

    return NAPs

# TODO: create_video_frames --> create_checkpoints
def get_NAPs(model, data, layers, n_random_examples, return_inputs=True, error_mode=None, create_video_frames=False, checkpoint_path=None):
    model = train_or_load_model(model.split('-')[0], model.split('-')[1], True, create_video_frames=create_video_frames, checkpoint_path=checkpoint_path)
    data = load_data(data)

    n_classes = len(np.unique(data['test']['labels']))

    if error_mode != "None" and error_mode is not None:
        lbls = data['test']['labels']
        if data['test']['inputs'].shape[1] == 224:
            preds = lbls
        else:
            preds = np.argmax(model(data['test']['inputs'])[-1], 1)

        example_inputs = dict()
        if error_mode in ['binary_split', 'binary_contrast']:
            for lbl in np.unique(lbls):
                correct_ids = np.argwhere(np.logical_and(lbls == lbl, preds == lbl))[:, 0]
                example_inputs[str(lbl) + '_correct'] = get_random_inputs(data, n_random_examples, correct_ids)
                wrong_ids = np.argwhere(np.logical_and(lbls == lbl, preds != lbl))[:, 0]
                example_inputs[str(lbl) + '_wrong'] = get_random_inputs(data, n_random_examples, wrong_ids)
        elif error_mode == 'confusion_matrix':
            for lbl1 in np.unique(lbls):
                for lbl2 in np.unique(lbls):
                    ex_ids = np.argwhere(np.logical_and(lbls == lbl1, preds == lbl2))[:, 0]
                    if len(ex_ids) > 0:
                        example_inputs[str(lbl1) + '_' + str(lbl2)] = get_random_inputs(data, n_random_examples, ex_ids)

    else:
        example_inputs = get_random_inputs_per_class(data, n_random_examples, np.arange(n_classes))

    NAPs = {'values': dict(), 'error_mode': error_mode}
    for layer in layers:
        activations = get_activations(model, layer, example_inputs)
        layer_NAPs = compute_NAPs(activations)

        NAPs['values'][layer] = layer_NAPs

    if not return_inputs:
        return NAPs
    else:
        return [NAPs, example_inputs]

def get_NAPs_of_checkpoint(model, data, layers, n_random_examples, return_inputs=True, error_mode=None, checkpoint_path=None):
    buffer = keras.models.load_model(checkpoint_path)
    model = keras.Model(inputs=buffer.inputs, outputs=[layer.output for layer in buffer.layers])
    data = load_data(data)

    n_classes = len(np.unique(data['test']['labels']))

    if error_mode != "None" and error_mode is not None:
        lbls = data['test']['labels']
        if data['test']['inputs'].shape[1] == 224:
            preds = lbls
        else:
            preds = np.argmax(model(data['test']['inputs'])[-1], 1)

        example_inputs = dict()
        if error_mode in ['binary_split', 'binary_contrast']:
            for lbl in np.unique(lbls):
                correct_ids = np.argwhere(np.logical_and(lbls == lbl, preds == lbl))[:, 0]
                example_inputs[str(lbl) + '_correct'] = get_random_inputs(data, n_random_examples, correct_ids)
                wrong_ids = np.argwhere(np.logical_and(lbls == lbl, preds != lbl))[:, 0]
                example_inputs[str(lbl) + '_wrong'] = get_random_inputs(data, n_random_examples, wrong_ids)
        elif error_mode == 'confusion_matrix':
            for lbl1 in np.unique(lbls):
                for lbl2 in np.unique(lbls):
                    ex_ids = np.argwhere(np.logical_and(lbls == lbl1, preds == lbl2))[:, 0]
                    if len(ex_ids) > 0:
                        example_inputs[str(lbl1) + '_' + str(lbl2)] = get_random_inputs(data, n_random_examples, ex_ids)

    else:
        example_inputs = get_random_inputs_per_class(data, n_random_examples, np.arange(n_classes))

    NAPs = {'values': dict(), 'error_mode': error_mode}
    for layer in layers:
        activations = get_activations(model, layer, example_inputs)
        layer_NAPs = compute_NAPs(activations)

        NAPs['values'][layer] = layer_NAPs

    model = None
    data = None
    
    if not return_inputs:
        return NAPs
    else:
        return [NAPs, example_inputs]

def get_neuron_activations(model, data, layers, neuron_activation_params):
    n_random_examples = neuron_activation_params["n_random_examples"]
    balance_by_label = neuron_activation_params["balance_by_label"]
    use_correct_only = neuron_activation_params["use_correct_only"]

    model = train_or_load_model(model.split('-')[0], model.split('-')[1])
    data = load_data(data)

    n_classes = len(np.unique(data['test']['labels']))
    n_examples = len(data['test']['labels'])

    lbls = data['test']['labels']
    lbl_classes = np.unique(lbls)
    preds = np.argmax(model(data['test']['inputs'])[-1], 1)

    example_inputs = list()

    if balance_by_label:
        n_random_examples = n_random_examples // n_classes

        for lbl in lbl_classes:
            if use_correct_only:
                ids = np.argwhere(np.logical_and(lbls == lbl, preds == lbl))[:, 0]
            else:
                ids = np.argwhere(lbls == lbl)[:, 0]
            example_inputs.append(get_random_inputs(data, n_random_examples, ids))

        example_inputs = np.concatenate(example_inputs)
    else:
        unbalanced_n_random_examples = np.random.uniform(size=len(lbl_classes))
        unbalanced_n_random_examples = unbalanced_n_random_examples * (n_random_examples / np.sum(unbalanced_n_random_examples))
        unbalanced_n_random_examples = unbalanced_n_random_examples.astype('int')
        for lbl, n_ex in zip(lbl_classes, unbalanced_n_random_examples):
            ids = np.argwhere(lbls == lbl)[:, 0]
            example_inputs.append(get_random_inputs(data, n_ex, ids))

        example_inputs = np.concatenate(example_inputs)

    neuron_activations = dict()
    for layer in layers:
        layer_neuron_activations = get_activations(model, layer, example_inputs)

        neuron_activations[layer] = layer_neuron_activations + 1e-15

    return neuron_activations

def get_neuron_activations_of_checkpoint(model, data, layers, neuron_activation_params, checkpoint_path):
    n_random_examples = neuron_activation_params["n_random_examples"]
    balance_by_label = neuron_activation_params["balance_by_label"]
    use_correct_only = neuron_activation_params["use_correct_only"]

    #model = train_or_load_model(model.split('-')[0], model.split('-')[1])
    
    # TODO: Add this to model.py and import - makes more sence
    buffer = keras.models.load_model(checkpoint_path)
    model = keras.Model(inputs=buffer.inputs, outputs=[layer.output for layer in buffer.layers])
    
    data = load_data(data)

    n_classes = len(np.unique(data['test']['labels']))
    n_examples = len(data['test']['labels'])

    lbls = data['test']['labels']
    lbl_classes = np.unique(lbls)
    preds = np.argmax(model(data['test']['inputs'])[-1], 1)

    example_inputs = list()

    if balance_by_label:
        n_random_examples = n_random_examples // n_classes

        for lbl in lbl_classes:
            if use_correct_only:
                ids = np.argwhere(np.logical_and(lbls == lbl, preds == lbl))[:, 0]
            else:
                ids = np.argwhere(lbls == lbl)[:, 0]
            example_inputs.append(get_random_inputs(data, n_random_examples, ids))

        example_inputs = np.concatenate(example_inputs)
    else:
        unbalanced_n_random_examples = np.random.uniform(size=len(lbl_classes))
        print(n_random_examples)
        print(np.sum(unbalanced_n_random_examples))
        unbalanced_n_random_examples = unbalanced_n_random_examples * (n_random_examples / np.sum(unbalanced_n_random_examples))
        unbalanced_n_random_examples = unbalanced_n_random_examples.astype('int')
        for lbl, n_ex in zip(lbl_classes, unbalanced_n_random_examples):
            ids = np.argwhere(lbls == lbl)[:, 0]
            example_inputs.append(get_random_inputs(data, n_ex, ids))

        example_inputs = np.concatenate(example_inputs)

    neuron_activations = dict()
    for layer in layers:
        layer_neuron_activations = get_activations(model, layer, example_inputs)

        neuron_activations[layer] = layer_neuron_activations + 1e-15

    return neuron_activations
