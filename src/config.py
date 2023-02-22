import json
import os

def assert_bool(par, name):
    assert type(par) == bool, \
        "'" + name + "' must be of type bool. Given type: " + str(type(par)) + \
        ", Given value: " + str(par) + "."


def assert_int(par, name):
    assert type(par) == int, \
        "'" + name + "' must be of type int. Given type: " + str(type(par)) + \
        ", Given value: " + str(par) + "."


def assert_number(par, name):
    assert type(par) == int or type(par) == float, \
        "'" + name + "' must be of type int or float. Given type: " + str(type(par)) + \
        ", Given value: " + str(par) + "."


def assert_element_of(pars, allowed_values, name, each_element=False):
    prefix = "Each element in "
    if not each_element:
        pars = [pars]
        prefix = ""
    for par in pars:
        assert par in allowed_values, \
            prefix + "'" + name + "' must be in " + str(allowed_values) + \
            ". \nGiven: " + str(par)


def check_key_completeness(config_dict):
    with open("configs/default_values.json") as config_file:
        defaults = json.load(config_file)

    # required keys
    required_keys = ["model", "data"]
    for key in required_keys:
        assert key in config_dict.keys(), key + " is a required config key"

    # set defaults for missing keys
    level_0_keys = ['layers', 'distance_metric', 'nap_params', 'topomap_params', 'plot_params']
    for key in level_0_keys:
        if key not in config_dict:
            config_dict[key] = defaults[key]

    nap_param_keys = ["n_random_examples", "error_mode"]
    for key in nap_param_keys:
        if key not in config_dict["nap_params"]:
            config_dict["nap_params"][key] = defaults["nap_params"][key]

    topomap_param_keys = ["neuron_activations", "methods"]
    for key in topomap_param_keys:
        if key not in config_dict["topomap_params"]:
            config_dict["topomap_params"][key] = defaults["topomap_params"][key]

    if "use_naps" not in config_dict["topomap_params"]["neuron_activations"].keys():
        config_dict["topomap_params"]["neuron_activations"]["use_naps"] = \
            defaults["topomap_params"]["neuron_activations"]["use_naps"]

    plot_param_keys = ["plot_inputs", "avg_inputs", "ordered", "scale", "use_title", "plot_individually"]
    for key in plot_param_keys:
        if key not in config_dict["plot_params"]:
            config_dict["plot_params"][key] = defaults["plot_params"][key]

    return config_dict


def check_general_params(config_dict):
    params = config_dict

    model = params['model']
    training_data, model_type = model.split('-')
    allowed_model_values = [('MNIST', 'MLP'), ('MNIST', 'CNN_SHALLOW'), ('FMNIST', 'MLP'),
                            ('FMNIST', 'CNN_SHALLOW'), ('FMNIST', 'CNN_DEEP_FMNIST'), ('CIFAR10', 'CNN_DEEP'),
                            ('DEF_MNIST', 'MLP'), ('DEF_FMNIST', 'CNN_DEEP_FMNIST'), ('FAIRFACE', 'VGG16')]
    assert_element_of((training_data, model_type), allowed_model_values, "model")

    data = params['data']
    allowed_dataset_values = ['MNIST', 'FMNIST', 'CIFAR10', 'DEF_MNIST', 'DEF_FMNIST', 'DEF_FMNIST2', 'FAIRFACE']
    assert_element_of(data, allowed_dataset_values, "data")

    layers = params['layers']
    layers_per_model = {'MLP': 4,
                        'CNN_SHALLOW': 6,
                        'CNN_DEEP_FMNIST': 10,
                        "CNN_DEEP": 10,
                        'VGG16': 23}
    assert type(layers) == list, \
        "'layers' must be of type list. Given: " + str(layers)
    for layer in layers:
        assert type(layer) == int, \
            "each entry in 'layers' must be of type int. Given: " + str(layer)
        assert 0 <= layer <= layers_per_model[model_type], \
            "Requested layer index " + str(layer) + " is not in model " + str(model) + "."

    distance_metric = params['distance_metric']
    allowed_distance_metrics = ['cosine', 'euclidean']
    assert_element_of(distance_metric, allowed_distance_metrics, "distance_metric")

    config_dict = params
    return config_dict


def check_nap_params(config_dict):
    params = config_dict['nap_params']

    n_random_examples = params['n_random_examples']
    assert_int(n_random_examples, "nap_params:n_random_examples")

    error_mode = params['error_mode']
    allowed_error_modes = ['None', 'binary_split', 'binary_contrast', 'confusion_matrix']
    assert_element_of(error_mode, allowed_error_modes, "nap_params:error_mode")

    config_dict['nap_params'] = params
    return config_dict


def check_topomap_params(config_dict):
    params = config_dict['topomap_params']
    use_naps = params['neuron_activations']['use_naps']
    assert_bool(use_naps, "topomap_params:neuron_activations:use_naps")

    if use_naps:
        params['neuron_activations']['n_random_examples'] = None
        params['neuron_activations']['balance_by_label'] = None
        params['neuron_activations']['use_correct_only'] = None
    else:
        n_random_examples = params['neuron_activations']['n_random_examples']
        assert_int(n_random_examples, "topomap_params:neuron_activations:n_random_examples")

        balance_by_label = params['neuron_activations']['balance_by_label']
        assert_bool(balance_by_label, "topomap_params:neuron_activations:balance_by_label")

        use_correct_only = params['neuron_activations']['use_correct_only']
        assert_bool(use_correct_only, "topomap_params:neuron_activations:use_correct_only")

    topomap_methods = params['methods']
    allowed_topomap_methods = ['random', 'graph', 'PSO', 'graph_PSO', 'SOM', 'SOM_PSO',
                               'PCA', 'PCA_PSO', 'TSNE', 'TSNE_PSO', 'UMAP', 'UMAP_PSO']
    assert_element_of(topomap_methods, allowed_topomap_methods, "topomap_params:methods", each_element=True)

    config_dict['topomap_params'] = params
    return config_dict


def check_plot_params(config_dict):
    params = config_dict['plot_params']
    plot_inputs = params['plot_inputs']
    assert_bool(plot_inputs, "plot_params:plot_inputs")

    avg_inputs = params['avg_inputs']
    assert_bool(avg_inputs, "plot_params:avg_inputs")

    ordered = params['ordered']
    assert_bool(ordered, "plot_params:ordered")

    scale = params['scale']
    assert_number(scale, "plot_params:scale")

    use_title = params['use_title']
    assert_bool(use_title, "plot_params:use_title")

    plot_individually = params['plot_individually']
    assert_bool(plot_individually, "plot_params:plot_individually")

    config_dict['plot_params'] = params
    return config_dict


def load_config(path):
    print(os.getcwd())
    with open(path) as config_file:
        config_dict = json.load(config_file)

    config_dict = check_key_completeness(config_dict)

    config_dict = check_general_params(config_dict)
    config_dict = check_nap_params(config_dict)
    config_dict = check_topomap_params(config_dict)
    config_dict = check_plot_params(config_dict)

    return config_dict
