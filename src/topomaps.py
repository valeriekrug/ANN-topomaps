"""
functions that are used for the topomap computation
but which are not exclusive for ANN topographic maps
"""

import numpy as np
import networkx as nx
from minisom import MiniSom
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import euclidean_distances
from umap import UMAP


def random_layout(n_neurons):
    """
    computes a random uniform layout
    :param n_neurons: number of neurons/feature maps
    :return: coordinates as ndarray
    """
    coordinates = np.random.uniform(0, 1, size=(n_neurons, 2))
    return coordinates


def make_threshold_graph(dist_mat, t, make_one_cluster=True):
    """
    creates a threshold graph from a distance matrix
    any two nodes are connected if their distance is below t
    :param dist_mat: condensed distance matrix
    :param t: distance threshold
    :param make_one_cluster: link small connected components to the largest component
    :return: nx.graph object
    """
    dist_mat = squareform(dist_mat)
    adj = np.zeros_like(dist_mat)
    adj[dist_mat < t] = 1
    graph = nx.from_numpy_matrix(adj)
    if make_one_cluster:
        conn_com = [list(x) for x in sorted(nx.connected_components(graph), key=len, reverse=True)]
        biggest_conn_com = conn_com[0]
        for sml_cluster in conn_com[1:]:
            red_dist_mat = dist_mat[biggest_conn_com,:]
            red_dist_mat = red_dist_mat[:, sml_cluster]

            n_big, n_small = red_dist_mat.shape

            min_idx = np.argmin(red_dist_mat)
            min_row_red = min_idx // n_small
            min_col_red = min_idx % n_small

            min_row = biggest_conn_com[min_row_red]
            min_col = sml_cluster[min_col_red]
            adj[min_row, min_col] = 1.0

    graph = nx.from_numpy_matrix(adj)
    return graph


def layout_graph(graph, as_array=True):
    """
    compute a graph layout and convert it to coordinates
    :param graph: nx.graph
    :param as_array: convert layout dict to ndarray
    :return: coordinates as dict, or ndarray if as_array=True
    """
    layout = nx.drawing.layout.fruchterman_reingold_layout(graph)
    if as_array:
        layout = np.array(list(layout.values()))
    return layout


def tanh(x, inv=False):
    tan = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if inv:
        tan = -tan
    return (tan + 1) / 2.


def train_PSO(dist_mat, n_steps=1000, pos_init=None):
    """
    Particle Swarm Optimization training
    :param dist_mat: condensed distance matrix
    :param n_steps: number of update steps
    :param pos_init: optional initial coordinates as ndarray. If given, global force weighting parameters are set to 0
    :return: coordinates as ndarray
    """
    dist_mat = squareform(dist_mat)
    if pos_init is None:
        positions = np.random.rand(dist_mat.shape[0], 2)
        a_glob_att = 1.5
        b_glob_att = 0.5
        c_glob_att = 2
    else:
        positions = np.copy(pos_init)
        a_glob_att = 0
        b_glob_att = 0
        #         c_glob_att = 0
        #         a_glob_att = 1.5
        #         b_glob_att = 0.5
        c_glob_att = 2

    attract = a_glob_att * (1 - (dist_mat / np.max(dist_mat)) ** 3)
    repulse = b_glob_att * np.exp(-(dist_mat / c_glob_att))

    f_glob = attract - repulse
    f_glob[np.isinf(f_glob)] = 0

    a_loc_att = 1.5
    b_loc_att = 15
    c_loc_att = 2

    i_cont = np.linspace(-3, 6, n_steps)

    for i in range(0, n_steps):
        # train step of the swarm
        f_g_coeff = tanh(i_cont[i], inv=True)
        f_l_coeff = tanh(i_cont[i])

        pairwise_differences = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]  # shape = (128,128,2)
        pairwise_distances = np.linalg.norm(pairwise_differences, axis=-1)  # (128,128)

        dist_mat[(dist_mat < 0.01)] = 0.01
        pairwise_distances[(pairwise_distances < 0.01)] = 0.01

        attract = a_loc_att * (1 / (pairwise_distances + 1) ** 3)
        repulse = b_loc_att * np.exp(-(pairwise_distances / c_loc_att))  # bounded repulsion; shape=(128,128)
        f = (attract - repulse)  # shape=(128,128)

        f = ((f * f_l_coeff) + (f_glob * f_g_coeff)) / 2

        x_ = pairwise_differences[:, :, 0] * f  # shape=(128,128)
        y_ = pairwise_differences[:, :, 1] * f
        x_ = x_.mean(axis=1)  # (128,)
        y_ = y_.mean(axis=1)
        x_ = x_[..., np.newaxis]  # (128,1)
        y_ = y_[..., np.newaxis]
        update = np.concatenate((x_, y_), axis=1)  # shape=(128, 2)
        # update = np.nan_to_num(update)
        positions += update

    return positions


def som_winners_to_coordinates(som, values):
    """
    Obtains coordinates for each element (neuron/feature map) in values according to winner SOM position.
    Multiple neurons matching to the same position are distributed in a circle around the coordinate.
    :param som: minisom object
    :param values: input values for which to find SOM positions
    :return: coordinates as ndarray
    """
    n_instances = values.shape[0]
    winners = som.win_map(values, return_indices=True)

    coordinates = np.zeros((n_instances, 2), 'float')
    for (x, y), indices in zip(winners.keys(), winners.values()):
        n_winners = len(indices)
        if n_winners > 1:
            # multiple winners are at identical coordinates
            # to distinguish them, the points are distributed uniformly on a small circle around the coordinate
            rad_per_point = (2 * np.pi) / n_winners
            random_rad_start = np.random.uniform(0, rad_per_point)
            rads = np.arange(random_rad_start, 2 * np.pi, rad_per_point)
            xshifts = np.sin(rads) / 5
            yshifts = np.cos(rads) / 5
        else:
            xshifts = [0]
            yshifts = [0]
        for i in range(n_winners):
            coordinates[indices[i]] = [x + xshifts[i], y + yshifts[i]]

    return coordinates


def train_SOM(values, n_epochs=10):
    """
    training a square SOM using the minisom package
    :param values: training values
    :param n_epochs: number of epochs
    :return: trained minisom object
    """
    n_instances, n_features = values.shape

    # three different size options
    # som_dim = int(np.sqrt(5*np.sqrt(values_shape[-1]))) # common rule of thumb for SOM
    # som_dim = som_dim//2  # manually adapting the rule of thumb to get less sparse topopmaps
    som_dim = int(np.sqrt(n_instances)) + 1  # available space for every channel

    if som_dim < 2:
        som_dim = 2

    som = MiniSom(som_dim,
                  som_dim,
                  n_features,
                  sigma=1,
                  learning_rate=0.5)

    som.random_weights_init(values)
    n_iter = int(n_epochs * n_instances)

    som.train(values, n_iter, random_order=True)

    # convert trained SOM into per-neuron coordinates like other topomaps
    coordinates = som_winners_to_coordinates(som, values)

    return coordinates


def compute_PCA(dist_mat):
    """
    Principal Component Analysis of a distance matrix
    :param dist_mat: condensed distance matrix
    :return:  first and second principal component as ndarray
    """
    dist_mat = squareform(dist_mat)
    pca = PCA(n_components=2)
    coordinates = pca.fit_transform(dist_mat)

    return coordinates


def compute_TSNE(dist_mat):
    """
    computes 2D tSNE
    :param dist_mat: condensed distance matrix
    :return: tSNE result as ndarray
    """
    dist_mat = squareform(dist_mat)
    return TSNE(init='pca', learning_rate='auto', n_components=2).fit_transform(dist_mat)


def compute_UMAP(dist_mat):
    """
    computes 2D UMAP
    :param dist_mat: condensed distance matrix
    :return: UMAP result as ndarray
    """
    dist_mat = squareform(dist_mat)
    return UMAP(n_components=2).fit_transform(dist_mat)


def rotate_via_numpy(xy, deg, scale=True):
    """
    rotates a ndarray of 2D coordinate values around the center of origin [0,0]
    :param xy: coordinates as ndarray
    :param deg: degree to rotate
    :param scale: after rotation, scale coordinates to [0,1] in both dimensions
    :return: coordinates as ndarray
    """
    radians = (deg * 2 * np.pi) / 360

    x, y = np.transpose(xy)
    c, s = np.cos(radians), np.sin(radians)
    j = np.matrix([[c, s], [-s, c]])
    m = np.dot(j, [x, y])

    m = np.transpose(m)

    if scale:
        m = m - np.min(m, 0)
        m = m / np.max(m, 0)

    return m


def get_all_rotations(xy, res=1):
    """
    compute rotations of the same values with different degrees
    :param xy: coordinates as ndarray
    :param res: degree difference between two consecutive steps
    :return: 3D ndarray of rotated coordinates
    """
    degs = np.arange(0, 360, res)[1:]
    n_degs = len(degs)

    coord_mat = np.zeros(shape=[n_degs + 1] + list(xy.shape))
    coord_mat[0] = xy
    for i, deg in enumerate(degs):
        coord_mat[i + 1] = rotate_via_numpy(xy, deg)

    return coord_mat


def get_rotation_dist_mats(reference, rotations):
    """
    compute distances between a reference layout and rotated query layouts
    :param reference: reference layout (2D ndarray)
    :param rotations: rotated query layouts (3D ndarray)
    :return: list of distance matrices between reference and rotation
    """
    dist_mats = np.zeros(shape=[rotations.shape[0]] + [reference.shape[0]] + [rotations.shape[1]])
    for i, rot in enumerate(rotations):
        dist_mats[i] = euclidean_distances(reference, rotations[i])
    return dist_mats
