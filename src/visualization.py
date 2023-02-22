import os
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

def compute_linkage(cond_dist_mat, linkage_method='average'):
    linkage = hierarchy.linkage(cond_dist_mat,
                                method=linkage_method,
                                optimal_ordering=True)
    return linkage


def plot_dendrogram(linkage, layer, scale=2, width_factor=10, orientation='bottom', output_dir=None):
    fig, ax = plt.subplots(1, 1, figsize=(scale * width_factor, scale))
    hierarchy.dendrogram(linkage, orientation=orientation, ax=ax, color_threshold=0, above_threshold_color='#000000')
    ax.axis('off')
    if output_dir is None:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'dendrogram_layer' + str(layer) + '.pdf'),
                    format='pdf')
        plt.close(fig)
    plt.show()


def get_dendrogram_order(linkage):
    dendrogram = hierarchy.dendrogram(linkage, no_plot=True)
    dendrogram_order = np.asarray(dendrogram['leaves'], dtype='int32')
    return dendrogram_order


def compute_distance_metric(values, metric):
    dist_mat = pdist(values, metric)
    values = None
    return dist_mat
