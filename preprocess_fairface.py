import os

import argparse
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


def _parse_function(file_name, lbls):
    image_string = tf.io.read_file(file_name)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    return image, lbls

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to fairface data set', required=True)
parser.add_argument('-o', '--output', help='output path to store tf-data', required=True)

args = parser.parse_args()
# specify the directory in which the fair face data set is stored
data_base_dir = args.input
# provide the data set output path
# and use this path as "data_path" config parameter when using fair face
data_out_dir = args.output

for data_split_subset in ["train", "val"]:
    raw_data_path = os.path.join(data_base_dir, data_split_subset)
    onlyfiles = [f for f in os.listdir(raw_data_path) if os.path.isfile(os.path.join(raw_data_path, f))]

    # training labels are given in cvs file
    label_csv_path = os.path.join(data_base_dir, 'fairface_label_' + data_split_subset + '.csv')
    label_df = pd.read_csv(label_csv_path)
    lb_make = LabelEncoder()
    # transform string to int
    label_df["race_int"] = lb_make.fit_transform(label_df["race"])
    train_labels = label_df["race_int"].values

    lbls_list = []
    file_paths = []

    for file in tqdm(onlyfiles, desc="preprocessing "+ data_split_subset + " subset"):
        file_name = os.path.join(data_split_subset, file)
        file_path = os.path.join(data_base_dir, file_name)
        file_paths.append(file_path)
        lbls_list.append(label_df.loc[label_df['file'] == file_name]['race_int'].values[0])

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, lbls_list))

    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(32)

    # save training data
    output_dir = os.path.join(data_out_dir, data_split_subset)
    tf.data.experimental.save(dataset, output_dir)
