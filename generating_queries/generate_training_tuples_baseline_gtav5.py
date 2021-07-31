import os
import pickle
import random
import sys
from glob import glob

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from registration import registration_withinit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = '../benchmark_datasets/'

runs_folder = "GTA5/"
filename = "pointcloud_locations_20m_10overlap.csv"
pointcloud_fols = "/pcl/"

all_folders = sorted(glob(os.path.join(BASE_DIR, base_path, runs_folder) + '/round*'))
for i in range(len(all_folders)):
    all_folders[i] = os.path.basename(all_folders[i])

# 正样本范围 点云单位上正负1
pos_dis = 0.5
# 负样本范围
neg_dis = 4


def check_in_test_set(northing, easting, points, x_width, y_width):
    in_test_set = False
    for point in points:
        if (point[0] - x_width < northing and northing < point[0] + x_width and point[
            1] - y_width < easting and easting < point[1] + y_width):
            in_test_set = True
            break
    return in_test_set


##########################################


def construct_query_dict(df_centroids, filename):
    tree = KDTree(df_centroids[['northing', 'easting', 'altitude']])
    ind_nn = tree.query_radius(df_centroids[['northing', 'easting', 'altitude']], r=pos_dis)
    ind_r = tree.query_radius(df_centroids[['northing', 'easting', 'altitude']], r=neg_dis)
    queries = {}
    for i in tqdm(range(len(ind_nn))):
        query = df_centroids.iloc[i]["file"]
        positives = np.setdiff1d(ind_nn[i], [i]).tolist()
        list_neg = np.setdiff1d(df_centroids.index.values.tolist(), ind_r[i]).tolist()
        positives_T = []
        for pos_i in range(len(positives)):
            trans_pre = np.asarray(df_centroids.iloc[i][['northing', 'easting', 'altitude']]) - np.asarray(
                df_centroids.iloc[positives[pos_i]][['northing', 'easting', 'altitude']])
            T = registration_withinit(base_path + query, base_path + df_centroids.iloc[positives[pos_i]]["file"],
                                      trans_pre=trans_pre)
            positives_T.append(T)
        # if len(list_neg) > 4001:
        #     random.shuffle(list_neg)
        #     list_neg = list_neg[:4001]
        negatives = list_neg
        random.shuffle(negatives)
        queries[i] = {"query": query, "positives": positives,
                      "positives_T": positives_T, "negatives": negatives}

    with open(filename, 'wb') as handle:
        pickle.dump(queries, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Done ", filename)


# Initialize pandas DataFrame
df_train = pd.DataFrame(columns=['file', 'northing', 'easting', 'altitude'])

for folder in all_folders:
    df_locations = pd.read_csv(os.path.join(
        base_path, runs_folder, folder, filename), sep=',')
    df_locations['timestamp'] = runs_folder + folder + \
                                pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
    df_locations = df_locations.rename(columns={'timestamp': 'file'})

    for index, row in df_locations.iterrows():
        df_train = df_train.append(row, ignore_index=True)

if not os.path.exists("pickles/"):
    os.mkdir("pickles/")

print("Number of training submaps: " + str(len(df_train['file'])))
construct_query_dict(df_train, "pickles/" + "gta5_training_queries_baseline.pickle")
