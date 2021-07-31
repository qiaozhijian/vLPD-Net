import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "..")))
from tqdm import tqdm
from glob import glob
from registration import registration_withinit

recall_dis = 1.25


def main():
    # Building database and query files for evaluation
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    base_path = '../benchmark_datasets/'

    # For Oxford
    runs_folder = "GTA5/"
    all_folders = sorted(glob(os.path.join(BASE_DIR, base_path, runs_folder) + '/round*'))
    for i in range(len(all_folders)):
        all_folders[i] = os.path.basename(all_folders[i])
    print("start process")
    construct_query_and_database_sets(base_path, runs_folder, all_folders, "/pcl/",
                                      "pointcloud_locations_20m.csv")


def check_in_test_set():
    return True


##########################################
def output_to_file(output, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)


def construct_query_and_database_sets(base_path, runs_folder, folders, pointcloud_fols, filename):
    database_trees = []
    test_trees = []
    for folder in tqdm(folders):
        # print(folder)
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting', 'altitude'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting', 'altitude'])

        df_locations = pd.read_csv(os.path.join(
            base_path, runs_folder, folder, filename), sep=',')

        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if (check_in_test_set()):
                df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting', 'altitude']])
        test_tree = KDTree(df_test[['northing', 'easting', 'altitude']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in tqdm(folders):
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(
            base_path, runs_folder, folder, filename), sep=',')
        df_locations['timestamp'] = runs_folder + folder + \
                                    pointcloud_fols + df_locations['timestamp'].astype(str) + '.bin'
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if (check_in_test_set()):
                test[len(test.keys())] = {
                    'query': row['file'], 'northing': row['northing'], 'easting': row['easting'],
                    'altitude': row['altitude']}
            database[len(database.keys())] = {
                'query': row['file'], 'northing': row['northing'], 'easting': row['easting'],
                'altitude': row['altitude']}
        database_sets.append(database)
        test_sets.append(test)

    for i in tqdm(range(len(database_sets))):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if (i == j):
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array(
                    [[test_sets[j][key]["northing"], test_sets[j][key]["easting"], test_sets[j][key]["altitude"]]])
                index = tree.query_radius(coor, r=recall_dis)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

                positives_T = []
                for pos_i in test_sets[j][key][i]:
                    query = test_sets[j][key]["query"]
                    test_i = test_sets[i][pos_i]
                    query_i = test_i["query"]
                    coor_i = np.array([test_i["northing"], test_i["easting"], test_i["altitude"]])
                    trans_pre = coor.squeeze() - coor_i.squeeze()
                    T = registration_withinit(base_path + query, base_path + query_i, trans_pre=trans_pre)
                    positives_T.append(T)
                test_sets[j][key]['positives_T'] = positives_T

    if not os.path.exists("pickles/"):
        os.mkdir("pickles/")

    output_to_file(database_sets, "pickles/" + 'gta5_evaluation_database.pickle')
    output_to_file(test_sets, "pickles/" + 'gta5_evaluation_query.pickle')


if __name__ == '__main__':
    main()
