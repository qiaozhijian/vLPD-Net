# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git

import argparse
import json
import os
import pickle

import numpy as np
import torch
from sklearn.neighbors import KDTree
from tqdm import tqdm

from misc.log import log_string
from misc.utils import MinkLocParams
from models.model_factory import model_factory, load_weights
from training.reg_train import testVCRNet
from torch.utils.data import DataLoader
from dataloader.oxford import Oxford
DEBUG = False


def evaluate(model, device, params, log=False):
    # Run evaluation on all eval datasets

    if DEBUG:
        params.eval_database_files = params.eval_database_files[0:1]
        params.eval_query_files = params.eval_query_files[0:1]

    assert len(params.eval_database_files) == len(params.eval_query_files)

    stats = {}
    for database_file, query_file in zip(params.eval_database_files, params.eval_query_files):
        # Extract location name from query and database files
        location_name = database_file.split('_')[0]
        temp = query_file.split('_')[0]
        assert location_name == temp, 'Database location: {} does not match query location: {}'.format(database_file,
                                                                                                       query_file)

        p = os.path.join(params.queries_folder, database_file)
        with open(p, 'rb') as f:
            database_sets = pickle.load(f)

        p = os.path.join(params.queries_folder, query_file)
        with open(p, 'rb') as f:
            query_sets = pickle.load(f)
        if log:
            print('Evaluation:{} on {}'.format(database_file, query_file))
        temp = evaluate_dataset(model, device, params, database_sets, query_sets, log=log)
        stats[location_name] = temp

    for database_name in stats:
        log_string('Dataset: {} '.format(database_name), end='')
        t = 'Avg. top 1 recall: {:.2f} Avg. top 1% recall: {:.2f} Avg. similarity: {:.4f}'
        log_string(t.format(stats[database_name]['ave_recall'][0],
                            stats[database_name]['ave_one_percent_recall'],
                            stats[database_name]['average_similarity']))
    return stats


def evaluate_dataset(model, device, params, database_sets, query_sets, log=False):
    # Run evaluation on a single dataset
    recall = np.zeros(25)
    count = 0
    similarity = []
    one_percent_recall = []

    database_embeddings = []
    query_embeddings = []

    model.eval()
    if log:
        tqdm_ = lambda x, desc: tqdm(x, desc=desc)
    else:
        tqdm_ = lambda x, desc: x

    torch.cuda.empty_cache()
    for set in tqdm_(database_sets, 'Database'):
        database_embeddings.append(get_latent_vectors(model, set, device, params))

    for set in tqdm_(query_sets, '   Query'):
        query_embeddings.append(get_latent_vectors(model, set, device, params))

    for i in tqdm_(range(len(query_sets)), '    Test'):
        for j in range(len(query_sets)):
            if i == j:
                continue
            pair_recall, pair_similarity, pair_opr = get_recall(i, j, database_embeddings, query_embeddings, query_sets,
                                                                database_sets, log=log)
            recall += np.array(pair_recall)
            count += 1
            one_percent_recall.append(pair_opr)
            for x in pair_similarity:
                similarity.append(x)

    ave_recall = recall / count
    average_similarity = np.mean(similarity)
    ave_one_percent_recall = np.mean(one_percent_recall)
    stats = {'ave_one_percent_recall': ave_one_percent_recall, 'ave_recall': ave_recall,
             'average_similarity': average_similarity, 'Loc rebuild': 1}
    return stats


def load_pc(file_name, params, make_tensor=True):
    # returns Nx3 matrix
    file_path = os.path.join(params.dataset_folder, file_name)
    pc = np.fromfile(file_path, dtype=np.float64)
    # coords are within -1..1 range in each dimension
    assert pc.shape[0] == params.num_points * 3, "Error in point cloud shape: {}".format(file_path)
    pc = np.reshape(pc, (pc.shape[0] // 3, 3))
    if make_tensor:
        pc = torch.tensor(pc, dtype=torch.float)
    return pc


def load_pc_files(elem_ndxs, set, params):
    pcs = []
    for elem_ndx in elem_ndxs:
        pc = load_pc(set[elem_ndx]["query"], params, make_tensor=False)
        if (pc.shape[0] != 4096):
            assert 0, 'pc.shape[0] != 4096'
        pcs.append(pc)
    pcs = np.asarray(pcs)

    pcs = torch.tensor(pcs, dtype=torch.float)
    return pcs


def genrate_batch(num, batch_size):
    sets = np.arange(0, num, batch_size)
    sets = sets.tolist()
    if sets[-1] != num:
        sets.append(num)
    return sets


def get_latent_vectors(model, set, device, params: MinkLocParams):
    if DEBUG:
        embeddings = np.random.rand(len(set), 256)
        return embeddings

    model.eval()
    embeddings_l = []

    batch_set = genrate_batch(len(set), int(params.batch_size * 1.5))
    for batch_id in range(len(batch_set) - 1):
        elem_ndx = np.arange(batch_set[batch_id], batch_set[batch_id + 1])
        x = load_pc_files(elem_ndx, set, params)
        with torch.no_grad():
            batch = {'cloud': x.cuda()}
            embedding = model(target_batch=batch)['embeddings']
            # embedding is (1, 1024) tensor
            if params.normalize_embeddings:
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)  # Normalize embeddings

        embedding = embedding.detach().cpu().numpy()
        embeddings_l.append(embedding)

    embeddings = np.vstack(embeddings_l)
    return embeddings


def get_recall(m, n, database_vectors, query_vectors, query_sets, database_sets, log=False):
    # Original PointNetVLAD code
    database_output = database_vectors[m]
    queries_output = query_vectors[n]

    database_nbrs = KDTree(database_output)

    num_neighbors = 25
    recall = [0] * num_neighbors

    top1_similarity_score = []
    one_percent_retrieved = 0
    threshold = max(int(round(len(database_output) / 100.0)), 1)

    num_evaluated = 0
    for i in range(len(queries_output)):
        # i is query element ndx
        query_details = query_sets[n][i]  # {'query': path, 'northing': , 'easting': }
        true_neighbors = query_details[m]
        if len(true_neighbors) == 0:
            continue
        num_evaluated += 1
        distances, indices = database_nbrs.query(np.array([queries_output[i]]), k=num_neighbors)

        for j in range(len(indices[0])):
            if indices[0][j] in true_neighbors:
                if j == 0:
                    similarity = np.dot(queries_output[i], database_output[indices[0][j]])
                    top1_similarity_score.append(similarity)
                recall[j] += 1
                break

        if len(list(set(indices[0][0:threshold]).intersection(set(true_neighbors)))) > 0:
            one_percent_retrieved += 1

    one_percent_recall = (one_percent_retrieved / float(num_evaluated)) * 100
    recall = (np.cumsum(recall) / float(num_evaluated)) * 100
    # log_string(recall)
    # log_string(np.mean(top1_similarity_score))
    # log_string(one_percent_recall)
    return recall, top1_similarity_score, one_percent_recall


def export_eval_stats(file_name, prefix, eval_stats):
    s = prefix
    ave_1p_recall_l = []
    ave_recall_l = []
    # Print results on the final model
    with open(file_name, "a") as f:
        for ds in ['oxford', 'university', 'residential', 'business']:
            ave_1p_recall = eval_stats[ds]['ave_one_percent_recall']
            ave_1p_recall_l.append(ave_1p_recall)
            ave_recall = eval_stats[ds]['ave_recall'][0]
            ave_recall_l.append(ave_recall)
            s += ", {:0.2f}, {:0.2f}".format(ave_1p_recall, ave_recall)

        mean_1p_recall = np.mean(ave_1p_recall_l)
        mean_recall = np.mean(ave_recall_l)
        s += ", {:0.2f}, {:0.2f}\n".format(mean_1p_recall, mean_recall)
        f.write(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate model on PointNetVLAD (Oxford) dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model-specific configuration file')
    parser.add_argument('--weights', type=str, required=False, help='Trained model weights')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.set_defaults(debug=False)
    parser.add_argument('--visualize', dest='visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--savejson', type=str, default='', help='')
    parser.add_argument('--eval_reg', type=str, default="")

    args = parser.parse_args()
    log_string('Config path: {}'.format(args.config))
    log_string('Model config path: {}'.format(args.model_config))
    if args.weights is None:
        w = 'RANDOM WEIGHTS'
    else:
        w = args.weights
    log_string('Weights: {}'.format(w))
    log_string('Debug mode: {}'.format(args.debug))
    log_string('Visualize: {}'.format(args.visualize))

    params = MinkLocParams(args.config, args.model_config)
    params.print()

    model, device, d_model, vcr_model = model_factory(params)

    load_weights(args.weights, model)

    if args.eval_reg != "":
        test_loader = DataLoader(
            Oxford(params=params, partition='test'),
            batch_size=int(params.reg.batch_size * 1.2), shuffle=False, drop_last=False, num_workers=16)
        checkpoint_dict = torch.load(args.eval_reg, map_location=torch.device('cpu'))
        vcr_model.load_state_dict(checkpoint_dict, strict=True)
        log_string('load vcr_model with {}'.format(args.eval_reg))
        testVCRNet(1, vcr_model, test_loader)
    else:
        stats = evaluate(model, device, params, True)

        for database_name in stats:
            log_string('   Avg. recall @N:')
            log_string(str(stats[database_name]['ave_recall']))

        if len(args.savejson) > 0:
            result = {}
            result['trainfile'] = params.train_file
            result['weightfile'] = args.weights
            result['lr'] = params.lr
            result['lamda_g'] = params.lamda_gd
            result['weight_decay'] = params.weight_decay
            result['domain_adapt'] = params.domain_adapt
            if params.domain_adapt:
                result['lr_d'] = params.d_lr
                result['lamda_d'] = params.lamda_d
                result['weight_decay_d'] = params.d_weight_decay
            else:
                result['lr_d'] = None
                result['lamda_d'] = None
                result['weight_decay_d'] = None

            for database_name in stats:
                result_database = {}
                result_database['recall_top1'] = float(stats[database_name]['ave_recall'][0])
                result_database['recall_top1per'] = float(stats[database_name]['ave_one_percent_recall'])
                result_database['similarity'] = float(stats[database_name]['average_similarity'])
                result[database_name] = result_database
            json.dump(result, open(args.savejson, 'w'))
