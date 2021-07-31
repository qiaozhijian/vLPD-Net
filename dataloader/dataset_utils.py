# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git

import MinkowskiEngine as ME
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader.gta5 import GTAV5
from dataloader.oxford import OxfordDataset, TrainTransform, TrainSetTransform, Oxford
from dataloader.samplers import BatchSampler
from misc.utils import MinkLocParams


def make_datasets(params: MinkLocParams, debug=False):
    # Create training and validation datasets
    datasets = {}
    train_transform = TrainTransform(params.aug_mode)
    train_set_transform = TrainSetTransform(params.aug_mode)
    if debug:
        max_elems = 1000
    else:
        max_elems = None

    datasets['train'] = OxfordDataset(params, params.train_file, train_transform, set_transform=train_set_transform,
                                      max_elems=max_elems)
    val_transform = None
    if params.val_file is not None:
        datasets['val'] = OxfordDataset(params, params.val_file, val_transform)
    return datasets


def make_eval_dataset(params: MinkLocParams):
    # Create evaluation datasets
    dataset = OxfordDataset(params, params.test_file, transform=None)
    return dataset


def sparse_quantize(xyz_batch, params=None):
    coords = [
        ME.utils.sparse_quantize(e, quantization_size=params.model_params.mink_quantization_size, return_index=True)
        for e in xyz_batch]
    coords_idx = [e[1] + i * e[1].shape[0] for i, e in enumerate(coords)]
    coords_idx = torch.cat(coords_idx)
    coords = [e[0] for e in coords]
    coords = ME.utils.batched_coordinates(coords)

    # Assign a dummy feature equal to 1 to each point
    # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
    feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
    batch = {'coords': coords, 'features': feats, 'index': coords_idx, 'cloud': xyz_batch}

    return batch


def make_collate_fn(dataset: OxfordDataset, params=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        batch_size = len(data_list)
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        xyz_batch = torch.stack(clouds, dim=0)  # Produces (batch_size, n_points, 3) tensor
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            xyz_batch, R_set, t_set = dataset.set_transform(xyz_batch)
        else:
            R_set = np.eye(3)
            t_set = np.zeros((3, 1))

        batch = sparse_quantize(xyz_batch, params)

        clouds_source = [e[5] for e in data_list]
        xyz_batch_source = torch.stack(clouds_source, dim=0)  # Produces (batch_size, n_points, 3) tensor
        source_batch = sparse_quantize(xyz_batch_source, params)

        if data_list[0][2] is not None:
            clouds_da = [e[2][0] for e in data_list]
            xyz_batch_da = torch.stack(clouds_da, dim=0)  # Produces (batch_size, n_points, 3) tensor
            da_batch = sparse_quantize(xyz_batch_da, params)
        else:
            da_batch = None

        # Compute positives and negatives mask
        # dataset.queries[label]['positives'] is bitarray
        positives_mask = [[dataset.queries[label]['positives'][e] for e in labels] for label in labels]
        negatives_mask = [[dataset.queries[label]['negatives'][e] for e in labels] for label in labels]

        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        R2 = [e[3].reshape(1, 3, 3) for e in data_list]
        R2 = torch.tensor(np.concatenate(R2, axis=0), dtype=torch.float32)
        t2 = [e[4].reshape(1, 3, 1) for e in data_list]
        t2 = torch.tensor(np.concatenate(t2, axis=0), dtype=torch.float32)

        R_set = torch.tensor(R_set.reshape(1, 3, 3), dtype=torch.float32).repeat(batch_size, 1, 1)
        t_set = torch.tensor(t_set.reshape(1, 3, 1), dtype=torch.float32).repeat(batch_size, 1, 1)

        R = torch.bmm(R_set, R2)
        t = t_set + torch.bmm(R_set, t2)

        gt_T = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
        gt_T[:, :3, :3] = R
        gt_T[:, :3, 3] = t.reshape(batch_size, 3)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        return batch, positives_mask, negatives_mask, da_batch, R.float(), t.float(), source_batch, gt_T.float()

    return collate_fn


def make_dataloaders(params: MinkLocParams, debug=False):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements
    :param train_params:
    :param model_params:
    :return:
    """
    datasets = make_datasets(params, debug=debug)

    dataloders = {}
    train_sampler = BatchSampler(datasets['train'], batch_size=params.batch_size,
                                 batch_size_limit=params.batch_size_limit,
                                 batch_expansion_rate=params.batch_expansion_rate)
    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(datasets['train'], params)
    dataloders['train'] = DataLoader(datasets['train'], batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=params.num_workers, pin_memory=True)

    if 'val' in datasets:
        val_sampler = BatchSampler(datasets['val'], batch_size=params.batch_size)
        # Collate function collates items into a batch and applies a 'set transform' on the entire batch
        # Currently validation dataset has empty set_transform function, but it may change in the future
        val_collate_fn = make_collate_fn(datasets['val'], params)
        dataloders['val'] = DataLoader(datasets['val'], batch_sampler=val_sampler, collate_fn=val_collate_fn,
                                       num_workers=params.num_workers, pin_memory=True)
    if params.is_register:
        if not "gta" in params.train_file.lower():
            train_loader = DataLoader(
                Oxford(params=params, partition='train'),
                batch_size=params.reg.batch_size, shuffle=True, drop_last=True, num_workers=16)
            test_loader = DataLoader(
                Oxford(params=params, partition='test'),
                batch_size=int(params.reg.batch_size * 1.2), shuffle=False, drop_last=False, num_workers=16)
        else:
            train_loader = DataLoader(
                GTAV5(params=params, partition='train'),
                batch_size=params.reg.batch_size, shuffle=True, drop_last=True, num_workers=16)
            test_loader = DataLoader(
                GTAV5(params=params, partition='test'),
                batch_size=int(params.reg.batch_size * 1.2), shuffle=False, drop_last=False, num_workers=16)
        dataloders['reg_train'] = train_loader
        dataloders['reg_test'] = test_loader

    return dataloders
