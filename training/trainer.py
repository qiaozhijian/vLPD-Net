# Author: Zhijian Qiao
# Shanghai Jiao Tong University
# Code adapted from PointNetVlad code: https://github.com/jac99/MinkLoc3D.git
# Train on Oxford dataset (from PointNetVLAD paper) using BatchHard hard negative mining.
import os

import numpy as np
import open3d as o3d
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from evaluate import evaluate
from loss.d_loss import make_d_loss
from loss.metric_loss import make_loss
from misc.log import log_dir, log_string, reg_log_dir
from misc.utils import MinkLocParams
from models.model_factory import model_factory, load_weights
from training.optimizer_factory import optimizer_factory, scheduler_factory
from training.reg_train import trainVCRNet, testVCRNet


class IOStream:
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        """print to cmd and save to a txt file at the same time

        """
        print(text)
        self.f.write(text + '\n')
        self.f.flush()

    def close(self):
        self.f.close()


def tensors_to_numbers(stats):
    stats = {e: stats[e].item() if torch.is_tensor(stats[e]) else stats[e] for e in stats}
    return stats


class Trainer:
    def __init__(self, dataloaders, params: MinkLocParams, checkpoint="", debug=False, visualize=False):
        log_string('Model name: {}'.format(params.model_params.model))

        self.params = params

        self.eval_simple = params.eval_simple

        self.lamda = params.lamda
        self.lamda_reg = params.lamda_reg

        self.domain_adapt = params.domain_adapt
        if params.domain_adapt:
            self.lamda_gd = params.lamda_gd
            self.lamda_d = params.lamda_d
            self.repeat_g = params.repeat_g

        self.visualize = visualize

        self.model, self.device, self.d_model, self.vcr_model = model_factory(self.params)

        self.loss_fn = make_loss(self.params)

        self.loss_d = make_d_loss(self.params) if self.domain_adapt else None

        self.optimizer, self.optimizer_d = optimizer_factory(self.params, self.model, self.d_model)

        self.scheduler, self.scheduler_d = scheduler_factory(self.params, self.optimizer, self.optimizer_d)

        self.resume = False
        if checkpoint == "":
            from glob import glob
            checkpoint_set = sorted(glob(os.path.join(log_dir, "weights/*")))
            if len(checkpoint_set) == 0:
                checkpoint = ""
            else:
                for i in range(len(checkpoint_set)):
                    dir, filename = os.path.split(checkpoint_set[i])
                    num = filename[filename.find('-') + 1:filename.find('.')]
                    checkpoint_set[i] = os.path.join(dir, filename.replace(num, num.zfill(3)))
                checkpoint = sorted(checkpoint_set)[-1]

                dir, filename = os.path.split(checkpoint)
                num = filename[filename.find('-') + 1:filename.find('.')]
                checkpoint = os.path.join(dir, filename.replace(num, str(int(num))))

                self.resume = True
        self.starting_epoch = load_weights(checkpoint, self.model, self.optimizer, self.scheduler)

        self.writer = SummaryWriter(log_dir)

        self.dataloaders_train = dataloaders['train']

        self.total_train = len(self.dataloaders_train) if not debug else 2
        if 'val' in dataloaders:
            self.dataloaders_val = self.dataloaders['val']
            self.total_val = len(self.dataloaders_train)
        else:
            self.dataloaders_val = None
        if params.is_register:
            self.params.reg.log_dir = reg_log_dir
            self.boardio = SummaryWriter(reg_log_dir)
            self.textio = IOStream(os.path.join(reg_log_dir, "run.log"))
            self.dataloaders_train_reg = dataloaders['reg_train']
            self.dataloaders_test_reg = dataloaders['reg_test']

    def do_train(self):
        if self.params.is_register:
            if not self.resume:
                log_string("***********start trainVCRNet*************")
                trainVCRNet(self.params.reg, self.vcr_model, self.dataloaders_train_reg, self.dataloaders_test_reg,
                            self.boardio, self.textio)
                saved_state_dict = self.vcr_model.state_dict()
                model_state_dict = self.model.state_dict()  # 获取已创建net的state_dict
                saved_state_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}
                model_state_dict.update(saved_state_dict)
                self.model.load_state_dict(model_state_dict, strict=True)
                self.optimizer, self.optimizer_d = optimizer_factory(self.params, self.model, self.d_model)
                self.scheduler, self.scheduler_d = scheduler_factory(self.params, self.optimizer, self.optimizer_d)
            self.params.lpd_fixed = True
            self.params.is_register = True
            self.params.lamda_reg = 0

        self.count_samples = 0
        self.count_batch = 0
        log_string("***********start train vLPD-Net*************")
        for epoch in range(self.starting_epoch, self.params.epochs + 1):

            if self.domain_adapt:
                epoch_stats = self.train_one_epoch_da(epoch)
            else:
                epoch_stats = self.train_one_epoch(epoch)

            self.update_statics(epoch, epoch_stats, 'train')

            if self.scheduler is not None:
                self.scheduler.step()

        # if not self.eval_simple:
        #     return
        eval_stats = self.save_every_epoch(log_dir, epoch)
        # 输出和记录eval_stats
        for database_name in eval_stats:
            self.writer.add_scalar('evaluation_{}/Avg. top 1 recall every epoch'.format(database_name),
                                   eval_stats[database_name]['ave_recall'][0], epoch)
            self.writer.add_scalar('evaluation_{}/Avg. top 1% recall every epoch'.format(database_name),
                                   eval_stats[database_name]['ave_one_percent_recall'], epoch)
            self.writer.add_scalar('evaluation_{}/Avg. similarity recall every epoch'.format(database_name),
                                   eval_stats[database_name]['average_similarity'], epoch)

    def prepare_data(self):
        batch, self.positives_mask, self.negatives_mask, da_batch, R_gt, t_gt, source_batch, gt_T = self.dataset_iter.next()
        self.source_xyz = source_batch['cloud'].to(self.device) if self.params.is_register else source_batch['cloud']
        self.target_xyz = batch['cloud'].to(self.device)
        # self.visual_pcl_simple(self.target_xyz[0], da_batch['cloud'].to(self.device)[0], "1")
        self.R_gt = R_gt.to(self.device)
        self.t_gt = t_gt.to(self.device)
        self.gt_T = gt_T.to(self.device)
        self.target_batch = {e: batch[e].to(self.device) if e != 'coords' and e != None else batch[e] for e in batch}
        self.da_batch = {e: da_batch[e].to(self.device) if e != 'coords' and e != None else da_batch[e] for e in
                         da_batch} if da_batch != None else None
        self.source_batch = {e: source_batch[e].to(self.device) if self.params.is_register else source_batch[
            e] if e != 'coords' and e != None else source_batch[e] for e in source_batch}
        n_positives = torch.sum(self.positives_mask).item()
        n_negatives = torch.sum(self.negatives_mask).item()
        if n_positives == 0 or n_negatives == 0:
            return False
        else:
            return True

    def train_one_epoch(self, epoch):
        self.model.train()
        self.model.emb_nn.eval()

        all_stats_in_epoch = []  # running stats for the current epoch

        self.rotations_ab = []
        self.translations_ab = []
        self.rotations_ab_pred = []
        self.translations_ab_pred = []

        self.dataset_iter = self.dataloaders_train.__iter__()
        self.total_train = len(self.dataset_iter)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        for i in tqdm(range(self.total_train)):
            try:
                if not self.prepare_data():
                    continue
            except ValueError:
                log_string('dataloader error.')
                continue
            # Move everything to the device except 'coords' which must stay on CPU

            self.temp_stats = {}
            self.optimizer.zero_grad()

            metric_loss, out_states = self.metric_train(self.source_batch, self.target_batch, self.positives_mask,
                                                        self.negatives_mask, self.gt_T)

            loss = self.lamda * metric_loss
            loss.backward()
            self.optimizer.step()

            temp_stats = tensors_to_numbers(self.temp_stats)
            all_stats_in_epoch.append(temp_stats)

            self.count_samples = self.count_samples + self.negatives_mask.shape[0]
            self.count_batch += self.negatives_mask.shape[0]
            for key, value in temp_stats.items():
                self.writer.add_scalar('{}/batch_train'.format(key), value, self.count_samples)

        # Compute mean stats for the epoch
        epoch_stats = {'epoch': epoch}

        for key in all_stats_in_epoch[0].keys():
            temp = [e[key] for e in all_stats_in_epoch]
            epoch_stats[key] = np.mean(temp)

        return epoch_stats

    def visual_pcl_simple(self, pcl1, pcl2, name='Open3D Origin'):
        pcl1 = pcl1.detach().cpu().numpy()
        pcl2 = pcl2.detach().cpu().numpy()
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pcl1[:, :3])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pcl2[:, :3])
        pcd1.paint_uniform_color([1, 0.706, 0])
        pcd2.paint_uniform_color([0, 0.651, 0.929])
        o3d.visualization.draw_geometries([pcd1, pcd2], window_name=name, width=1920, height=1080,
                                          left=50,
                                          top=50,
                                          point_show_normal=False, mesh_show_wireframe=False,
                                          mesh_show_back_face=False)

    def metric_train(self, source_batch, target_batch, positives_mask, negatives_mask, gt_T):

        # Compute embeddings of all elements
        out_states = self.model(source_batch, target_batch, gt_T)

        if self.lamda <= 0:
            return 0.0, out_states

        metric_loss, temp_stats, self.hard_triplets = self.loss_fn(out_states['embeddings'], positives_mask,
                                                                   negatives_mask)
        self.temp_stats.update(temp_stats)
        self.temp_stats.update({'metric_loss_lamda': self.lamda * metric_loss.detach().cpu().item()})

        return metric_loss, out_states

    def backward_G(self, batch, positives_mask, negatives_mask):

        self.temp_stats = {}
        self.optimizer.zero_grad()

        out_states = self.model(None, batch, None)
        metric_loss, temp_stats, self.hard_triplets = self.loss_fn(out_states['embeddings'], positives_mask,
                                                                   negatives_mask)
        adv_gen_loss = self.adv_gen_train(out_states['embeddings'])

        loss = self.lamda * metric_loss + self.lamda_gd * adv_gen_loss

        loss.backward()
        self.optimizer.step()

    def adv_gen_train(self, embeddings):

        if not self.domain_adapt:
            return 0.0

        pred_syn = self.d_model(embeddings)
        adv_gen_loss, temp_stats = self.loss_d(pred_syn, self.hard_triplets)
        self.temp_stats.update(temp_stats)

        return adv_gen_loss

    def backward_D(self, batch, da_batch, positives_mask, negatives_mask):
        if not self.domain_adapt:
            return

        self.optimizer_d.zero_grad()

        # Compute embeddings of all elements
        out_states = self.model(None, batch, None)
        out_states_da = self.model(None, da_batch, None)

        pred_syn = self.d_model(out_states['embeddings'])
        pred_real = self.d_model(out_states_da['embeddings'])

        # embeddings [B,256]
        d_loss, d_stats = self.loss_d(pred_syn, self.hard_triplets, pred_real)

        self.temp_stats.update(d_stats)

        d_loss = self.lamda_d * d_loss
        d_loss.backward()

        self.optimizer_d.step()

    def update_statics(self, epoch, epoch_stats, phase):

        log_string('{} epoch {}. '.format(phase, epoch_stats['epoch']), end='')
        for key in epoch_stats:
            if key != 'epoch':
                log_string("{}: {}. ".format(key, epoch_stats[key]), end='')
                self.writer.add_scalar('{}/epoch_{}/'.format(key, phase), epoch_stats[key], epoch)
        log_string('')

        if self.params.batch_expansion_th is not None:
            # Dynamic batch expansion
            epoch_train_stats = epoch_stats
            if 'num_non_zero_triplets' not in epoch_train_stats:
                pass
                # log_string('WARNING: Batch size expansion is enabled, but the loss function is not supported')
            else:
                # Ratio of non-zero triplets
                rnz = epoch_train_stats['num_non_zero_triplets'] / epoch_train_stats['num_triplets']

                if rnz < self.params.batch_expansion_th:
                    self.dataloaders_train.batch_sampler.expand_batch()

        if phase != 'train':
            return

        eval_stats = self.save_every_epoch(log_dir, epoch)

        # 输出和记录eval_stats
        for database_name in eval_stats:
            self.writer.add_scalar('evaluation_{}/Avg. top 1 recall every epoch'.format(database_name),
                                   eval_stats[database_name]['ave_recall'][0], epoch)
            self.writer.add_scalar('evaluation_{}/Avg. top 1% recall every epoch'.format(database_name),
                                   eval_stats[database_name]['ave_one_percent_recall'], epoch)
            self.writer.add_scalar('evaluation_{}/Avg. similarity recall every epoch'.format(database_name),
                                   eval_stats[database_name]['average_similarity'], epoch)

    def save_batch_model(self, batch, epoch):
        return
        weights_path = os.path.join(log_dir, 'weights')
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        model_path = self.params.model_params.model + '-{}_{}.pth'.format(epoch, batch)
        model_path = os.path.join(weights_path, model_path)

        if isinstance(self.model, torch.nn.DataParallel):
            model_to_save = self.model.module
        else:
            model_to_save = self.model
        tqdm.write('Model saved.Epoch:{},Batch:{}'.format(epoch, batch))
        torch.save({
            'epoch': epoch,
            'state_dict': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, model_path)

    def save_every_epoch(self, log_dir, epoch):
        # Save final model weights
        weights_path = os.path.join(log_dir, 'weights')
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)
        model_path = self.params.model_params.model + '-{}.pth'.format(epoch)
        model_path = os.path.join(weights_path, model_path)

        if isinstance(self.model, torch.nn.DataParallel):
            model_to_save = self.model.module
        else:
            model_to_save = self.model

        torch.save({
            'epoch': epoch,
            'state_dict': model_to_save.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, model_path)
        # Evaluate the final model

        # if self.eval_simple:
        #     return {}
        self.model.eval()
        eval_stats = evaluate(self.model, self.device, self.params)

        return eval_stats

    def train_one_epoch_da(self, epoch):
        self.model.train()
        all_stats_in_epoch = []  # running stats for the current epoch

        self.dataset_iter = self.dataloaders_train.__iter__()
        self.total_train = len(self.dataset_iter)

        torch.cuda.empty_cache()  # Prevent excessive GPU memory consumption by SparseTensors
        for i in tqdm(range(self.total_train)):
            try:
                if not self.prepare_data():
                    continue
            except ValueError:
                log_string('dataloader error.')
                continue

            # Move everything to the device except 'coords' which must stay on CPU
            self.backward_G(self.target_batch, self.positives_mask, self.negatives_mask)

            self.backward_D(self.target_batch, self.da_batch, self.positives_mask, self.negatives_mask)

            if self.count_batch > 2000:
                self.save_batch_model(self.count_batch, epoch)
                # evaluate(self.model, self.device, self.params)
                self.count_batch -= 2000
            temp_stats = tensors_to_numbers(self.temp_stats)
            all_stats_in_epoch.append(temp_stats)

            self.count_samples = self.count_samples + self.negatives_mask.shape[0]
            self.count_batch += self.negatives_mask.shape[0]
            for key, value in temp_stats.items():
                self.writer.add_scalar('{}/batch_train'.format(key), value, self.count_samples)

        # Compute mean stats for the epoch
        epoch_stats = {'epoch': epoch}

        for key in all_stats_in_epoch[0].keys():
            temp = [e[key] for e in all_stats_in_epoch]
            epoch_stats[key] = np.mean(temp)

        return epoch_stats
