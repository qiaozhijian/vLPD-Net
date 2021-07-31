import gc

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from misc.point_utils import transform_point_cloud, npmat2euler


def vcrnetIter(net, src, tgt, iter=1):
    transformed_src = src
    bFirst = True

    for i in range(iter):
        srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(
            transformed_src.transpose(-1, -2), tgt.transpose(-1, -2))
        transformed_src = transform_point_cloud(transformed_src, rotation_ab_pred, translation_ab_pred)

        if bFirst:
            bFirst = False
            rotation_ab_pred_final = rotation_ab_pred.detach()
            translation_ab_pred_final = translation_ab_pred.detach()
        else:
            rotation_ab_pred_final = torch.matmul(rotation_ab_pred.detach(), rotation_ab_pred_final)
            translation_ab_pred_final = torch.matmul(rotation_ab_pred.detach(),
                                                     translation_ab_pred_final.unsqueeze(2)).squeeze(
                2) + translation_ab_pred.detach()

    rotation_ba_pred_final = rotation_ab_pred_final.transpose(2, 1).contiguous()
    translation_ba_pred_final = -torch.matmul(rotation_ba_pred_final, translation_ab_pred_final.unsqueeze(2)).squeeze(2)

    return srcK, src_corrK, rotation_ab_pred_final, translation_ab_pred_final, rotation_ba_pred_final, translation_ba_pred_final

def test_one_epoch(iter, net, test_loader):
    net.eval()
    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss_VCRNet = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
                test_loader):

            src = src.cuda()
            target = target.cuda()
            rotation_ab = rotation_ab.cuda()
            translation_ab = translation_ab.cuda()
            rotation_ba = rotation_ba.cuda()
            translation_ba = translation_ba.cuda()

            batch_size = src.size(0)
            num_examples += batch_size

            if iter > 0:
                srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = vcrnetIter(
                    net, src, target, iter=iter)
            elif iter == 0:
                srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = vcrnetIcpNet(
                    net, src, target)
            else:
                raise RuntimeError('iter')

            ## save rotation and translation
            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.numpy())
            ##
            rotations_ba.append(rotation_ba.detach().cpu().numpy())
            translations_ba.append(translation_ba.detach().cpu().numpy())
            rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
            translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
            eulers_ba.append(euler_ba.numpy())

            # Predicted point cloud
            transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
            # Real point cloud
            transformed_srcK = transform_point_cloud(srcK, rotation_ab, translation_ab)

            ###########################
            identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)

            loss_VCRNet = torch.nn.functional.mse_loss(transformed_srcK, src_corrK)

            loss_pose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                        + F.mse_loss(translation_ab_pred, translation_ab)

            total_loss_VCRNet += loss_VCRNet.item() * batch_size

            total_loss += loss_pose.item() * batch_size

            mse_ab += torch.mean((transformed_srcK - src_corrK) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ab += torch.mean(torch.abs(transformed_srcK - src_corrK), dim=[0, 1, 2]).item() * batch_size

            mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
            mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba, total_loss_VCRNet * 1.0 / num_examples


def train_one_epoch(params, net, train_loader, opt):
    net.train()

    mse_ab = 0
    mae_ab = 0
    mse_ba = 0
    mae_ba = 0

    total_loss_VCRNet = 0

    total_loss = 0
    total_cycle_loss = 0
    num_examples = 0
    rotations_ab = []
    translations_ab = []
    rotations_ab_pred = []
    translations_ab_pred = []

    rotations_ba = []
    translations_ba = []
    rotations_ba_pred = []
    translations_ba_pred = []

    eulers_ab = []
    eulers_ba = []
    torch.cuda.empty_cache()

    for src, target, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba, label in tqdm(
            train_loader):
        src = src.cuda()
        target = target.cuda()
        rotation_ab = rotation_ab.cuda()
        translation_ab = translation_ab.cuda()
        rotation_ba = rotation_ba.cuda()
        translation_ba = translation_ba.cuda()

        batch_size = src.size(0)
        opt.zero_grad()
        num_examples += batch_size
        srcK, src_corrK, rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(
            src.transpose(-1, -2), target.transpose(-1, -2))

        ## save rotation and translation
        rotations_ab.append(rotation_ab.detach().cpu().numpy())
        translations_ab.append(translation_ab.detach().cpu().numpy())
        rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
        translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
        eulers_ab.append(euler_ab.numpy())
        ##
        rotations_ba.append(rotation_ba.detach().cpu().numpy())
        translations_ba.append(translation_ba.detach().cpu().numpy())
        rotations_ba_pred.append(rotation_ba_pred.detach().cpu().numpy())
        translations_ba_pred.append(translation_ba_pred.detach().cpu().numpy())
        eulers_ba.append(euler_ba.numpy())

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)

        transformed_srcK = transform_point_cloud(srcK, rotation_ab, translation_ab)
        ###########################
        identity = torch.eye(3).cuda().unsqueeze(0).repeat(batch_size, 1, 1)
        loss_VCRNet = torch.nn.functional.mse_loss(transformed_srcK, src_corrK)

        loss_VCRNet.backward()
        total_loss_VCRNet += loss_VCRNet.item() * batch_size

        loss_pose = F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)

        opt.step()
        total_loss += loss_pose.item() * batch_size

        mse_ab += torch.mean((transformed_srcK - src_corrK) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ab += torch.mean(torch.abs(transformed_srcK - src_corrK), dim=[0, 1, 2]).item() * batch_size

        mse_ba += torch.mean((transformed_target - src) ** 2, dim=[0, 1, 2]).item() * batch_size
        mae_ba += torch.mean(torch.abs(transformed_target - src), dim=[0, 1, 2]).item() * batch_size

    rotations_ab = np.concatenate(rotations_ab, axis=0)
    translations_ab = np.concatenate(translations_ab, axis=0)
    rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
    translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)

    rotations_ba = np.concatenate(rotations_ba, axis=0)
    translations_ba = np.concatenate(translations_ba, axis=0)
    rotations_ba_pred = np.concatenate(rotations_ba_pred, axis=0)
    translations_ba_pred = np.concatenate(translations_ba_pred, axis=0)

    eulers_ab = np.concatenate(eulers_ab, axis=0)
    eulers_ba = np.concatenate(eulers_ba, axis=0)

    return total_loss * 1.0 / num_examples, total_cycle_loss / num_examples, \
           mse_ab * 1.0 / num_examples, mae_ab * 1.0 / num_examples, \
           mse_ba * 1.0 / num_examples, mae_ba * 1.0 / num_examples, rotations_ab, \
           translations_ab, rotations_ab_pred, translations_ab_pred, rotations_ba, \
           translations_ba, rotations_ba_pred, translations_ba_pred, eulers_ab, eulers_ba, total_loss_VCRNet * 1.0 / num_examples


def trainVCRNet(params, net, train_loader, test_loader, boardio, textio):
    # opt = optim.Adam(net.parameters(), lr=0.01, weight_decay=1e-4)
    opt = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, verbose=False, threshold=0.0001)

    best_test_loss = np.inf
    best_test_cycle_loss = np.inf
    best_test_mse_ab = np.inf
    best_test_rmse_ab = np.inf
    best_test_mae_ab = np.inf

    best_test_r_mse_ab = np.inf
    best_test_r_rmse_ab = np.inf
    best_test_r_mae_ab = np.inf
    best_test_t_mse_ab = np.inf
    best_test_t_rmse_ab = np.inf
    best_test_t_mae_ab = np.inf

    for epoch in range(params.epochs):
        train_loss_Pose, train_cycle_loss, \
        train_mse_ab, train_mae_ab, train_mse_ba, train_mae_ba, train_rotations_ab, train_translations_ab, \
        train_rotations_ab_pred, \
        train_translations_ab_pred, train_rotations_ba, train_translations_ba, train_rotations_ba_pred, \
        train_translations_ba_pred, train_eulers_ab, train_eulers_ba, train_loss_VCRNet = train_one_epoch(params, net,
                                                                                                          train_loader,
                                                                                                          opt)

        test_loss_Pose, test_cycle_loss_Pose, \
        test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
        test_rotations_ab_pred, \
        test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
        test_translations_ba_pred, test_eulers_ab, test_eulers_ba, test_loss_VCRNet = test_one_epoch(params.iter, net,
                                                                                                     test_loader)

        train_rmse_ab = np.sqrt(train_mse_ab)
        test_rmse_ab = np.sqrt(test_mse_ab)

        train_rotations_ab_pred_euler = npmat2euler(train_rotations_ab_pred)
        train_r_mse_ab = np.mean((train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)) ** 2)
        train_r_rmse_ab = np.sqrt(train_r_mse_ab)
        train_r_mae_ab = np.mean(np.abs(train_rotations_ab_pred_euler - np.degrees(train_eulers_ab)))
        train_t_mse_ab = np.mean((train_translations_ab - train_translations_ab_pred) ** 2)
        train_t_rmse_ab = np.sqrt(train_t_mse_ab)
        train_t_mae_ab = np.mean(np.abs(train_translations_ab - train_translations_ab_pred))

        test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
        test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
        test_r_rmse_ab = np.sqrt(test_r_mse_ab)
        test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
        test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
        test_t_rmse_ab = np.sqrt(test_t_mse_ab)
        test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

        if best_test_loss >= test_loss_Pose:
            best_test_loss = test_loss_Pose
            best_test_cycle_loss = test_cycle_loss_Pose

            best_test_mse_ab = test_mse_ab
            best_test_rmse_ab = test_rmse_ab
            best_test_mae_ab = test_mae_ab

            best_test_r_mse_ab = test_r_mse_ab
            best_test_r_rmse_ab = test_r_rmse_ab
            best_test_r_mae_ab = test_r_mae_ab

            best_test_t_mse_ab = test_t_mse_ab
            best_test_t_rmse_ab = test_t_rmse_ab
            best_test_t_mae_ab = test_t_mae_ab

            import os
            from os.path import join, exists
            model_dir = join(params.log_dir, "model")
            if not exists(model_dir):
                os.makedirs(model_dir)

            if torch.cuda.device_count() > 1:
                torch.save(net.module.state_dict(), join(model_dir, "model.{}.t7".format(epoch)))
            else:
                torch.save(net.state_dict(), join(model_dir, "model.{}.t7".format(epoch)))

        # scheduler.step()
        scheduler.step(best_test_loss)
        lr = opt.param_groups[0]['lr']

        if lr <= 0.0000011:
            break

        textio.cprint('==TRAIN==')
        textio.cprint('A--------->B')
        textio.cprint(
            'EPOCH:: %d, Loss: %f, LossPose: %f, Cycle Loss:, %f, lr: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
            'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
            % (
                epoch, train_loss_VCRNet, train_loss_Pose, train_cycle_loss, lr, train_mse_ab, train_rmse_ab,
                train_mae_ab,
                train_r_mse_ab,
                train_r_rmse_ab, train_r_mae_ab, train_t_mse_ab, train_t_rmse_ab, train_t_mae_ab))

        textio.cprint('==TEST==')
        textio.cprint('A--------->B')
        textio.cprint(
            'EPOCH:: %d, Loss: %f, LossPose: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
            'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
            % (epoch, test_loss_VCRNet, test_loss_Pose, test_cycle_loss_Pose, test_mse_ab, test_rmse_ab, test_mae_ab,
               test_r_mse_ab,
               test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))

        textio.cprint('==BEST TEST==')
        textio.cprint('A--------->B')
        textio.cprint('EPOCH:: %d, Loss: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
                      'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
                      % (epoch, best_test_loss, best_test_cycle_loss, best_test_mse_ab, best_test_rmse_ab,
                         best_test_mae_ab, best_test_r_mse_ab, best_test_r_rmse_ab,
                         best_test_r_mae_ab, best_test_t_mse_ab, best_test_t_rmse_ab, best_test_t_mae_ab))

        # boardio.add_scalar('A->B/train/loss', train_loss_VCRNet, epoch)
        # boardio.add_scalar('A->B/train/lossPose', train_loss_Pose, epoch)

        ############TEST
        boardio.add_scalar('A->B/test/loss', test_loss_VCRNet, epoch)
        boardio.add_scalar('A->B/test/lossPose', test_loss_Pose, epoch)
        boardio.add_scalar('A->B/test/rotation/RMSE', test_r_rmse_ab, epoch)
        boardio.add_scalar('A->B/test/translation/MAE', test_t_mae_ab, epoch)

        ############BEST TEST
        boardio.add_scalar('A->B/best_test/lr', lr, epoch)
        boardio.add_scalar('A->B/best_test/loss', best_test_loss, epoch)
        boardio.add_scalar('A->B/best_test/rotation/MAE', best_test_r_mae_ab, epoch)
        boardio.add_scalar('A->B/best_test/translation/MAE', best_test_t_mae_ab, epoch)

        import os
        from os.path import join, exists
        model_dir = join(params.log_dir, "model")
        if not exists(model_dir):
            os.makedirs(model_dir)

        if torch.cuda.device_count() > 1:
            torch.save(net.module.state_dict(), join(model_dir, "model.{}.t7".format(epoch)))
        else:
            torch.save(net.state_dict(), join(model_dir, "model.{}.t7".format(epoch)))

        gc.collect()


def testVCRNet(iter, net, test_loader):
    test_loss_Pose, test_cycle_loss_Pose, \
    test_mse_ab, test_mae_ab, test_mse_ba, test_mae_ba, test_rotations_ab, test_translations_ab, \
    test_rotations_ab_pred, \
    test_translations_ab_pred, test_rotations_ba, test_translations_ba, test_rotations_ba_pred, \
    test_translations_ba_pred, test_eulers_ab, test_eulers_ba, test_loss_VCRNet = test_one_epoch(iter, net,
                                                                                                 test_loader)

    test_rmse_ab = np.sqrt(test_mse_ab)

    test_rotations_ab_pred_euler = npmat2euler(test_rotations_ab_pred)
    test_r_mse_ab = np.mean((test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)) ** 2)
    test_r_rmse_ab = np.sqrt(test_r_mse_ab)
    test_r_mae_ab = np.mean(np.abs(test_rotations_ab_pred_euler - np.degrees(test_eulers_ab)))
    test_t_mse_ab = np.mean((test_translations_ab - test_translations_ab_pred) ** 2)
    test_t_rmse_ab = np.sqrt(test_t_mse_ab)
    test_t_mae_ab = np.mean(np.abs(test_translations_ab - test_translations_ab_pred))

    print('==TEST==')
    print('A--------->B')
    print(
        'Loss: %f, LossPose: %f, Cycle Loss: %f, MSE: %f, RMSE: %f, MAE: %f, rot_MSE: %f, rot_RMSE: %f, '
        'rot_MAE: %f, trans_MSE: %f, trans_RMSE: %f, trans_MAE: %f'
        % (test_loss_VCRNet, test_loss_Pose, test_cycle_loss_Pose, test_mse_ab, test_rmse_ab, test_mae_ab,
           test_r_mse_ab,
           test_r_rmse_ab, test_r_mae_ab, test_t_mse_ab, test_t_rmse_ab, test_t_mae_ab))
