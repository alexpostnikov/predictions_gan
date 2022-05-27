import gc
import os
import math
import random
import numpy as np
from collections import defaultdict
from waymoDs import preprocess_batch_to_predict_with_img
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from data import data_loader
from utils import get_dset_path
from utils import relative_to_abs
from utils import gan_g_loss, gan_d_loss, l2_loss, displacement_error, final_displacement_error
from models_waymo import TrajectoryGenerator, TrajectoryDiscriminator
import wandb
from visualization import vis_cur_and_fut
from constants import *


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight)


def get_dtypes():
    return torch.cuda.LongTensor, torch.cuda.FloatTensor




def main():
    from waymoDs import WaymoSophieDS, d_collate_fn
    from torch.utils.data import DataLoader

    wandb.init(project="waymo-sophie", entity="aleksey-postnikov", name="waymo_sophie_train_base_angle")

    print("Initializing train dataset")
    ds_path = "/media/robot/hdd1/waymo_ds/"
    in_path = "/media/robot/hdd1/waymo_ds/training_mapstyle/index_file.txt"

    train_dset = WaymoSophieDS(data_path=ds_path, index_file=in_path,
                                  rgb_index_path="/media/robot/hdd1/waymo_ds/rendered/train/index.pkl",
                                  rgb_prefix="/media/robot/hdd1/waymo_ds/")
    train_loader = DataLoader(train_dset, batch_size=2, shuffle=False, num_workers=0, collate_fn=d_collate_fn)

    long_dtype, float_dtype = get_dtypes()



    print("Initializing val dataset")
    val_ds_path = "/media/robot/hdd1/waymo_ds/"
    val_in_path = "/media/robot/hdd1/waymo_ds/val_mapstyle/index_file.txt"

    val_dset = WaymoSophieDS(data_path=ds_path, index_file=val_in_path,
                             rgb_index_path="/media/robot/hdd1/waymo_ds/rendered/val/index.pkl",
                             rgb_prefix="/media/robot/hdd1/waymo_ds/")

    val_loader = DataLoader(val_dset, batch_size=2, shuffle=False, num_workers=0, collate_fn=d_collate_fn)

    iterations_per_epoch = len(train_dset) / (D_STEPS + G_STEPS)
    NUM_ITERATIONS = int(iterations_per_epoch * NUM_EPOCHS)
    print('There are {} iterations per epoch'.format(iterations_per_epoch))

    generator = TrajectoryGenerator()
    generator.apply(init_weights)
    generator.type(float_dtype).train()
    print('Here is the generator:')
    print(generator)

    discriminator = TrajectoryDiscriminator()
    discriminator.apply(init_weights)
    discriminator.type(float_dtype).train()
    print('Here is the discriminator:')
    print(discriminator)

    optimizer_g = optim.Adam(generator.parameters(), lr=G_LR)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=D_LR)

    t, epoch = 0, 0
    t0 = None
    min_ade = None
    while t < NUM_ITERATIONS:
        gc.collect()
        d_steps_left = D_STEPS
        g_steps_left = G_STEPS
        epoch += 1
        print('Starting epoch {}'.format(epoch))
        pbar = tqdm.tqdm(train_loader, total=iterations_per_epoch)
        for data in pbar:
            out = preprocess_batch_to_predict_with_img(data)
            past_current_local, future_to_predict_local, state_to_predict, future_to_predict, rot,\
             trans, imgs, state_to_predict_with_neighbours_local, state_to_predict_with_neighbours, future_valid = out
            inv_rot = torch.inverse(rot)
            batch = (state_to_predict_with_neighbours, future_to_predict, state_to_predict_with_neighbours_local,
                     future_to_predict_local, imgs, inv_rot, future_valid)
            str_to_log = 'Epoch {} Batch {}'.format(epoch, t)
            if d_steps_left > 0:
                losses_d = discriminator_step(batch, generator,
                                              discriminator, gan_d_loss,
                                              optimizer_d)
                str_to_log += ' D_loss: {}'.format(losses_d)
                d_steps_left -= 1
            if g_steps_left > 0:
                losses_g, predictions = generator_step(batch, generator,
                                          discriminator, gan_g_loss,
                                          optimizer_g)
                g_steps_left -= 1
                str_to_log += ' G_loss: {}'.format(losses_g)
                pbar.set_description(str_to_log)
                if (t + 1) % 100 == 0:
                    predictions = torch.stack(predictions).detach().cpu()
                    state_to_predict_p = state_to_predict_with_neighbours.permute(2, 0, 1, 3)
                    predictions_abs, predictions_rel = rotate_predictions_to_abs_cs(predictions, state_to_predict_p, inv_rot)
                    predictions_abs = predictions_abs.permute(0, 2, 1).reshape(-1, 20, PRED_LEN, 2).permute(0,2,1,3)
                    imgs = vis_cur_and_fut(data, predictions_abs)
                    wandb.log({'train/G_loss': losses_g, "examples": [wandb.Image(imgs)]})
                    # wandb.log({'train/G_loss': losses_g, 'train/D_loss': losses_d, "examples": [wandb.Image(imgs)]})
                else:
                    # wandb.log({'train/G_loss': losses_g, 'train/D_loss': losses_d})
                    wandb.log({'train/G_loss': losses_g})
            # if d_steps_left > 0 or g_steps_left > 0:
            if  g_steps_left > 0:
                continue



            if (t+1) % PRINT_EVERY == 0:
                print('t = {} / {}'.format(t + 1, NUM_ITERATIONS))
                for k, v in sorted(losses_d.items()):
                    print('  [D] {}: {:.3f}'.format(k, v))
                for k, v in sorted(losses_g.items()):
                    print('  [G] {}: {:.3f}'.format(k, v))

                print('Checking stats on val ...')
                metrics_val = check_accuracy(val_loader, generator, discriminator, gan_d_loss)

                print('Checking stats on train ...')
                metrics_train = check_accuracy(train_loader, generator, discriminator, gan_d_loss, limit=True)

                for k, v in sorted(metrics_val.items()):
                    print('  [val] {}: {:.3f}'.format(k, v))
                for k, v in sorted(metrics_train.items()):
                    print('  [train] {}: {:.3f}'.format(k, v))

                if min_ade is None or metrics_val['ade'] < min_ade:
                    min_ade = metrics_val['ade']
                    checkpoint = {'t': t, 'g': generator.state_dict(), 'd': discriminator.state_dict(),
                                  'g_optim': optimizer_g.state_dict(), 'd_optim': optimizer_d.state_dict()}
                    print("Saving checkpoint to model.pt")
                    torch.save(checkpoint, "model.pt")
                    print("Done.")

            t += 1
            d_steps_left = D_STEPS
            g_steps_left = G_STEPS
            if t >= NUM_ITERATIONS:
                break


def discriminator_step(batch, generator, discriminator, d_loss_fn, optimizer_d):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj_, pred_traj_gt_, obs_traj_rel_, pred_traj_gt_rel_, vgg_list, rot_mat_inv, future_valid) = batch
    obs_traj = obs_traj_.permute(2, 0, 1, 3)
    obs_traj_rel = obs_traj_rel_.permute(2, 0, 1, 3)
    pred_traj_gt_rel = pred_traj_gt_rel_.permute(1, 0, 2)
    pred_traj_gt = pred_traj_gt_.permute(1, 0, 2)

    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    generator_out = generator(obs_traj, obs_traj_rel, vgg_list)

    pred_traj_fake_abs, pred_traj_fake_rel = rotate_predictions_to_abs_cs(generator_out, obs_traj, rot_mat_inv)
    pred_traj_fake_abs[future_valid.permute(1, 0) == 0] *= 0
    pred_traj_fake_rel[future_valid.permute(1, 0) == 0] *= 0
    pred_traj_gt_rel[future_valid.permute(1, 0) == 0] *= 0
    pred_traj_gt[future_valid.permute(1, 0) == 0] *= 0
    #pred_traj_fake_abs = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

    traj_real = torch.cat([obs_traj[:, :, 0, :], pred_traj_gt], dim=0)
    traj_real_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_gt_rel], dim=0)
    traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake_abs], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel)
    scores_real = discriminator(traj_real, traj_real_rel)

    data_loss = d_loss_fn(scores_real, scores_fake)
    losses['D_data_loss'] = data_loss.item()
    loss += data_loss
    losses['D_total_loss'] = loss.item()

    optimizer_d.zero_grad()
    loss.backward()
    optimizer_d.step()
    return losses


def rotate_predictions_to_abs_cs(generator_out, obs_traj, rot_mat_inv):
    if generator_out.ndimension() == 3:
        BATCH_SIZE = obs_traj.size(1)
        assert generator_out.shape == (PRED_LEN, BATCH_SIZE, 2)
        assert rot_mat_inv.shape == (BATCH_SIZE, 2, 2)
        assert obs_traj.shape == (OBS_LEN, BATCH_SIZE, MAX_PEDS, 2)
        pred_traj_fake_rel = generator_out
        # rotate back
        pred_traj_fake_rel_rot = (rot_mat_inv.float().bmm(pred_traj_fake_rel.permute(1, 2, 0))).permute(2, 0, 1)
        # translate back
        pred_traj_fake_abs = pred_traj_fake_rel_rot + obs_traj[-1:, :, 0]
        return pred_traj_fake_abs, pred_traj_fake_rel
    if generator_out.ndimension() == 4:
        BATCH_SIZE = obs_traj.size(1)
        assert generator_out.shape == (20, PRED_LEN, BATCH_SIZE, 2)
        assert rot_mat_inv.shape == (BATCH_SIZE, 2, 2)
        assert obs_traj.shape == (OBS_LEN, BATCH_SIZE, MAX_PEDS, 2)
        pred_traj_fake_rel = generator_out.permute(2, 3, 0,  1, ).reshape(-1, 2, PRED_LEN * 20)
        # rotate back
        pred_traj_fake_rel_rot = torch.bmm(rot_mat_inv.float(), pred_traj_fake_rel)
        # translate back
        pred_traj_fake_abs = pred_traj_fake_rel_rot + obs_traj[-1, :, 0, :].unsqueeze(-1)
        return pred_traj_fake_abs, generator_out


def generator_step(batch, generator, discriminator, g_loss_fn, optimizer_g):
    batch = [tensor.cuda() for tensor in batch]
    (obs_traj_, pred_traj_gt_, obs_traj_rel_, pred_traj_gt_rel_, vgg_list, rot_mat_inv, future_valid) = batch
    obs_traj = obs_traj_.permute(2, 0, 1, 3)
    obs_traj_rel = obs_traj_rel_.permute(2, 0, 1, 3)
    pred_traj_gt_rel = pred_traj_gt_rel_.permute(1, 0, 2)
    pred_traj_gt = pred_traj_gt_.permute(1, 0, 2)

    pred_traj_gt_rel[future_valid.permute(1, 0) == 0] *= 0
    pred_traj_gt[future_valid.permute(1, 0) == 0] *= 0
    losses = {}
    loss = torch.zeros(1).to(pred_traj_gt)

    g_l2_loss_rel = []
    predictions = []
    for _ in range(BEST_K):
        generator_out = generator(obs_traj, obs_traj_rel, vgg_list)
        predictions.append(generator_out.clone())
        pred_traj_fake_abs, pred_traj_fake_rel = rotate_predictions_to_abs_cs(generator_out, obs_traj, rot_mat_inv)
        pred_traj_fake_abs[future_valid.permute(1, 0)[torch.arange(80)[4::5]] == 0] *= 0
        pred_traj_fake_rel[future_valid.permute(1, 0)[torch.arange(80)[4::5]] == 0] *= 0

        dist = torch.norm(pred_traj_fake_rel - pred_traj_gt_rel[torch.arange(80)[4::5]], dim=2)
        g_l2_loss_rel.append(dist)
        # g_l2_loss_rel.append(l2_loss(
        #     pred_traj_fake_abs,
        #     pred_traj_gt,
        #     mode='raw'))

    npeds = obs_traj.size(1)  # bs
    g_l2_loss_sum_rel = torch.zeros(1).to(pred_traj_gt)
    g_l2_loss_rel = torch.stack(g_l2_loss_rel, dim=1)  # times, modes, bs
    diver_loss = 0.1 * 1 / (g_l2_loss_rel.std(1) / 100 + 0.1)
    _g_l2_loss_rel = torch.sum(g_l2_loss_rel, dim=0)  # modes, bs

    losses['G_diversity_loss'] = diver_loss.mean().item()
    loss += diver_loss.mean()
    _g_l2_loss_rel = torch.min(_g_l2_loss_rel, 0).values.mean() / (PRED_LEN)
    g_l2_loss_sum_rel += _g_l2_loss_rel
    losses['G_l2_loss_rel'] = g_l2_loss_sum_rel.item()

    loss += g_l2_loss_sum_rel

    traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake_abs], dim=0)
    traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

    scores_fake = discriminator(traj_fake, traj_fake_rel)
    discriminator_loss = g_loss_fn(scores_fake)

    loss += discriminator_loss
    losses['G_discriminator_loss'] = discriminator_loss.item()
    losses['G_total_loss'] = loss.item()

    optimizer_g.zero_grad()
    loss.backward()
    optimizer_g.step()

    return losses, predictions


def check_accuracy(loader, generator, discriminator, d_loss_fn, limit=False):
    d_losses = []
    metrics = {}
    g_l2_losses_abs, g_l2_losses_rel = ([],) * 2
    disp_error = []
    f_disp_error = []
    total_traj = 0

    mask_sum = 0
    generator.eval()
    with torch.no_grad():
        for data in loader:
            out = preprocess_batch_to_predict_with_img(data)
            past_current_local, future_to_predict_local, state_to_predict, future_to_predict, rot_mat,\
            trans, imgs, state_to_predict_with_neighbours_local, state_to_predict_with_neighbours, future_valid = out

            state_to_predict_with_neighbours = state_to_predict_with_neighbours.permute(2, 0, 1, 3)
            state_to_predict_with_neighbours_local = state_to_predict_with_neighbours_local.permute(2, 0, 1, 3)
            future_to_predict_local = future_to_predict_local.permute(1, 0, 2)
            future_to_predict = future_to_predict.permute(1, 0, 2)

            batch = (state_to_predict_with_neighbours, future_to_predict, state_to_predict_with_neighbours_local,
                     future_to_predict_local, imgs, torch.inverse(rot_mat), future_valid)
            batch = [tensor.cuda() for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, vgg_list, rot_mat_inv, future_valid) = batch

            pred_traj_fake_rel = generator(obs_traj, obs_traj_rel, vgg_list)
            pred_traj_fake = relative_to_abs(pred_traj_fake_rel, obs_traj[-1, :, 0, :])

            g_l2_loss_abs = l2_loss(pred_traj_fake, pred_traj_gt, mode='sum')
            g_l2_loss_rel = l2_loss(pred_traj_fake_rel, pred_traj_gt_rel, mode='sum')

            ade = displacement_error(pred_traj_fake, pred_traj_gt)
            fde = final_displacement_error(pred_traj_fake[-1], pred_traj_gt[-1])

            traj_real = torch.cat([obs_traj[:, :, 0, :], pred_traj_gt], dim=0)
            traj_real_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_gt_rel], dim=0)
            traj_fake = torch.cat([obs_traj[:, :, 0, :], pred_traj_fake], dim=0)
            traj_fake_rel = torch.cat([obs_traj_rel[:, :, 0, :], pred_traj_fake_rel], dim=0)

            scores_fake = discriminator(traj_fake, traj_fake_rel)
            scores_real = discriminator(traj_real, traj_real_rel)

            d_loss = d_loss_fn(scores_real, scores_fake)
            d_losses.append(d_loss.item())

            g_l2_losses_abs.append(g_l2_loss_abs.item())
            g_l2_losses_rel.append(g_l2_loss_rel.item())
            disp_error.append(ade.item())
            f_disp_error.append(fde.item())

            mask_sum += (pred_traj_gt.size(1) * PRED_LEN)
            total_traj += pred_traj_gt.size(1)
            if limit and total_traj >= NUM_SAMPLES_CHECK:
                break

    metrics['d_loss'] = sum(d_losses) / len(d_losses)
    metrics['g_l2_loss_abs'] = sum(g_l2_losses_abs) / mask_sum
    metrics['g_l2_loss_rel'] = sum(g_l2_losses_rel) / mask_sum

    metrics['ade'] = sum(disp_error) / (total_traj * PRED_LEN)
    metrics['fde'] = sum(f_disp_error) / total_traj
    generator.train()
    return metrics


if __name__ == '__main__':
    main()

