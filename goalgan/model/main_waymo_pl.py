import pytorch_lightning as pl
from argparse import Namespace
from utils.utils import get_batch_k
import numpy as np
from omegaconf import DictConfig
import torch.nn as nn
from utils.losses import l2_loss, GANLoss, cal_ade, cal_fde, crashIntoWall
from get_model import get_model
import torch
from data import TrajectoryDataset, seq_collate
from torch.utils.data import DataLoader
from utils.radam import RAdam
from utils.utils import re_im
from data.waymoDSGoalGan import WaymoGoalGanDS, d_collate_fn
from torch import Tensor
from utils.visualize import visualize_traj_probabilities
import os
import itertools
import psutil
from typing import List, Tuple, Optional
from torch.optim.optimizer import Optimizer


class TrajPredictor(pl.LightningModule):
    """
    PyTorch Lightning Module for training GOAL GAN
    Hyperparamters of of training are set in 'config/training/training.yaml'
    and explained in 'HYPERPARAMETERS.md'
    """

    def __init__(self, hparams: DictConfig = None, args: Namespace = None, loss_fns=None, ):
        super().__init__()

        self.args = args
        self.hparams.update(hparams)
        self.generator, self.discriminator = get_model(self.hparams)
        print(self.generator)
        print(self.discriminator)
        # init loss functions
        self.loss_fns = loss_fns if loss_fns else {'L2': l2_loss,  # L2 loss
                                                   'ADV': GANLoss(hparams.gan_mode),  # adversarial Loss
                                                   'G': l2_loss,  # goal achievement loss
                                                   'GCE': nn.CrossEntropyLoss()}  # Goal Cross Entropy loss
        # init loss weights

        self.loss_weights = {'L2': hparams.w_L2,
                             'ADV': hparams.w_ADV,  # adversarial Loss
                             'G': hparams.w_G,  # goal achievement loss
                             'GCE': hparams.w_GCE}  # Goal Cross Entropy loss

        self.current_batch_idx = -1
        self.plot_val = True
        if self.hparams.batch_size_scheduler:
            self.batch_size = self.hparams.batch_size_scheduler
        else:
            self.batch_size = self.hparams.batch_size

    def train_dataloader(self):

        ds_path = "/media/robot/hdd1/waymo_ds/"
        ind_path = "/media/robot/hdd1/waymo_ds/training_mapstyle/index_file.txt"
        train_dset = WaymoGoalGanDS(data_path=ds_path, index_file=ind_path,
                                    rgb_index_path="/media/robot/hdd1/waymo_ds/rendered/train/index.pkl",
                                    rgb_prefix="/media/robot/hdd1/waymo_ds/")

        train_loader = DataLoader(train_dset, batch_size=self.batch_size, shuffle=False,
                                  num_workers=self.hparams.num_workers, collate_fn=d_collate_fn)
        return train_loader

    def val_dataloader(self):
        # OPTIONAL
        return None

    """########## DATE PREPARATION ##########"""

    def training_step(self, batch, batch_idx, optimizer_idx):
        self.batch_idx = batch_idx
        self.generator.gen()
        self.logger.experiment.add_scalar('train/CPU Usage', psutil.cpu_percent(), self.global_step)

        if self.device.type != 'cpu':
            self.logger.experiment.add_scalar('train/GPU Usage',
                                              torch.cuda.get_device_properties(self.device).total_memory,
                                              self.global_step)
        self.plot_val = True
        if self.gsteps and optimizer_idx == 0:  # and self.current_batch_idx != batch_idx:

            self.output = self.generator_step(batch)
            return self.output
        elif self.dsteps and optimizer_idx == 1:  # and self.current_batch_idx != batch_idx:

            output = self.discriminator_step(batch)

            self.output = output
            return output
        else:
            return self.output

    def forward(self, batch):
        return self.generator(batch)

    def generator_step(self, batch, vis=0):
        """
        Generator optimization step.
        Args:
            batch: Batch from the data loader.

            Returns:
               discriminator loss on fake
               norm loss on trajectory
               kl loss
        """
        tqdm_dict = {}
        total_loss = 0.
        ade_sum, fde_sum = [], []
        ade_sum_pixel, fde_sum_pixel = [], []
        batch = get_batch_k(batch, self.hparams.best_k)
        batch_size = batch["in_xy"].shape[1] // self.hparams.best_k
        generator_out = self.generator(batch)

        if vis:
            self.vis_step(batch)


        l2 = self.loss_fns["L2"](
            batch["gt_dxdy"] * batch["gt_valid"].unsqueeze(-1),
            generator_out["out_dxdy"] * batch["gt_valid"].unsqueeze(-1),
            mode='raw',
            type="mse", masks=batch["gt_valid"])
        ade_error = cal_ade(
            batch["gt_dxdy"] * batch["gt_valid"].unsqueeze(-1),
            generator_out["out_dxdy"] * batch["gt_valid"].unsqueeze(-1),
            mode='raw'
        ) / 16

        fde_error = cal_fde(
            batch["gt_dxdy"] * batch["gt_valid"].unsqueeze(-1),
            generator_out["out_dxdy"] * batch["gt_valid"].unsqueeze(-1),
            mode='raw'
        )

        ade_error = ade_error.view(self.hparams.best_k, batch_size)

        # fde_error = fde_error.view(self.hparams.best_k, batch_size)
        tqdm_dict["train/ade_mean20"] = ade_error.mean().item()
        tqdm_dict["train/fde_mean20"] = torch.mean(fde_error).item()

        l2 = l2.view(self.hparams.best_k, -1)

        loss_l2, _ = l2.min(dim=0, keepdim=True)
        loss_l2 = torch.mean(loss_l2)
        tqdm_dict["train/loss_l2"] = loss_l2.item() / 16
        loss_l2 = self.loss_weights["L2"] * loss_l2
        total_loss += loss_l2
        if self.generator.global_vis_type == "goal":
            target_reshaped = batch["prob_mask"][:batch_size].view(batch_size, -1)
            output_reshaped = generator_out["y_scores"][:batch_size].view(batch_size, -1)

            _, targets = target_reshaped.max(dim=1)

            loss_gce = self.loss_weights["GCE"] * self.loss_fns["GCE"](
                output_reshaped[batch["gt_valid"][-1, :batch_size] > 0],
                targets[batch["gt_valid"][-1, :batch_size] > 0])
            tqdm_dict["train/loss_gce"] = loss_gce.item()
            total_loss += loss_gce
            # tqdm_dict["GCE_train"] = loss_gce

            final_end = torch.sum(generator_out["out_dxdy"], dim=0, keepdim=True)

            final_end = generator_out["out_dxdy"][-1:, :, :]
            final_end_gt = batch["gt_dxdy"][-1:, :, :]

            final_pos = generator_out["final_pos"]

            goal_error = self.loss_fns["G"](final_pos.detach(), final_end_gt)
            goal_error = goal_error.view(self.hparams.best_k, -1)
            _, id_min = goal_error.min(dim=0, keepdim=False)
            # id_min*=torch.range(0, len(id_min))*10

            final_pos = final_pos.view(self.hparams.best_k, batch_size, -1)
            final_end = final_end.view(self.hparams.best_k, batch_size, -1)

            final_pos = torch.cat([final_pos[id_min[k], k].unsqueeze(0) for k in range(final_pos.size(1))]).unsqueeze(0)
            final_end = torch.cat([final_end[id_min[k], k].unsqueeze(0) for k in range(final_end.size(1))]).unsqueeze(0)
            fp_masked = final_pos.detach()[batch["gt_valid"][-1:, :batch_size] > 0].unsqueeze(0)
            fe_masked = final_end[batch["gt_valid"][-1:, :batch_size] > 0].unsqueeze(0)
            loss_G = self.loss_weights["G"] * torch.mean(self.loss_fns["G"](fp_masked, fe_masked, mode='raw'))
            # tqdm_dict["train/loss_G"] = loss_G.item()
            total_loss += loss_G

            traj_fake = generator_out["out_xy"][:, :batch_size]
            traj_fake_rel = generator_out["out_dxdy"][:, :batch_size]

            if self.generator.rm_vis_type == "attention":
                image_patches = generator_out["image_patches"][:, :batch_size]
            else:
                image_patches = None

            fake_scores = self.discriminator(in_xy=batch["in_xy"][:, :batch_size],
                                             in_dxdy=batch["in_dxdy"][:, :batch_size],
                                             out_xy=traj_fake,
                                             out_dxdy=traj_fake_rel,
                                             images_patches=image_patches)

            loss_adv = self.loss_weights["ADV"] * self.loss_fns["ADV"](fake_scores, True).clamp(min=0)

            total_loss += loss_adv
            tqdm_dict["train/ADV_train"] = loss_adv
            tqdm_dict["train/G_loss"] = total_loss
            for key, loss in tqdm_dict.items():
                self.logger.experiment.add_scalar('{}'.format(key), loss, self.global_step)

            # tqdm_dict["G_train"] = loss_G
        return {"loss": total_loss}

    def discriminator_step(self, batch):

        """Discriminator optimization step.
        Args:
            batch: Batch from the data loader.

        Returns:
            discriminator loss on fake
            discriminator loss on real
        """
        # init loss and loss dict
        tqdm_dict = {}
        total_loss = 0.

        self.generator.gen()
        self.discriminator.grad(True)

        with torch.no_grad():

            out = self.generator(batch)

        traj_fake = out["out_xy"]
        traj_fake_rel = out["out_dxdy"]

        if self.generator.rm_vis_type == "attention":
            image_patches = out["image_patches"]
        else:
            image_patches = None

        dynamic_fake = self.discriminator(in_xy=batch["in_xy"],
                                          in_dxdy=batch["in_dxdy"],
                                          out_xy=traj_fake,
                                          out_dxdy=traj_fake_rel,
                                          images_patches=image_patches)

        if self.generator.rm_vis_type == "attention":
            image_patches = batch["local_patches"]  # .permute(1, 0, 2, 3, 4)
        else:
            image_patches = None

        dynamic_real = self.discriminator(in_xy=batch["in_xy"],
                                          in_dxdy=batch["in_dxdy"],
                                          out_xy=batch["gt_xy"],
                                          out_dxdy=batch["gt_dxdy"],
                                          images_patches=image_patches)

        disc_loss_real_dynamic = self.loss_fns["ADV"](dynamic_real, True).clamp(min=0)
        disc_loss_fake_dynamic = self.loss_fns["ADV"](dynamic_fake, False).clamp(min=0)

        disc_loss = disc_loss_real_dynamic + disc_loss_fake_dynamic

        tqdm_dict = {"D_train": disc_loss,
                     "D_real_train": disc_loss_real_dynamic,
                     "D_fake_train": disc_loss_fake_dynamic}

        for key, loss in tqdm_dict.items():
            self.logger.experiment.add_scalar('train/{}'.format(key), loss, self.global_step)
        return {
            'loss': disc_loss
        }

    def backward(self, loss: Tensor, optimizer: Optional[Optimizer], optimizer_idx: Optional[int], **kwargs):
        # condition that backward is not called when nth is passed
        if 1 and (
                # if self.current_batch_idx != self.batch_idx and (
                ((optimizer_idx == 0) and self.gsteps) or ((optimizer_idx == 1) and self.dsteps)):
            loss.backward()

    def optimizer_step(self,
                       epoch: int,
                       batch_idx: int,
                       optimizer,
                       optimizer_idx: int = 0,
                       optimizer_closure=None,
                       on_tpu: bool = False,
                       using_native_amp: bool = False,
                       using_lbfgs: bool = False
                       ):
        # Step using d_loss or g_loss
        # update generator opt every 2 steps
        if self.gsteps and optimizer_idx == 0:  # and self.current_batch_idx != batch_idx:

            self.current_batch_idx = batch_idx
            optimizer.step(closure=optimizer_closure)  # optimizer.step()
            optimizer.zero_grad()
            self.gsteps -= 1
            if not self.gsteps:
                if self.discriminator:
                    self.dsteps = self.hparams.d_steps
                else:
                    self.gsteps = self.hparams.g_steps

        # update discriminator opt every 4 steps
        if self.dsteps and optimizer_idx == 1:  # and self.current_batch_idx != batch_idx:
            optimizer.step(closure=optimizer_closure)
            optimizer.zero_grad()
            self.dsteps -= 1
            if not self.dsteps:
                self.gsteps = self.hparams.g_steps
            self.current_batch_idx = batch_idx

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        :return:
        """

        # opt_g = RAdam(self.generator.parameters(), lr=self.hparams.lr_gen)
        # opt_d = RAdam(self.discriminator.parameters(), lr=self.hparams.lr_dis)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr_gen)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr_dis)

        schedulers = []
        if self.hparams.lr_scheduler_G:
            lr_scheduler_G = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler_G)(opt_g,
                                                                                            **self.hparams.lr_scheduler_G_args)
            schedulers.append(lr_scheduler_G)
        else:
            schedulers.append(None)

        if self.hparams.lr_scheduler_D:
            lr_scheduler_D = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler_D)(opt_d,
                                                                                            **self.hparams.lr_scheduler_D_args)
            schedulers.append(lr_scheduler_D)
        else:
            schedulers.append(None)

        self.gsteps = self.hparams.g_steps
        self.dsteps = 0
        return [opt_g, opt_d], schedulers

    def vis_step(self, batch, best_k=10):

        ade_sum, fde_sum = [], []
        ade_sum_pixel, fde_sum_pixel = [], []

        # get pixel ratios
        ratios = []
        # for img in batch["scene_img"]:
        #     ratios.append(torch.tensor(img["ratio"]))
        # ratios = torch.stack(ratios).to(self.device)

        # batch = get_batch_k(batch, best_k)

        self.generator.test()
        with torch.no_grad():
            out = self.generator(batch)

        if self.plot_val:
            self.plot_val = False
            self.visualize_results(batch, out)
        self.generator.train()

    def visualize_results(self, batch, out):

        # background_image = batch["scene_img"][0]["scaled_image"].copy()
        background_image = batch["scene_img"][0].clone().permute(1,2,0)
        inp = batch["in_dxdy"] * 224/200 + torch.tensor([224/4, 224/2]).cuda()
        gt = (batch["gt_dxdy"] * 224/200 + torch.tensor([224/4, 224/2]).cuda()).detach()
        pred = out["out_dxdy"] * 224/200 + torch.tensor([224/4, 224/2]).cuda()
        pred = pred.view(pred.size(0), self.hparams.best_k_val, -1, pred.size(-1))

        y = out["y_map"]
        y_softmax = out["y_softmax"]

        image = visualize_traj_probabilities(
            input_trajectory=inp.cpu()[:, 0],
            gt_trajectory=gt,
            prediction_trajectories=pred.cpu()[:, :, 0],
            background_image=background_image.detach().cpu(),
            img_scaling=1,  # self.val_dset.img_scaling,
            scaling_global=1,  # self.val_dset.scaling_global,
            grid_size=20,
            y_softmax=y_softmax,
            y=y,
            global_patch=re_im(batch["global_patch"]).cpu().numpy(),  # re_im(batch["global_patch"][0]).cpu().numpy(),
            probability_mask=batch["prob_mask"][0][0].cpu().numpy(),
            grid_size_in_global=16  # self.val_dset.grid_size_in_global

        )

        self.logger.experiment.add_image(f'Trajectories', image, self.current_epoch)


# main

if __name__ == "__main__":
    import hydra
    from model.pretrain_pl import pretrain_func

    from pytorch_lightning import Trainer, seed_everything
    import os
    from pytorch_lightning.loggers import TensorBoardLogger

    from pytorch_lightning.callbacks import ModelCheckpoint
    import logging
    import torch


    @hydra.main(config_path="../config", config_name="config")
    def main(cfg: DictConfig):
        print(cfg)

        ckpt_path = "/media/robot/hdd1/predictions_gan/GoalGAN/model/logs/lightning_logs/version_64/checkpoints/epoch=9-step=874440.ckpt"
        model = TrajPredictor(cfg)
        # model = TrajPredictor.load_from_checkpoint(ckpt_path, hparams=cfg)
        trainer = Trainer(
            num_sanity_val_steps=0,
            progress_bar_refresh_rate=10,
            **cfg.trainer)
        trainer.fit(model)


    main()
