import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.factory.model_factory import xFormer, xFormerConfig
import math
from prettytable import PrettyTable
import pytorch_lightning as pl
from waymoDs import WaymoDS, d_collate_fn, RoadGraph, get_pose_from_batch_to_predict, simpl_collate_fn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import wandb
import time

LOSS_COEFS = {
    "NLL": 1.0,
    "BCE": 0.1,
    "BEST_ADE": 0.01,
    "INNIER_FDE": 0.01,
    "AVERAGE_ADE": 0
}


def pytorch_neg_multi_log_likelihood_batch(gt, predictions, confidences, avails):
    """
    Compute a negative log-likelihood for the multi-modal scenario.
    Args:
        gt (Tensor): array of shape (bs)x(time)x(2D coords)
        predictions (Tensor): array of shape (bs)x(modes)x(time)x(2D coords)
        confidences (Tensor): array of shape (bs)x(modes) with a confidence for each mode in each sample
        avails (Tensor): array of shape (bs)x(time) with the availability for each gt timestep
    Returns:
        Tensor: negative log-likelihood for this example, a single float number
    """

    bs, modes, seq_len, ddim = predictions.shape
    assert (bs, seq_len, ddim) == gt.shape
    assert (bs, modes) == confidences.shape
    assert (bs, seq_len) == avails.shape
    assert torch.allclose(confidences.sum(dim=1), torch.ones_like(confidences.sum(dim=1)))

    gt = torch.unsqueeze(gt, 1)  # add modes
    avails = avails[:, None, :, None]  # add modes and cords
    error = torch.sum(
        ((gt - predictions) * avails) ** 2, dim=-1
    )  # reduce coords and use availability
    with np.errstate(
            divide="ignore"
    ):  # when confidence is 0 log goes to -inf, but we're fine with it
        # error (batch_size, num_modes)
        error = nn.functional.log_softmax(confidences, dim=1) - 0.5 * torch.sum(
            error, dim=-1
        )  # reduce time
    # error (batch_size, num_modes)
    error = -torch.logsumexp(error, dim=-1, keepdim=True)
    return torch.mean(error)


def save_fig_to_numpy(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    buf.shape = (w, h, 3)
    return buf


def count_parameters(model_):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model_.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


EMB = 16
SEQ = 64
BATCH = 16
att = "linformer"
my_co = {
    "reversible": False,  # Optionally make these layers reversible, to save memory
    "block_type": "encoder",
    "num_layers": 4,  # Optional, this means that this config will repeat N times
    "dim_model": EMB,
    "layer_norm_style": "pre",  # Optional, pre/post
    "position_encoding_config": {
        "name": "sine",  # whatever position encodinhg makes sense
        "seq_len": SEQ

    },
    "multi_head_config": {
        "num_heads": 8,
        "residual_dropout": 0.,
        "attention": {
            "name": att,  # whatever attention mechanism
            "dropout": 0,
            "causal": False,
            "seq_len": SEQ,
        },
    },
    "feedforward_config": {
        "name": "MLP",
        "dropout": 0,
        "activation": "relu",
        "hidden_layer_multiplier": 2,
    },
    # Optional Simplicial Embeddings on the last encoder layer
    # the temperature parameter is itself optional
    "simplicial_embeddings": {"L": 4, "temperature": 0.5}
}

EMB = 6*200
SEQ = 50
BATCH = 16
att = "linformer"
my_co1 = {
    "reversible": False,  # Optionally make these layers reversible, to save memory
    "block_type": "encoder",
    "num_layers": 4,  # Optional, this means that this config will repeat N times
    "dim_model": EMB,
    "layer_norm_style": "pre",  # Optional, pre/post
    "position_encoding_config": {
        "name": "sine",  # whatever position encodinhg makes sense
        "seq_len": SEQ

    },
    "multi_head_config": {
        "num_heads": 16,
        "residual_dropout": 0.,
        "attention": {
            "name": att,  # whatever attention mechanism
            "dropout": 0,
            "causal": False,
            "seq_len": SEQ,
        },
    },
    "feedforward_config": {
        "name": "MLP",
        "dropout": 0,
        "activation": "relu",
        "hidden_layer_multiplier": 2,
    },
    # Optional Simplicial Embeddings on the last encoder layer
    # the temperature parameter is itself optional
    "simplicial_embeddings": {"L": 4, "temperature": 0.5}
}



class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MapTransf_big(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_model = xFormer.from_config(xFormerConfig([my_co1]))

        self.lines_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=1200, nhead=8, batch_first=True, dim_feedforward=512),
            num_layers=2
        )

        self.decoder_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=1200,
                nhead=4,
                dim_feedforward=1024,
                dropout=0,
                activation="relu",
                batch_first=True
            ),
            num_layers=4,
        )
        self.mp = nn.MaxPool1d(kernel_size=256)
        self.anchor = nn.Parameter(torch.randn(1, 1, 1200))
        self.out_linear = nn.Linear(1200, 1024)


    def forward(self, x, histories=None):

        bs, num_lines, num_points, _ = x.shape
        # start.record()
        lines_embs = self.lines_transformer(x.view(bs, num_lines, -1))
        x = self.decoder_model(self.anchor.repeat(x.shape[0], 1, 1), lines_embs)
        x = self.out_linear(x)
        return x.view(x.shape[0], -1)



class MapTransf(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_mlp = nn.Sequential(
            nn.Linear(6, 24),
            nn.ReLU(),
            nn.Linear(24, 16)
        )
        self.pre_encoder = nn.Sequential(
            nn.Linear(6, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        # self.encoder_model = xFormer.from_config(xFormerConfig([my_co]))
        self.encoder_model2 = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=16,
                nhead=4,
                dim_feedforward=32,
                dropout=0,
                activation="relu",
                batch_first=True
            ),
            num_layers=4,
        )

        self.lines_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=16, nhead=8, batch_first=True, dim_feedforward = 64),
            num_layers=2
        )

        self.decoder_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=16,
                nhead=4,
                dim_feedforward=128,
                dropout=0,
                activation="relu",
                batch_first=True
            ),
            num_layers=4,
        )
        self.mp = nn.MaxPool1d(kernel_size=256)
        self.anchor = nn.Parameter(torch.randn(1, 64, 16))

    def choose_closest(self, points: torch.Tensor, centers: torch.Tensor, num_lines_max=100):
        """
        Given a set of points and a set of centers, find the closest center for each point.
        :param points: (batch_size, num_lines, num_points, dim)
        :param centers: (batch_size, dim)
        :return: (batch_size, num_points)
        """
        batch_size, num_lines, num_points, dim = points.shape

        points_ = points.view(batch_size, num_lines, num_points, dim)
        centers_ = centers.view(batch_size, 1, 1, 2)
        distances = torch.norm(points_[:, :, :, :2] - centers_, dim=3).min(dim=2)[0]
        # sort points in second dimension (num_lines)
        distances, indeces = torch.sort(distances, dim=1)
        out_p = torch.zeros((batch_size, num_lines_max, num_points, dim), dtype=torch.float32, device=points.device)
        for i in range(batch_size):
            out_p[i, :, :, :] = points_[i, indeces[i][:num_lines_max], :, :]
        return out_p
        # choose the closest num_lines_max points for each point






    def forward(self, x, histories):


        finan_mean_p = torch.stack([h[:, -1, :2].mean(0) for h in histories])
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        x_near = self.choose_closest(x, finan_mean_p)
        # end.record()
        # torch.cuda.synchronize()
        # print(f"----Time to choose closest: {start.elapsed_time(end)}")
        bs, num_lines, num_points, _ = x_near.shape
        # start.record()
        lines_emb = self.pre_encoder(x_near)
        # end.record()
        # torch.cuda.synchronize()
        # print(f"----Time to pre_encoder: {start.elapsed_time(end)}")
        # start.record()
        src = lines_emb.view(bs * num_lines, num_points, -1)
        tgt = torch.randn(bs * num_lines, 1, 16, device=lines_emb.device)
        lines_embs = self.encoder_model2(tgt, src).view(bs, num_lines,  -1)
        # end.record()
        # torch.cuda.synchronize()
        # print(f"----Time to encoder_model: {start.elapsed_time(end)}")

        # lines_embs = lines_embs.max(-2).values
        # start.record()
        x = self.lines_transformer(lines_embs)
        x = self.decoder_model(self.anchor.repeat(x.shape[0], 1, 1), x)
        # end.record()
        # torch.cuda.synchronize()
        # print(f"----Time to decoder_model and lines_tr: {start.elapsed_time(end)}")
        return x.view(x.shape[0], -1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.h_dim = 128
        self.embedding_dim = 128

        self.lstm = nn.LSTM(self.embedding_dim, self.h_dim, 1, batch_first=True)
        self.spatial_embedding = nn.Sequential(
            nn.Linear(7, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim)
        )

    def init_hidden(self, batch):
        h = torch.zeros(1, batch, self.h_dim).cuda()
        c = torch.zeros(1, batch, self.h_dim).cuda()
        return h, c

    def forward(self, obs_traj):
        bs, seq_len, _ = obs_traj.shape
        relative = obs_traj[:, :, :2] - obs_traj[:, :, :2][:, -1:]

        poses_state = torch.cat([obs_traj, relative], dim=2)
        obs_traj_embedding = self.spatial_embedding(poses_state)
        obs_traj_embedding = obs_traj_embedding.view(bs, seq_len, self.embedding_dim)
        state = self.init_hidden(bs)
        output, state = self.lstm(obs_traj_embedding, state)
        final_h = state[0].view(bs, self.embedding_dim)
        # if padded:
        #     final_h = final_h.view(npeds, MAX_PEDS, self.h_dim)
        # else:
        #     final_h = final_h.view(npeds, self.h_dim)
        return final_h

    def sanity_check(self):
        obs_traj = torch.randn(3, 4, 8, 2).cuda()
        print(obs_traj.shape)
        out = self.forward(obs_traj)
        print(out.shape)
        print(out)
        return


def BCELoss_class_weighted(weights):
    def loss(inp, target):
        inp = torch.clamp(inp, min=1e-7, max=1 - 1e-7)
        bce = - weights[1] * target * torch.log(inp) - (1 - target) * weights[0] * torch.log(1 - inp)
        return torch.mean(bce)

    return loss


def local_mapgoals_to_coordinates(map_goals: torch.Tensor) -> torch.Tensor:
    assert map_goals.ndim == 3
    bs, num_samples, pose = map_goals.shape
    # x range (-100, 100)m, y range (-100, 100)m, center (50/2, 50/2)pixels, 1 pix = 4 m
    pose_goals = (map_goals - torch.tensor([25, 25], device=map_goals.device)) * 4
    return pose_goals


class TrajPred(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.map_enc = MapTransf_big()

        # self.region_enc = nn.Sequential(
        #     nn.Linear(1024 + 128, 25 * 50),
        #     nn.ReLU(),
        #     nn.Linear(25 * 50, 50 * 50),
        #     nn.Softmax(dim=-1)
        # )
        self.region_enc = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)
        )


        self.state_to_decoder = nn.Sequential(
            nn.Linear(1024 + 128, 128),
            nn.ReLU(),
            nn.Linear(128, 256))

        self.regions_to_decoder = nn.Sequential(
            nn.Linear(50 * 50, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
        )
        self.pe = PositionalEncoding(256)
        self.routing_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0,
                activation="relu",
                batch_first=True
            ),
            num_layers=2,
        )
        self.final = nn.Linear(256, 2 * 16 + 1)

        self.automatic_optimization = False

    def init_hidden(self, batch, h_dim):
        h = torch.zeros(1, batch, h_dim).cuda()
        c = torch.zeros(1, batch, h_dim).cuda()
        return h, c

    def forward(self, obs_traj, points, num_peds_per_batch, past_cur_valid, num_samples=10):
        bs, seq_len, _ = obs_traj.shape
        map_bs, num_objects, num_points, _ = points.shape
        obs_traj[past_cur_valid != 1] *= 0
        disp = []
        # calc displacements for each map
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        for i in range(points.shape[0]):
            mean_x = points[i, :, :, 0][points[i, :, :, 0] != -1].mean()
            points[i, :, :, 0] -= mean_x
            points[i, :, :, 0] /= 50.
            mean_y = points[i, :, :, 1][points[i, :, :, 1] != -1].mean()
            points[i, :, :, 1] -= mean_y
            points[i, :, :, 1] /= 50.
            mean_z = points[i, :, :, 2][points[i, :, :, 2] != -1].mean()
            points[i, :, :, 2] -= mean_z
            points[i, :, :, 2] /= 50.
            disp.append(torch.tensor([mean_x, mean_y, mean_z], device=points.device))
        disp = torch.stack(disp)
        cur_ped = 0
        poses_per_batch = []

        for bn, i in enumerate(num_peds_per_batch):
            obs_traj[cur_ped:cur_ped + i, :, :2] -= disp[bn, :2]
            poses_per_batch.append(obs_traj[cur_ped:cur_ped + i])
            cur_ped += i
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to calculate displacements:", start.elapsed_time(end))
        #
        #
        # start.record()

        obs_traj_enc = self.encoder(obs_traj).unsqueeze(1)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to encode obs_traj:", start.elapsed_time(end))



        # points[:, :, :, :3] -= disp.unsqueeze(1).unsqueeze(1)
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        map = self.map_enc(points, poses_per_batch)
        map_r = []
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to encode map:", start.elapsed_time(end))

        # start.record()

        for i in range(num_peds_per_batch.shape[0]):
            num_peds_at_sample = num_peds_per_batch[i]
            map_r.append(map[i].unsqueeze(0).expand(num_peds_at_sample, -1))
        map = torch.cat(map_r, dim=0).unsqueeze(1)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to repeat map:", start.elapsed_time(end))

        state = torch.cat([obs_traj_enc, map], dim=2)
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()
        regions = self.region_enc(state.view(bs, 128, 3, 3))
        regions = nn.Softmax(dim=-1)(regions.view(bs, 1, -1)).view(bs, 1, 50, 50)

        map_samples = []
        for sample in range(num_samples):
            sample = torch.nn.functional.gumbel_softmax(torch.log(regions.view(bs, -1)), tau=1, hard=False)
            map_samples.append(sample.view(bs, 50, 50))

        # stack the samples
        map_samples = torch.stack(map_samples, dim=1)  # shape is bs, num_samples, num_ped, 50, 50
        y_ = torch.argmax(torch.max(map_samples, -2).values, -1)
        x_ = torch.argmax(torch.max(map_samples, -1).values, -1)
        goals_pix = torch.stack([x_, y_], dim=-1)  # bs, num_samples, num_ped, 2
        goals_meters = local_mapgoals_to_coordinates(goals_pix)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to sample goals:", start.elapsed_time(end))

        # given state, shape is bs,  num_ped, 1024+32 and goal proposals (bs, num_samples, num_ped, 2)
        # generate trajectories for each sample (bs, num_samples, num_ped, seq_len, 2)
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        ms_ = self.pe(self.regions_to_decoder(map_samples.view(bs, num_samples, -1)))
        st_ = self.pe(self.state_to_decoder(state).view(bs, 1, -1))
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to encode goals:", start.elapsed_time(end))

        # start.record()

        trajes = self.routing_model(ms_, st_)
        out = self.final(trajes)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to generate trajectories:", start.elapsed_time(end))

        trajes = out[:, :, :-1].view(bs, num_samples, 16, 2)
        confs = nn.Softmax(1)(out[:, :, -1]).view(bs, num_samples)
        # outs = []
        # for sample in range(num_samples):
        #     output = self.traj_to_lstm(obs_traj_enc)
        #     state_tuple = (trajes[:, sample:sample + 1, :].permute(1, 0, 2).contiguous(),
        #                    trajes[:, sample:sample + 1, :].permute(1, 0, 2).contiguous())
        #     sample_traj = []
        #     for t in range(16):
        #         output, state_tuple = self.f_lstm_(output, state_tuple)
        #         sample_traj.append(output[:, 0])
        #     sample_traj = torch.stack(sample_traj, dim=1)
        #     outs.append(sample_traj)
        # out = torch.stack(outs, dim=1)
        # out = self.final(out).view(bs, num_samples, 16, 2)
        return (trajes, confs), regions, map_samples, goals_pix, goals_meters

    def sanity_check(self):
        obs_traj = torch.randn(3, 4, 8, 2).cuda()
        points = torch.randn(3, 20 * 256, 6).cuda()
        out, regions, poses, goals_pix, goals_meters = self.forward(obs_traj, points)
        # err from proposal:

        print(regions.shape)
        return

    def train_dataloader(self):
        ds_path = "/media/robot/hdd1/waymo_ds/"
        in_path = "/media/robot/hdd1/waymo_ds/training_mapstyle/index_file.txt"
        waymo_dataset = WaymoDS(data_path=ds_path, index_file=in_path)
        import torch.utils.data as data_utils
        # waymo_dataset = data_utils.Subset(waymo_dataset, indices)
        loader = DataLoader(waymo_dataset, batch_size=32, shuffle=False, num_workers=0, collate_fn=simpl_collate_fn)
        return loader

    def val_dataloader(self):
        ds_path = "/media/robot/hdd1/waymo_ds/"
        in_path = "/media/robot/hdd1/waymo_ds/val_mapstyle/index_file.txt"
        waymo_dataset = WaymoDS(data_path=ds_path, index_file=in_path)
        indices = list(range(10000))
        waymo_dataset = torch.utils.data.Subset(waymo_dataset, indices)
        loader = DataLoader(waymo_dataset, batch_size=32, shuffle=False, num_workers=7, collate_fn=simpl_collate_fn)
        return loader

    def test_dataloader(self):
        pass

    def training_step(self, batch, batch_idx):
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        roadGraph = RoadGraph(batch["roadgraph_samples/xyz"], ids=batch["roadgraph_samples/id"],
                              types=batch["roadgraph_samples/type"],
                              valid=batch["roadgraph_samples/valid"])
        num_peds_per_batch = (batch["state/tracks_to_predict"] > 0).sum(-1)
        _, _, history, future_to_predict, _, _, \
        _, _, future_valid = \
            get_pose_from_batch_to_predict(batch)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to get pose:", start.elapsed_time(end))
        # start.record()
        masks = batch["state/tracks_to_predict"].reshape(-1, 128) > 0
        past_valid = batch["state/past/valid"].reshape(-1, 128, 10)[masks]
        cur_valid = batch["state/current/valid"].reshape(-1, 128, 1)[masks]
        past_cur_valid = torch.cat([past_valid, cur_valid], dim=-1)
        gt_goalmap = self.generate_gt_goalmap(history, future_to_predict, future_valid)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to generate gt goalmap:", start.elapsed_time(end))
        # start.record()
        selector = np.arange(4, 80, 5)
        future_to_predict = future_to_predict[:, selector].cuda()
        future_valid = future_valid[:, selector].cuda()
        # points = roadGraph.all_objects(by_type=True)
        points = roadGraph.rg[:,::2].view(-1, 50, 200, 6)

        # end.record()
        # torch.cuda.synchronize()
        # print("Time to roadGraph.all_objects:", start.elapsed_time(end))
        # start.record()

        # points = roadGraph.rg
        # bs, nump, data_shape = points.shape
        # points = points[:, ::5].unsqueeze(1).reshape(bs, 16, 250, data_shape)
        obs_traj = history.clone()
        orient = torch.cat((batch["state/past/vel_yaw"].reshape(-1, 128, 10, 1)[masks > 0],
                            batch["state/current/vel_yaw"].reshape(-1, 128, 1, 1)[masks > 0]), -2)
        ag_type = batch["state/type"][masks > 0].unsqueeze(-1).unsqueeze(-1).repeat(1, 11, 1)
        obs_traj = torch.cat([obs_traj, past_cur_valid.unsqueeze(-1), orient, ag_type], -1)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time before forward:", start.elapsed_time(end))
        # start.record()

        traj, regions, map_samples, goals_pix, goals_meters = self.forward(obs_traj, points.clone(), num_peds_per_batch,
                                                                           past_cur_valid)
        # end.record()
        # torch.cuda.synchronize()
        # print("Time to forward:", start.elapsed_time(end))

        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        # start.record()

        # fig, ax = plt.subplots(1, 3)
        # ax[0].imshow(torch.log(regions).reshape(-1, 50, 50)[0].detach().cpu().numpy(), cmap="twilight")
        # ax[1].imshow(torch.log(gt_goalmap + 1e-6).reshape(-1, 50, 50)[0].detach().cpu().numpy(), cmap="PiYG")
        # ax[2].scatter(points[0, :, :, 0].detach().cpu().numpy(), points[0, :, :, 1].detach().cpu().numpy(), s=0.1)
        # ax[2].scatter(history[0, :, 0][obs_traj[0, :, -3] > 0].detach().cpu(),
        #               history[0, :, 1][obs_traj[0, :, -3] > 0].detach().cpu(), s=1, color="r")
        # ax[2].scatter(history[0, -1, 0].detach().cpu(),
        #               history[0, -1, 1].detach().cpu(), s=1, color="y")
        # ax[2].scatter(future_to_predict[0, :, 0].detach().cpu(),
        #               future_to_predict[0, :, 1].detach().cpu(), s=1, color="g")
        #
        # plt.show()
        loss = self.calc_losses(future_to_predict, gt_goalmap, traj, regions, goals_meters, future_valid, history)
        # end.record()
        # torch.cuda.synchronize()
        # print('calc_losses {}'.format(start.elapsed_time(end)))
        poses, confs = traj
        self.vis_pred(future_to_predict.detach().cpu(), history.detach().cpu(),
                      poses.detach().cpu(), points.detach().cpu(), future_valid.detach().cpu(),
                      past_cur_valid.detach().cpu(), confs=confs.detach().cpu())
        # optimmizations
        # start.record()

        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        # end.record()
        # torch.cuda.synchronize()
        # print('time after forward {}'.format(start.elapsed_time(end)))
        # print()
        # print()
        # sch = self.lr_schedulers()
        # sch.step(loss)
        return {'loss': loss}

    def calc_losses(self, gt_future, gt_goalmap, pred_out, pred_goalmap, pred_goal_meters, future_valid, history):
        # keep batch and sample dim

        loss_dict = {}

        bs = gt_future.shape[0]
        loss = torch.zeros(1).cuda()
        pred_traj, confs = pred_out

        loss_nll = pytorch_neg_multi_log_likelihood_batch(gt_future - history[:, -1:], pred_traj, confs, future_valid)
        loss += loss_nll
        loss_dict["loss_nll"] = loss_nll
        ade_ = torch.norm(
            ((gt_future - history[:, -1:]).unsqueeze(1) - pred_traj) * future_valid.unsqueeze(-1).unsqueeze(1),
            dim=(-1)).mean(-1)
        # choose the best samples
        best_samples_ade = torch.min(ade_, -1).values
        best_ade = best_samples_ade.mean()
        loss_dict["ade10"] = best_ade
        loss += LOSS_COEFS["BEST_ADE"] * best_ade
        loss += LOSS_COEFS["AVERAGE_ADE"] * ade_.mean()
        disp_from_pred_goal = torch.norm(((pred_goal_meters - pred_traj[:, :, -1]) * future_valid[:, -1:].unsqueeze(1)),
                                         dim=-1).mean()
        loss_dict["goal_inconsistency"] = disp_from_pred_goal
        loss += LOSS_COEFS["INNIER_FDE"] * disp_from_pred_goal
        # r_ade = torch.stack([torch.stack([(torch.norm(((gt_future - history[:, -1:]) - pred_traj[:, i]), dim=-1)[j][future_valid[j] > 0]).mean() for
        #      i in range(10)]).min() for j in range(len(gt_future))]).mean()
        # make plt fig with 2 plots:
        # 1. pred_goalmap.reshape(bs, 50, 50)[0]
        # 2. gt_goalmap.reshape(bs, 50, 50)[0]
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(torch.log(pred_goalmap).reshape(bs, 50, 50)[0].detach().cpu().numpy(), cmap="twilight")
        # ax[1].imshow(torch.log(gt_goalmap + 1e-6).reshape(bs, 50, 50)[0].detach().cpu().numpy(), cmap="PiYG")
        # plt.show()
        map_bce = BCELoss_class_weighted((1 / 2500, 2500))(pred_goalmap.reshape(bs, -1)*future_valid[:,-1:], gt_goalmap.reshape(bs, -1)*future_valid[:,-1:])


        loss += LOSS_COEFS["BCE"] * map_bce
        loss_dict["bce"] = map_bce
        if self.optimizers():
            lr = self.optimizers().optimizer.param_groups[0]['lr']
            loss_dict["lr"] = lr
        log_losses = {}
        for key, loss_ in loss_dict.items():
            # self.logger.experiment.add_scalar(f"train/{key}", loss_, self.global_step)
            log_losses[f"train/{key}"] = loss_
        wandb.log(log_losses)
        return loss

    def vis_pred(self, gt_future, history, pred_traj, objects_grouped, future_valid, past_cur_valid, confs=None):
        if (self.global_step + 1) % 500 == 0:
            bn = 0
            pred_traj_ = pred_traj.detach().cpu().numpy()[bn]

            # create figure
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111)
            if confs is not None:
                for sample in range(pred_traj_.shape[0]):
                    if confs[bn, sample].item() > 0.05:
                        ax.plot(pred_traj_[sample, :, 0], pred_traj_[sample, :, 1], "--", alpha=0.5)
                        ax.text(pred_traj_[sample, -1, 0], pred_traj_[sample, -1, 1], f"{confs[bn, sample]:.2}", color="b", fontsize=8)
            else:
                ax.scatter(pred_traj_[:, :, 0], pred_traj_[:, :, 1], c="r", s=0.2)
            ax.scatter(
                (gt_future[bn, :, 0] - history[bn, -1, 0])[future_valid[bn] > 0].reshape(-1),
                (gt_future[bn, :, 1] - history[bn, -1, 1])[future_valid[bn] > 0].reshape(-1),
                c="b", s=0.2)
            ax.scatter((history[bn, :, 0] - history[bn, -1, 0])[past_cur_valid[bn] > 0].reshape(-1),
                       (history[bn, :, 1] - history[bn, -1, 1])[past_cur_valid[bn] > 0].reshape(-1),
                       c="g", s=0.2)

            for i in range(objects_grouped.shape[1]):
                og_masked = (objects_grouped[bn, i, :, :2][objects_grouped[bn, i, :, 5] > 0] - history[
                    0, -1])
                x = og_masked[:, 0]
                y = og_masked[:, 1]
                ax.scatter(x, y, s=0.02)

            img = save_fig_to_numpy(fig)
            # close figure
            plt.close(fig)
            plt.close('all')
            plt.close('all')
            # self.logger.experiment.add_image("pred_traj", img, global_step=self.global_step, dataformats="HWC")
            wandb.log({"examples": [wandb.Image(img)]})
            pass

    def validation_step(self, batch, batch_idx):
        roadGraph = RoadGraph(batch["roadgraph_samples/xyz"], ids=batch["roadgraph_samples/id"],
                              types=batch["roadgraph_samples/type"],
                              valid=batch["roadgraph_samples/valid"])
        num_peds_per_batch = (batch["state/tracks_to_predict"] > 0).sum(-1)
        _, _, history, future_to_predict, _, _, \
        _, _, future_valid = \
            get_pose_from_batch_to_predict(batch)

        masks = batch["state/tracks_to_predict"].reshape(-1, 128) > 0
        past_valid = batch["state/past/valid"].reshape(-1, 128, 10)[masks]
        cur_valid = batch["state/current/valid"].reshape(-1, 128, 1)[masks]
        past_cur_valid = torch.cat([past_valid, cur_valid], dim=-1)
        gt_goalmap = self.generate_gt_goalmap(history, future_to_predict, future_valid)
        selector = np.arange(4, 80, 5)
        future_to_predict = future_to_predict[:, selector].cuda()
        future_valid = future_valid[:, selector].cuda()
        # points = roadGraph.all_objects(by_id=True)
        points = roadGraph.rg[:, ::2].view(-1, 50, 200, 6)

        obs_traj = history.clone()
        orient = torch.cat((batch["state/past/vel_yaw"].reshape(-1, 128, 10, 1)[masks > 0],
                            batch["state/current/vel_yaw"].reshape(-1, 128, 1, 1)[masks > 0]), -2)
        ag_type = batch["state/type"][masks > 0].unsqueeze(-1).unsqueeze(-1).repeat(1, 11, 1)
        obs_traj = torch.cat([obs_traj, past_cur_valid.unsqueeze(-1), orient, ag_type], -1)
        traj, regions, map_samples, goals_pix, goals_meters = self.forward(obs_traj, points.clone(), num_peds_per_batch,
                                                                           past_cur_valid)

        loss = self.calc_losses(future_to_predict, gt_goalmap, traj, regions, goals_meters, future_valid, history)

        return {'loss': loss}

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['loss'] for x in outputs]).mean()
        sch = self.lr_schedulers()
        sch.step(mean_loss)


    def test_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        # opt = torch.optim.Adam(self.parameters(), lr=0.0001)
        opt = torch.optim.AdamW(self.parameters(), lr=0.0001)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, verbose=True, factor=0.5)
        return [opt], lr_scheduler

    def generate_gt_goalmap(self, history, future_to_predict, future_valid):
        relative_goals = future_to_predict[:, -1, :2] - history[:, -1, :2]
        # generate torch grid of size 50x50
        grid = torch.zeros(1, 50, 50).repeat(future_to_predict.shape[0], 1, 1).to(future_to_predict.device)
        # center is (25,25) 1 pix = 4m
        relative_goals = relative_goals / 4
        # convert to grid
        relative_goals = relative_goals + 25
        relative_goals = relative_goals.long()
        # convert to grid
        relative_goals = relative_goals.clamp(0, 49)
        for i in range(future_to_predict.shape[0]):
            if future_valid[i, -1]:
                grid[i, relative_goals[i, 0], relative_goals[i, 1]] = 1
        return grid


def sanity_checks():
    import time

    Encoder().cuda().sanity_check()
    TrajPred().cuda().sanity_check()
    model = MapTransf().cuda()
    print(model)
    count_parameters(model)

    # measure time next:

    data = torch.randn(2, 256, 6).cuda()
    start = time.time()
    for i in range(20):
        model(data)
    print("sh: 2, 300, 256, 6 time is: ", (time.time() - start))

    # clear cahce
    torch.cuda.empty_cache()
    data = torch.randn(2, 20 * 256, 6).cuda()
    start = time.time()
    model(data)
    print("sh: 2, 20*256, 6 time is: ", (time.time() - start))


# main:
if __name__ == '__main__':
    from pytorch_lightning import Trainer

    wandb.init(project="waymo-PointTransf", entity="aleksey-postnikov", name="nll")
    # model = TrajPred().cuda()
    path = "/media/robot/hdd1/predictions_gan/GoalGanTr/lightning_logs/version_35/checkpoints/epoch=0-step=13663.ckpt"
    # load  model from path
    # model.load_state_dict(torch.load(path))
    model = TrajPred.load_from_checkpoint(path)
    trainer = Trainer(num_sanity_val_steps=0,
                      progress_bar_refresh_rate=10, accelerator="gpu", max_epochs=100)
    # trainer.validate(model)
    trainer.fit(model)
