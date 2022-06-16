from torch.utils.data import Dataset, DataLoader
import pathlib
import numpy as np
from six.moves import cPickle as pickle
import torch
from typing import Tuple, Optional
import collections
import matplotlib.pyplot as plt

class WaymoGoalGanDS(Dataset):
    """
    Loads the Waymo dataset (preprocessed, saved in torch map style).
    """

    def __init__(self, data_path: str, index_file: str, rgb_index_path: str = None, rgb_prefix: str = "",
                 rgb_index_path1: str = None, rgb_prefix1: str = ""):
        """
        Args:
        :param data_path: path to the dataset
        :param index_file: path to the index file
        :param rgb_index_path: path to the rgb index file (absolute)
        :param rgb_prefix: prefix to saved rgb images pathes (Optional)
        :param rgb_index_path1: path to the rgb index file (absolute)  (optinally can load 2 images per sample)
        :param rgb_prefix1: prefix to saved rgb images pathes (Optional)
        """

        self.ds_path = data_path
        # assert self.ds_path.exists()
        assert pathlib.Path(self.ds_path).exists()
        self.index_file = index_file
        assert pathlib.Path(self.index_file).exists()
        self.index = None
        self.read_files = {}
        with open(index_file, 'rb') as f:
            self.index = pickle.load(f)
        self.rgb_index_path = rgb_index_path
        if rgb_index_path is not None:
            assert pathlib.Path(rgb_index_path).exists()
            self.rgb_prefix = rgb_prefix
            self.rgb_loader = RgbLoader(rgb_index_path)
        self.rgb_index_path1 = rgb_index_path1
        if rgb_index_path1 is not None:
            assert pathlib.Path(rgb_index_path1).exists()
            self.rgb_prefix1 = rgb_prefix1
            self.rgb_loader1 = RgbLoader(rgb_index_path1)

    def __getitem__(self, item):
        path = self.index[item][0]
        frame_idx = self.index[item][1]
        # load np from path
        if path not in self.read_files:
            # join ds_path + path
            path_glob = pathlib.Path(self.ds_path) / path
            data = np.load(path_glob, allow_pickle=True)["data"].reshape(-1)[0]

            new_data = {}
            for key, val in data.items():
                new_sub_d = {}
                for vkey in data[key].keys():
                    if "roadgraph123" not in vkey:
                        new_sub_d[vkey] = data[key][vkey]
                new_data[key] = new_sub_d
            data = new_data
            self.read_files[path] = new_data
        else:
            data = self.read_files[path]

        chunk = data[frame_idx]
        #chunk["roadgraph_samples/xyz"]
        if self.rgb_index_path:
            rgb = torch.tensor(
                self.rgb_loader.load_singlebatch_rgb(chunk, prefix=self.rgb_prefix).astype(np.float32))
            chunk["rgbs"] = np.zeros((8, rgb.shape[1], rgb.shape[2], rgb.shape[3]), dtype=np.float32)
            chunk["rgbs"][:rgb.shape[0]] = rgb
            chunk["rgbs"] = chunk["rgbs"][np.newaxis]
        if self.rgb_index_path1:
            rgb = torch.tensor(
                self.rgb_loader1.load_singlebatch_rgb(chunk, prefix=self.rgb_prefix1).astype(np.float32))
            chunk["rgbs2"] = np.zeros((8, rgb.shape[1], rgb.shape[2], rgb.shape[3]), dtype=np.float32)
            chunk["rgbs2"][:rgb.shape[0]] = rgb
            chunk["rgbs2"] = chunk["rgbs2"][np.newaxis]

        # chunk["file"] = path
        if len(self.read_files) > 2:
            self.read_files.clear()
        #             print("clearing")
        return chunk

    def __len__(self):
        return len(self.index)


def namefile_to_file_name_index(names, files):
    out_files = {}
    for i, (name, file) in enumerate(zip(names, files)):
        if file in out_files:
            out_files[file].append([name, i])
        else:
            out_files[file] = [[name, i]]
    return out_files


class RgbLoader:
    """
    Loads rgb images from a given index file.
    """

    def __init__(self, index_path="rendered/index.pkl"):
        self.index_path = index_path
        self.index_dict = self.load_dict()
        self.opened_files = {}
        self.rgbs = {}

    def load_dict(self, filename_=None):
        if filename_ is None:
            filename_ = self.index_path

        with open(filename_, 'rb') as f:

            try:
                lim = 1e5
                dicts = []
                while 1 and lim > 0:
                    lim -= 1
                    loaded_dict = pickle.load(f)
                    dicts.append(loaded_dict)
                    # out_d = {**out_d, **loaded_dict}

            except EOFError:
                pass
        d = {k: v for e in dicts for (k, v) in e.items()}
        print(f"loaded index file contains {1e5 - lim} indexes to files")
        return d

    def load_batch_rgb(self, data, prefix="tutorial/"):
        batch = data['scenario/id']
        try:
            scenario_ids = [sc.cpu().numpy().tobytes().decode("utf-8") for sc in batch]
        except:
            scenario_ids = batch
        # logging.info(f'----load_batch_rgb() scenario_ids = {scenario_ids}')
        mask = (data["state/tracks_to_predict"] > 0)
        scenarios_id = []
        for bn, scenario in enumerate(scenario_ids):
            num_nonzero_in_each_batch = (mask.nonzero()[:, 0] == bn).sum()
            [scenarios_id.append(scenario) for _ in range(num_nonzero_in_each_batch)]

        aids = data["state/id"][mask]
        names = [scenarios_id[i] + str(aids[i].item()) for i in range(len(aids))]
        # logging.info(f'----load_batch_rgb() names = {names}')
        files = [self.index_dict[name] for name in names]
        file_name_index = namefile_to_file_name_index(names, files)
        batch_rgb = np.ones((len(aids), 224, 224, 3), dtype=np.float32)
        for file, name_index in file_name_index.items():
            indexes = [ni[1] for ni in name_index]
            # logging.info(f' --------load_batch_rgb() name_index = {name_index}, prefix= {prefix}')
            batch_rgb[indexes] = self.load_rgb_by_name_file(name_index, file, prefix)
        return batch_rgb

    def load_singlebatch_rgb(self, data, prefix="tutorial/"):

        batch = data['scenario/id']
        try:
            scenario_ids = "".join(([sc.tobytes().decode("utf-8") for sc in batch]))
        except:
            scenario_ids = batch
        # logging.info(f'----load_batch_rgb() scenario_ids = {scenario_ids}')
        mask = (data["state/tracks_to_predict"] > 0)
        # repeat scenario_ids number of nonzero elements to predict
        scenarios_id = [scenario_ids for _ in range(len(mask.nonzero()[0]))]
        # for bn, scenario in enumerate(scenario_ids):
        #     num_nonzero_in_each_batch = (mask.nonzero()[:, 0] == bn).sum()
        #     [scenarios_id.append(scenario) for _ in range(num_nonzero_in_each_batch)]

        aids = data["state/id"][mask]
        names = [scenarios_id[i] + str(aids[i].item()) for i in range(len(aids))]
        # logging.info(f'----load_batch_rgb() names = {names}')
        files = [self.index_dict[name] for name in names]
        file_name_index = namefile_to_file_name_index(names, files)
        batch_rgb = np.ones((len(aids), 224, 224, 3), dtype=np.float32)
        for file, name_index in file_name_index.items():
            indexes = [ni[1] for ni in name_index]
            # logging.info(f' --------load_batch_rgb() name_index = {name_index}, prefix= {prefix}')
            batch_rgb[indexes] = self.load_rgb_by_name_file(name_index, file, prefix)
        if len(self.opened_files) > 25:
            self.opened_files.clear()
            self.rgbs.clear()
        return batch_rgb

    def load_rgb_by_name_file(self, name_index, file_path, prefix="tutorial/"):
        if file_path not in self.opened_files:
            # logging.info(f'----file_path: {prefix + file_path}')
            rgbs = np.load(prefix + file_path, allow_pickle=True)["rgb"].reshape(-1)[0]
            self.rgbs = {**self.rgbs, **rgbs}
            self.opened_files[file_path] = 1
        out = []
        for n_i in name_index:
            # logging.info(f'--------load_rgb_by_name_file() n_i = {n_i}')
            out.append(self.rgbs[n_i[0]])
        out = np.concatenate([out])

        return out[:, 0]


def rotate_neighbours(poses_b: torch.Tensor, rot_mat: torch.Tensor, masks: torch.Tensor):
    """
    Rotates the neighbours of each pedestrian by the given rotation matrix.
    :param poses_b: (batch_size, num_peds,  11, 2)
    :param rot_mat: (batch_size_real, 3, 3)
    :param masks: (batch_size, num_peds)
    :return: (batch_size_real, num_peds, 11, 2)
    """

    batch_size, num_peds, _, _ = poses_b.shape
    batch_size_real = masks.sum().item()
    out = torch.zeros((batch_size_real, num_peds, 11, 2)).to(poses_b.device)
    for i, index in enumerate(masks.nonzero()):
        out[i, :, :, :] = torch.bmm(
            poses_b[index[0]].reshape(1, -1, 2) - poses_b[index[0], index[1], 0].reshape(1, 1, 2),
            rot_mat[i:i + 1, :2, :2].float()).reshape(128, 11, 2)
    return out


def batch_to_poses_all(data: dict) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ''' data is a dictionary with keys:
            "state/current/x": (bs, 128,)
            "state/current/y": (bs, 128,)
            "state/past/x": (bs, 128*10,)
            "state/past/y": (bs, 128*10,)
            "state/future/x": (bs, 128*80,)
            "state/future/y": (bs, 128*80,)
            "state/future/valid": (bs, 128*80,)
            "state/current/valid": (bs, 128,)
            "state/past/valid": (bs, 128*11,)
    concating past and current, move to local coordinate system,
    with current pose orientation fixed (looking along x axis)
    '''
    bs, num_peds = data["state/current/x"].shape
    current = torch.stack([data["state/current/x"], data["state/current/y"]], dim=2)  # (bs, 128, 2)
    current = current.reshape(bs, num_peds, 1, 2)  # (bs, 128, 1, 2)
    # reshape past to (bs, num_peds, 10, 2)
    past = torch.stack([data["state/past/x"], data["state/past/y"]], dim=2)
    past = past.reshape(bs, num_peds, 10, 2)
    # reshape future to (bs, num_peds, 80, 2)
    future = torch.stack([data["state/future/x"], data["state/future/y"]], dim=2)
    future = future.reshape(bs, num_peds, 80, 2)

    # reshape future valid to (bs, num_peds, 80)
    future_valid = torch.stack([data["state/future/valid"]], dim=2)
    future_valid = future_valid.reshape(bs, num_peds, 80)
    # reshape current valid to (bs, num_peds)
    current_valid = torch.stack([data["state/current/valid"]], dim=2)
    current_valid = current_valid.reshape(bs, num_peds)
    # reshape past valid to (bs, num_peds, 11)
    past_valid = torch.stack([data["state/past/valid"]], dim=2)
    past_valid = past_valid.reshape(bs, num_peds, 10)
    # past_current valid:
    past_current_valid = torch.stack([past_valid, current_valid], dim=2)
    # cat past and current
    past_current = torch.cat([past, current], dim=2)
    # move past_current to local coordinate system
    current_repeated = current.unsqueeze(2).repeat(1, 1, 1, 11, 1)  # shape (bs, 128, 1, 11, 2)
    hist_local = past_current.unsqueeze(2).repeat(1, 1, num_peds, 1,
                                                  1) - current_repeated  # shape (bs, 128, 128, 11, 2)

    # move future to local coordinate system
    future_repeated = current.unsqueeze(2).repeat(1, 1, 1, 80, 1)  # shape (bs, 128, 1, 80, 2)
    future_local = future.unsqueeze(2).repeat(1, 1, num_peds, 1, 1) - future_repeated  # shape (bs, 128, 128, 80, 2)
    return hist_local, past_current, future_local, future, future_valid, past_current_valid


def get_imgs_from_batch(data: dict) -> torch.Tensor:
    """
    :param data: dict of data from WaymoSophieDS
    :return:
    """
    # state/to/predict
    masks = data["state/tracks_to_predict"].reshape(-1, 128) > 0
    maps = data["rgbs"].reshape(data["rgbs"].shape[0], -1, data["rgbs"].shape[3], data["rgbs"].shape[4],
                                data["rgbs"].shape[5])
    maps = maps[masks.nonzero(as_tuple=True)]
    maps = maps.permute(0, 3, 1, 2) / 255.
    return maps[:, :, ::2, ::2] # scqueeze 2 times


def get_pose_from_batch_to_predict(data: dict):
    """
    :param data: dict of data from WaymoSophieDS
    :return:
    """
    # state/to/predict
    masks = data["state/tracks_to_predict"].reshape(-1, 128) > 0
    # current pose
    current_pose = torch.cat(
        [data["state/current/x"].reshape(-1, 128, 1, 1), data["state/current/y"].reshape(-1, 128, 1, 1)], dim=3)
    past_pose = torch.cat(
        [data["state/past/x"].reshape(-1, 128, 10, 1), data["state/past/y"].reshape(-1, 128, 10, 1)], dim=3)
    future_pose = torch.cat(
        [data["state/future/x"].reshape(-1, 128, 80, 1), data["state/future/y"].reshape(-1, 128, 80, 1)], dim=3)
    # cat past and current
    past_current = torch.cat([past_pose, current_pose], dim=2)
    past_valid = data["state/past/valid"].reshape(-1, 128, 10) > 0
    past_current[:, :, :-1][past_valid == 0] = 0
    state_to_predict = past_current[masks.nonzero(as_tuple=True)]

    state_to_predict_with_neighbours = torch.zeros(state_to_predict.shape[0], 128, state_to_predict.shape[1], 2)
    real_bn = 0
    for original_bn, index in masks.nonzero():
        state_to_predict_with_neighbours[real_bn] = past_current[original_bn]
        state_to_predict_with_neighbours[real_bn, index], state_to_predict_with_neighbours[real_bn, 0] = state_to_predict_with_neighbours[real_bn, 0].clone(), state_to_predict_with_neighbours[real_bn, index].clone()
        real_bn += 1

    future_to_predict = future_pose[masks.nonzero(as_tuple=True)]

    # create 2d rotation matrix
    orient = data["state/current/bbox_yaw"][masks]
    past_valid = data["state/past/valid"].reshape(-1, 128, 10)[:, :, -1][masks]
    rot, trans = create_rot_matrix(state=state_to_predict, bbox_yaw=orient, past_valid=past_valid)
    # move past_current to local coordinate system
    # inverse rot matrix

    past_current_local = torch.bmm(rot, (state_to_predict - trans).permute(0, 2, 1).to(torch.float64)).permute(0, 2, 1).to(torch.float32)
    state_to_predict_with_neighbours_local = (state_to_predict_with_neighbours.reshape(past_current_local.shape[0], -1, 2) - trans).permute(0, 2, 1)
    state_to_predict_with_neighbours_local = torch.bmm(rot, state_to_predict_with_neighbours_local.to(torch.float64)).permute(0, 2, 1).to(torch.float32)
    state_to_predict_with_neighbours_local = state_to_predict_with_neighbours_local.reshape(past_current_local.shape[0], -1, 11, 2)
    # move future to local coordinate system
    future_to_predict_local = torch.bmm(rot, (future_to_predict - trans).permute(0, 2, 1).to(torch.float64)).permute(0, 2, 1).to(torch.float32)
    future_valid = data["state/future/valid"].reshape(-1, 128, 80)[masks]
    return past_current_local, future_to_predict_local, state_to_predict, future_to_predict, rot, trans, \
           state_to_predict_with_neighbours_local, state_to_predict_with_neighbours, future_valid


def preprocess_batch_to_predict_with_img(data):
    out = get_pose_from_batch_to_predict(data)
    past_current_local, future_to_predict_local, state_to_predict, future_to_predict,\
    rot, trans, state_to_predict_with_neighbours_local, state_to_predict_with_neighbours, future_valid = out
    imgs = get_imgs_from_batch(data)

    return past_current_local, future_to_predict_local, state_to_predict, future_to_predict, rot, trans, imgs,\
           state_to_predict_with_neighbours_local, state_to_predict_with_neighbours, future_valid


def create_rot_matrix(state: torch.Tensor, bbox_yaw: torch.Tensor, past_valid: torch.Tensor = None):
    """
    :param state_masked: torch.Tensor [bs, 11, 2]; 11 - number of positions observed
    :param bbox_yaw: torch.Tensor [bs, 1]
    :return: torch.Tensor [bs, 2, 2] - transformation matrix, torch.Tensor [bs, 1, 2] - translation vector
    """
    last_step = (state[:, -1] - state[:, -2])
    bbox_yaw_my = torch.atan2(last_step[:, 1], last_step[:, 0])
    bbox_yaw[past_valid != 0] = bbox_yaw_my[past_valid != 0]
    rot_mat = torch.zeros([state.shape[0], 2, 2], device=state.device, dtype=torch.float64)
    rot_mat[:, 0, 0] = torch.cos(bbox_yaw)
    rot_mat[:, 0, 1] = -torch.sin(bbox_yaw)
    rot_mat[:, 1, 0] = torch.sin(bbox_yaw)
    rot_mat[:, 1, 1] = torch.cos(bbox_yaw)

    trans_vec = state[:, -1].reshape(-1, 1, 2)
    rot_mat_to_right_looking = torch.inverse(rot_mat)
    return rot_mat_to_right_looking, trans_vec

def convert_gt_dxdy_to_prob_mask(final_poses: torch.Tensor, valid_mask: torch.Tensor,
                                 max_fp=(150, 100), min_fp=(-50, -100), center=(0, 0)):
    """
    given batch of 2d poses in meters in local coords, create a probability mask of shape 33x33 for each batch
    :param final_poses: torch.Tensor [bs, 2]
    :param valid_mask: torch.Tensor [bs]
    :param max_fp: tuple (x, y)
    :param min_fp: tuple (x, y)
    :param center: tuple (x, y)
    :return: torch.Tensor [bs, 33, 33]
    """
    # create grid
    x_grid = torch.linspace(min_fp[0], max_fp[0], 33)
    y_grid = torch.linspace(min_fp[1], max_fp[1], 33)
    x_grid, y_grid = torch.meshgrid(x_grid, y_grid)
    # x_grid = x_grid.reshape(-1, 1)
    # y_grid = y_grid.reshape(-1, 1)
    grid = torch.stack([x_grid, y_grid], dim=-1)
    # convert to local coords
    grid = grid - torch.tensor(center)
    # # convert to global coords
    # grid = torch.bmm(final_poses.reshape(-1, 1, 2), grid.reshape(1, -1, 2)).reshape(-1, 2)
    # create probability mask
    prob_mask = torch.zeros([final_poses.shape[0], 33, 33], device=grid.device, dtype=torch.float32)
    for i in range(final_poses.shape[0]):
        dist = (torch.norm(grid - final_poses[i], dim=-1))
        prob_mask[i] = torch.exp(-dist.permute(1,0))
    prob_mask = prob_mask / (torch.sum(prob_mask, dim=(1, 2), keepdim=True)+1e-6)

    return prob_mask





def d_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    out_batch = []
    if isinstance(batch[0], dict) and "state/current/x" in batch[0]:
        for elem in batch:

            elem_out = {}
            masks = elem["state/tracks_to_predict"].reshape(128)>0
            elem_out["to_pred"] = masks[np.newaxis]
            cur = np.stack([elem["state/current/x"], elem["state/current/y"]], axis=1).reshape(128, 1, 2)[masks]
            past = np.stack([elem["state/past/x"].reshape(128,10), elem["state/past/y"].reshape(128,10)], axis=2)[masks]
            #make np range with step 5 such that it will contain: 4, 9, 14, 19, 24, ... 79
            selector = np.arange(4, 80, 5)


            elem_out["in_xy"] = np.concatenate([past, cur], axis=1)


            elem_out["gt_xy"] = np.stack([elem["state/future/x"].reshape(128,80), elem["state/future/y"].reshape(128,80)],
                                     axis=2)[masks][:, selector]
            elem_out["orient"] = elem["state/current/bbox_yaw"][masks]
            elem_out["gt_valid"] = (elem["state/future/valid"].reshape(128, 80)[masks] > 0)[:, selector]
            elem_out["in_valid"] = np.concatenate([elem["state/past/valid"].reshape(128, 10),
                                               elem["state/current/valid"].reshape(128, 1)], axis=1)[masks]
            if "rgbs" in elem:
                elem_out["scene_img"] = elem["rgbs"][0][:masks.sum()]
            if "rgbs2" in elem:
                elem_out["local_image"] = elem["rgbs2"][0][:masks.sum()]

            out_batch.append(elem_out)
        batch = out_batch
        out_batch = {}
        for key in batch[0].keys():
            out_batch[key] = torch.cat([torch.tensor(batch[i][key]) for i in range(len(batch))], dim=0)
        rot, trans = create_rot_matrix(state=out_batch["in_xy"],
                                       bbox_yaw=out_batch["orient"],
                                       past_valid=out_batch["in_valid"][:,-2])
        out_batch["in_dxdy"] = torch.bmm(rot.float(), (out_batch["in_xy"] - trans).permute(0, 2, 1)).permute(0, 2, 1)
        out_batch["gt_dxdy"] = torch.bmm(rot.float(), (out_batch["gt_xy"] - trans).permute(0, 2, 1)).permute(0, 2, 1)
        out_batch["prob_mask"] = convert_gt_dxdy_to_prob_mask(out_batch["gt_dxdy"][:, -1], out_batch["gt_valid"][:, -1])
        out_batch["global_patch"] = out_batch['scene_img'][:, ::7, ::7]
        from model.model_modules import Patch_gen
        local_pathces = [Patch_gen.gen_patch_waymo(out_batch["scene_img"].permute(0, 3, 1, 2),
                                                   (out_batch["gt_dxdy"] * out_batch["gt_valid"].unsqueeze(2))[:, i])
                         for i in range(out_batch["gt_dxdy"].shape[1])]
        #stack
        out_batch["local_patches"] = torch.stack(local_pathces, dim=1)

        # permute every value at dict to make batch dim #1, not #0
        for key in out_batch:
            if key == "scene_img" or key == "local_image":
                out_batch[key] = out_batch[key].permute(0, 3, 1, 2)
                continue
            if key == "prob_mask":
                continue
            if key == "global_patch":
                out_batch[key] = out_batch[key].permute(0, 3, 1, 2)
                continue
            if out_batch[key].ndim == 3:
                out_batch[key] = out_batch[key].permute(1, 0, 2)
            elif out_batch[key].ndim == 4:
                out_batch[key] = out_batch[key].permute(1, 0, 2, 3)
            elif out_batch[key].ndim == 5:
                out_batch[key] = out_batch[key].permute(1, 0, 2, 3, 4)
            elif out_batch[key].ndim == 2:
                out_batch[key] = out_batch[key].permute(1, 0)
        return out_batch

    raise  # TypeError(default_collate_err_msg_format.format(elem_type))


# main
if __name__ == "__main__":
    import tqdm
    import matplotlib.pyplot as plt
    import matplotlib
    # test waymo dataset
    ds_path = "/media/robot/hdd1/waymo_ds/"
    in_path = "/media/robot/hdd1/waymo_ds/training_mapstyle/index_file.txt"
    waymo_dataset = WaymoGoalGanDS(data_path=ds_path, index_file=in_path)
    for num, waymo_data in enumerate(tqdm.tqdm(waymo_dataset)):
        if num>10:
            break

    waymo_dataset = WaymoGoalGanDS(data_path=ds_path, index_file=in_path,
                                  rgb_index_path="/media/robot/hdd1/waymo_ds/rendered/train/index.pkl",
                                  rgb_prefix="/media/robot/hdd1/waymo_ds/")
    loader = DataLoader(waymo_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=d_collate_fn)
    for num, waymo_data in enumerate(tqdm.tqdm(loader)):
        waymo_data
        continue
        past_current_local, future_to_predict_local, state_to_predict, future_to_predict, rot, trans, imgs, \
        with_n_local, with_n, valid = res
        # past_current_local shape is [bs, 11, 2]
        # future_to_predict_local shape is [bs, 11, 2]
        # state_to_predict shape is [bs, 11, 2]
        # future_to_predict shape is [bs, 11, 2]
        # rot shape is [bs, 2, 2]
        # trans shape is [bs, 1, 2]
        # res shape is [bs, 3, 224, 224]

        # create figure with 3 plots with equal axes
        fig, ax = plt.subplots(3,1)
        ax[0].scatter(past_current_local[0, :, 0], past_current_local[0, :, 1])
        # ax[0] equal axis
        ax[0].set_aspect('equal')
        ax[1].scatter(state_to_predict[0, :, 0], state_to_predict[0, :, 1])
        ax[1].set_aspect('equal')
        ax[2].imshow(imgs[0].permute(1, 2, 0))

        fig.show()
        pass
        # close
        plt.close(fig)







