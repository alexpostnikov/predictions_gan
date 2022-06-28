import math

from torch.utils.data import Dataset, DataLoader
import pathlib
import numpy as np
from six.moves import cPickle as pickle
import torch
from typing import Tuple, Optional
import collections
import wandb

class RoadGraph:
    def __init__(self, rg: torch.Tensor, types: torch.Tensor, ids: torch.Tensor, valid: torch.Tensor):
        self.rg = rg.view(-1, 20000, 3)
        self.types = types.view(-1, 20000, 1)
        self.ids = ids.view(-1, 20000, 1)
        self.valid = valid.view(-1, 20000, 1)

        self.rg = torch.cat([self.rg, self.types, self.ids, self.valid], dim=2)
        self.valid_ind = 5
        pass

    def grouped_by_object(self, object_id: int, num_points_max: int = 64) -> torch.Tensor:
        out = []
        max_seq_len = 0
        for i in range(self.rg.shape[0]):
            object = self.rg[i][(self.rg[i, :, 4] == object_id)]
            object = object[object[:, self.valid_ind] > 0]
            if object.shape[0] > num_points_max:
                pick_every = object.shape[0] // num_points_max +1
                object = object[::pick_every]
            seq_len = object.shape[0]
            if seq_len > max_seq_len:
                max_seq_len = seq_len

            out.append(object)
        out_tensor = torch.zeros((self.rg.shape[0], num_points_max, self.rg.shape[2]), device=self.rg.device)
        for i in range(self.rg.shape[0]):
            out_tensor[i, :, :] = torch.cat([out[i], torch.zeros((num_points_max - out[i].shape[0], self.rg.shape[2]),
                                                                 device = out[i].device)], dim=0)
        return out_tensor


    def all_objects(self, by_type=0, by_id=0) -> torch.Tensor:
        out = []
        extra = []
        max_seq_len = 0
        assert by_type == 0 or by_id == 0
        assert by_type + by_id == 1

        selector, searcher = None, None
        if by_type:
            selector = 3
            searcher = self.grouped_by_type
        if by_id:
            selector = 4
            searcher = self.grouped_by_object

        unique_objects = torch.unique(self.rg[:, :, selector])
        for obj_id in unique_objects:
            if obj_id == -1:
                continue
            object = searcher(obj_id)
            seq_len = object.shape[1]
            if seq_len > max_seq_len:
                max_seq_len = seq_len
            if object.ndim == 3:
                out.append(object)
            if object.ndim == 4:
                out.append(object[:, 0])
                extra.append(object[:, 1:])
        out_tensor = torch.zeros((self.rg.shape[0], len(out), max_seq_len, self.rg.shape[2]), device=self.rg.device)
        for j in range(len(out)):
            out_tensor[:, j, :, :] = torch.cat([out[j],
                                                torch.zeros((self.rg.shape[0], max_seq_len - out[j].shape[1], self.rg.shape[2]),device=self.rg.device)
                                                ], dim=1)
        return out_tensor

    def split_big_line_to_smaller(self, line, n_splits, max_points_per_split=256):
        nump_points, data_shape = line.shape

        np_per_new_line = math.ceil(nump_points / n_splits)
        div_coef = 1
        if np_per_new_line > max_points_per_split:
            div_coef = max(1, np_per_new_line//max_points_per_split+1)
            # np_per_new_line = max_points_per_split

        out = torch.zeros(n_splits, max_points_per_split, data_shape).to(line.device)

        for i in range(n_splits):
            points = line[i * np_per_new_line: (i + 1) * np_per_new_line][::div_coef]
            out[i, :len(points)] = points
        return out

    def grouped_by_type(self, type_id: int, num_points_max: int = 256) -> torch.Tensor:
        out = []
        max_seq_len = 0
        for i in range(self.rg.shape[0]):
            object = self.rg[i][(self.rg[i, :, 3] == type_id)]
            object = object[object[:, self.valid_ind] > 0]
            if object.shape[0] > num_points_max:
                pick_every = object.shape[0] // num_points_max + 1
                # if pick_every > 5:
                #     num_splits = pick_every // 5 + 1
                #     object = self.split_big_line_to_smaller(object, num_splits, num_points_max)
                #     [out.append(object[i]) for i in range(object.shape[0])]
                #     continue
                object = object[::pick_every]
            out.append(object)
        out_tensor = torch.zeros((self.rg.shape[0], num_points_max, self.rg.shape[2]), device=self.rg.device)
        for i in range(self.rg.shape[0]):
            out_tensor[i, :, :] = torch.cat([out[i], torch.zeros((num_points_max - out[i].shape[0], self.rg.shape[2]),
                                                                 device=self.rg.device)], dim=0)
        return out_tensor



class WaymoDS(Dataset):
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
    return maps[:, :, :, :] # scqueeze 2 times


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

    state_to_predict_with_neighbours = torch.zeros(state_to_predict.shape[0], 128, state_to_predict.shape[1], 2, device=state_to_predict.device)
    real_bn = 0
    for original_bn, index in masks.nonzero():
        state_to_predict_with_neighbours[real_bn] = past_current[original_bn]
        state_to_predict_with_neighbours[real_bn, index], state_to_predict_with_neighbours[real_bn, 0] = state_to_predict_with_neighbours[real_bn, 0].clone(), state_to_predict_with_neighbours[real_bn, index].clone()
        real_bn += 1

    future_to_predict = future_pose[masks.nonzero(as_tuple=True)]

    # create 2d rotation matrix
    orient = data["state/current/bbox_yaw"][masks]
    past_valid = data["state/past/valid"].reshape(-1,128,10)[:,:,-1][masks]
    rot, trans = create_rot_matrix(state=state_to_predict, bbox_yaw=orient, past_valid=past_valid)
    # move past_current to local coordinate system
    # inverse rot matrix
    #
    past_current_local = torch.bmm(rot, (state_to_predict - trans).permute(0, 2, 1).to(torch.float64)).permute(0, 2, 1).to(torch.float32)
    state_to_predict_with_neighbours_local = (state_to_predict_with_neighbours.reshape(past_current_local.shape[0], -1, 2) - trans).permute(0,2,1)
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

import re
np_str_obj_array_pattern = re.compile(r'[SaUO]')
string_classes = (str, bytes)


def simpl_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    out_batch = []
    if isinstance(batch[0], dict) and "state/current/x" in batch[0]:
        out_batch = {}
        for key in batch[0]:
            if key == "scenario/id":
                continue
            k = [batch[i][key] for i in range(len(batch))]
            if isinstance(k[0], np.ndarray):
                k = torch.from_numpy(np.stack(k, axis=0))
            else:
                raise
            out_batch[key] = k
        return out_batch
    raise  # TypeError(default_collate_err_msg_format.format(elem_type))

def d_collate_fn(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum(x.numel() for x in batch)
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except:
            scenario_ids = [sc.numpy().tobytes().decode("utf-8") for sc in batch]
            return scenario_ids
            # scenarios_id = []
            # for bn, scenario in enumerate(scenario_id):
            #     [scenarios_id.append(scenario) for i in range((mask.nonzero()[:, 0] == bn).sum())]
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise  # TypeError(default_collate_err_msg_format.format(elem.dtype))
            if batch[0].flags["WRITEABLE"]:
                return d_collate_fn([torch.as_tensor(b) for b in batch])
            else:
                return d_collate_fn([torch.as_tensor(np.copy(b)) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: d_collate_fn([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(d_collate_fn(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [d_collate_fn(samples) for samples in transposed]

    raise  # TypeError(default_collate_err_msg_format.format(elem_type))


# main
if __name__ == "__main__":
    import tqdm
    import matplotlib.pyplot as plt
    import matplotlib


    # test waymo dataset
    ds_path = "/media/robot/hdd1/waymo_ds/"
    in_path = "/media/robot/hdd1/waymo_ds/training_mapstyle/index_file.txt"
    waymo_dataset = WaymoDS(data_path=ds_path, index_file=in_path)
    for num, waymo_data in enumerate(tqdm.tqdm(waymo_dataset)):
        if num>2:
            break

    waymo_dataset = WaymoDS(data_path=ds_path, index_file=in_path,
                                  rgb_index_path="/media/robot/hdd1/waymo_ds/rendered/train/index.pkl",
                                  rgb_prefix="/media/robot/hdd1/waymo_ds/")
    loader = DataLoader(waymo_dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=d_collate_fn)
    for num, waymo_data in enumerate(tqdm.tqdm(loader)):
        res = preprocess_batch_to_predict_with_img(waymo_data)

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
        #  polylines:
        pc = waymo_data["roadgraph_samples/xyz"].reshape(-1,20000,3)
        types = waymo_data["roadgraph_samples/type"].reshape(-1,20000,1)

        fig, ax = plt.subplots(3, 2)
        ax[0, 0].scatter(past_current_local[0, :, 0], past_current_local[0, :, 1])
        # ax[0] equal axis
        ax[0, 0].set_aspect('equal')
        # title
        ax[0, 0].set_title("history in LCS")
        ax[0, 1].scatter(state_to_predict[0, :, 0], state_to_predict[0, :, 1])
        ax[0, 1].set_aspect('equal')
        ax[0, 1].set_title("Hist in GCS")
        ax[1, 0].imshow(imgs[0].permute(1, 2, 0))
        for rs_type in waymo_data["roadgraph_samples/type"].reshape(-1, 20000, 1)[0].unique():
            pc_t = pc[0][(types == rs_type)[0, :, 0]]
            ax[1, 1].scatter(pc_t[:, 0], pc_t[:, 1], s=0.02)
        # add poses with red stars
        num_preds_in_batch = (waymo_data["state/tracks_to_predict"][0] > 0).sum()
        for i in range(num_preds_in_batch):
            ax[1, 1].scatter(state_to_predict[i, :, 0], state_to_predict[i, :, 1], s=4.2, c="r", marker="*")
            ax[1, 1].set_aspect('equal')
        center = (state_to_predict[0, -1, 0], state_to_predict[0, -1, 1])
        # generate a circle with radius 100:
        circle = plt.Circle(center, 100, color='r', fill=False)
        # 125 with another color
        circle2 = plt.Circle(center, 125, color='b', fill=False)
        # 150 with another color
        circle3 = plt.Circle(center, 150, color='g', fill=False)
        # 175 with another color
        circle4 = plt.Circle(center, 175, color='y', fill=False)

        ax[1, 1].add_artist(circle)
        ax[1, 1].add_artist(circle2)
        ax[1, 1].add_artist(circle3)
        ax[1, 1].add_artist(circle4)
        ax[1, 1].set_title("Roadgraph")

        # from first batch of tensor of  pc get points that are in 50m l1 distance to state_to_predict[0, -1, :]
        # and plot them in the second plot
        to_include = torch.norm(pc[0, :, :2] - state_to_predict[0, -1, :].unsqueeze(0), 1, dim=1) < 50

        # measure time of operation above:
        import time
        start = time.time()
        # get points that are in 50m l1 distance to state_to_predict[0, -1, :]
        to_include = torch.norm(pc[0, :, :2] - state_to_predict[0, -1, :].unsqueeze(0), 1, dim=1) < 50
        # measure time of operation above:
        end = time.time()
        print("time for 50m l1 distance: ", end - start)
        pc_t = pc[0][to_include]
        types = types[0][to_include]
        for rs_type in types.unique():
            ax[2, 0].scatter(pc_t[(types == rs_type)[:, 0], 0], pc_t[(types == rs_type)[ :, 0], 1], s=0.02)
        ax[2, 0].set_aspect('equal')
        ax[2, 0].set_title("Roadgraph in 50m")

        roadGraph = RoadGraph(waymo_data["roadgraph_samples/xyz"], ids=waymo_data["roadgraph_samples/id"],
                              types=waymo_data["roadgraph_samples/type"],
                              valid=waymo_data["roadgraph_samples/valid"])
        # measure time of operation below:
        start = time.time()
        roadGraph.grouped_by_object(1)
        end = time.time()
        print("time for grouped_by_object: ", end - start)
        # measure time of operation below:
        start = time.time()
        objects_grouped = roadGraph.all_objects(by_id=1)
        end = time.time()
        print("time for all_objects: ", end - start)
        print("shape of objects_grouped: ", objects_grouped.shape)

        # objects_grouped is tensor of shape [bs, n_objects, n_points, 3]
        # scatter all objects in the last plot
        for i in range(objects_grouped.shape[1]):
            og_masked = objects_grouped[0, i][objects_grouped[0, i, :, 5] > 0]
            x = og_masked[:, 0]
            y = og_masked[:, 1]
            ax[2, 1].scatter(x, y, s=0.02)
        ax[2, 1].set_aspect('equal')
        ax[2, 1].set_title("Roadgraph in by obj")


        plt.show()
        pass
        # close
        plt.close(fig)
