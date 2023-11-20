import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    # Separate the data into separate lists
    image_data_list, qpos_data_list, action_data_list, is_pad_list = zip(*batch)

    # Get sequence lengths
    lengths = torch.tensor([ t.shape[0] for t in image_data_list ])

    # Pad each list separately
    image_data = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t) for t in image_data_list], batch_first=True)
    qpos_data = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t) for t in qpos_data_list], batch_first=True)
    action_data = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t) for t in action_data_list], batch_first=True)
    is_pad = torch.nn.utils.rnn.pad_sequence([torch.Tensor(t) for t in is_pad_list], batch_first=True)

    # Compute mask
    mask = (image_data != 0)

    return image_data, qpos_data, action_data, is_pad, lengths, mask


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    mask = []
    all_qpos_data = []
    all_action_data = []
    
    longest_data = 0
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action))

        if len(qpos) > longest_data:
            longest_data = qpos.shape[0]
    
    # pad all data to longest data
    padded_qpos = torch.zeros((longest_data, qpos.shape[1]))
    padded_action = torch.zeros((longest_data, action.shape[1]))
    for i in range(num_episodes):
        qpos = all_qpos_data[i]
        action = all_action_data[i]
        padded_qpos = torch.zeros((longest_data, qpos.shape[1]))
        padded_action = torch.zeros((longest_data, action.shape[1]))
        padded_qpos[:qpos.shape[0]] = qpos
        padded_action[:action.shape[0]] = action
        all_qpos_data[i] = padded_qpos
        all_action_data[i] = padded_action

        # Create masks
        mask.append(torch.cat([torch.ones_like(qpos), torch.zeros((longest_data - qpos.shape[0], qpos.shape[1]))]))

    mask = torch.stack(mask)
    all_qpos_data = padded_qpos
    all_action_data = padded_action

    # Calculate the sum of the data, ignoring the zeros
    sum_action_data = (all_action_data * mask).sum(dim=[0, 1], keepdim=True)
    sum_qpos_data = (all_qpos_data * mask).sum(dim=[0, 1], keepdim=True)

    # Calculate the number of non-zero values
    num_non_zero_action = mask.sum(dim=[0, 1], keepdim=True)
    num_non_zero_qpos = mask.sum(dim=[0, 1], keepdim=True)

    # Calculate the mean, ignoring the zeros
    action_mean = sum_action_data / num_non_zero_action
    qpos_mean = sum_qpos_data / num_non_zero_qpos

    # Calculate the standard deviation, ignoring the zeros
    action_var = ((all_action_data - action_mean) ** 2 * mask).sum(dim=[0, 1], keepdim=True) / num_non_zero_action
    qpos_var = ((all_qpos_data - qpos_mean) ** 2 * mask).sum(dim=[0, 1], keepdim=True) / num_non_zero_qpos
    action_std = torch.sqrt(action_var)
    qpos_std = torch.sqrt(qpos_var)

    # Clip the standard deviations
    action_std = torch.clip(action_std, 1e-2, np.inf)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, variable_length=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=10, collate_fn=collate_fn_padd)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=10, collate_fn=collate_fn_padd)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim

### env utils
def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

