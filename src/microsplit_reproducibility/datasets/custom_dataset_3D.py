import os
from pathlib import Path
from typing import Literal

import numpy as np

import tifffile

from careamics.lvae_training.dataset import DataSplitType
from careamics.lvae_training.dataset.utils.data_utils import get_datasplit_tuples
from careamics.dataset.dataset_utils.dataset_utils import reshape_array
from careamics.lvae_training.dataset.types import DataType 


def load_one_file(fpath):
    """Load a single 3D image file."""
    data = tifffile.imread(fpath).astype(np.float32)
    if len(data.shape) == 3:
        axes = 'ZXY'
    elif len(data.shape) == 4:
        axes = 'SZXY' 
    elif len(data.shape) == 5:
        axes = 'STZXY'
    else: 
        raise ValueError(f"Invalid data shape: {data.shape}")
    data = reshape_array(data, axes)
    data = data.reshape(-1, data.shape[-3], data.shape[-2], data.shape[-1])
    return data


def load_data(datadir):
    data_path = Path(datadir)

    channel_dirs = sorted(p for p in data_path.iterdir() if p.is_dir())
    channels_data = []

    for channel_dir in channel_dirs:
        image_files = sorted(f for f in channel_dir.iterdir() if f.is_file())
        channel_images = [load_one_file(image_path) for image_path in image_files]
            
        channel_stack = np.concatenate(channel_images, axis=0) # FIXME: this line works iff images have
        # a singleton channel dimension. Specify in the notebook or change with `torch.stack`??
        channels_data.append(channel_stack)
    
    final_data = np.stack(channels_data, axis=-1)
    return final_data


def get_train_val_data(
    data_config,
    datadir,
    datasplit_type: DataSplitType,
    val_fraction=None,
    test_fraction=None,
    **kwargs,
):
    data = load_data(datadir)
    train_idx, val_idx, test_idx = get_datasplit_tuples(
        val_fraction, test_fraction, len(data)
    )
    # FIXME: this is a hack to make the data split work with 2D custom datasets
    val_idx = train_idx
    test_idx = train_idx
    # breakpoint()
    # if data_config.data_type == DataType.Care3D:
        # train_idx = [];

    if datasplit_type == DataSplitType.All:
        data = data.astype(np.float32)
    elif datasplit_type == DataSplitType.Train:
        data = data[train_idx].astype(np.float32)[:,-69:]
    elif datasplit_type == DataSplitType.Val:
        data = data[val_idx].astype(np.float32)[:,-69:]
    elif datasplit_type == DataSplitType.Test:
        data = data[test_idx].astype(np.float32)[:,-69:]
    else:
        raise Exception("invalid datasplit")

    return data
