# %%
import os
from pathlib import Path
import numpy as np
import pickle
import pooch
import tifffile
from careamics.lightning import VAEModule
import matplotlib.pyplot as plt
from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config
)
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.utils.utils import plot_input_patches
from microsplit_reproducibility.datasets import create_train_val_datasets

# Dataset specific imports...
from microsplit_reproducibility.configs.parameters.custom_dataset_2D import (
    get_microsplit_parameters,
)
from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data


from usplit.core.tiff_reader import save_tiff, load_tiff

import random
import os
import numpy as np
import torch
import pickle
import ml_collections
import glob
import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from usplit.training import create_dataset, create_model
import matplotlib.pyplot as plt
from usplit.core.loss_type import LossType
from usplit.config_utils import load_config
from usplit.analysis.lvae_utils import get_img_from_forward_output
from usplit.analysis.plot_utils import clean_ax
from usplit.core.data_type import DataType
from usplit.core.psnr import PSNR
from usplit.analysis.plot_utils import get_k_largest_indices,plot_imgs_from_idx
from usplit.core.psnr import PSNR, RangeInvariantPsnr
from usplit.core.data_split_type import DataSplitType
from usplit.analysis.stitch_prediction import stitch_predictions
from usplit.analysis.mmse_prediction import get_dset_predictions
import pdb
from usplit.core.model_type import ModelType
torch.multiprocessing.set_sharing_strategy('file_system')

# %%

# %% [markdown]
# # **Step 2.1:** Data Preparation

# %% [markdown]
# ### OR path to your own data

dataset = "HAGEN"
if dataset == "PAVIA_ATN":
    DATA_PATH = Path("/group/jug/aman/Datasets/PAVIA_ATN")
    ckpt_dir= "/home/aman.kukde/Projects/models/paper_models/PaviaATN/ActTub/LeanLC/"
elif dataset == "HAGEN":
    DATA_PATH = Path("/group/jug/aman/Datasets/HAGEN")
    ckpt_dir= "/home/aman.kukde/Projects/models/paper_models/Hagen/MitoVsAct/LeanLC/"

NM_PATH = Path("./noise_models/")

# %% [markdown]
# ### Next, we load the image data to be processed

# %%
# setting up train, validation, and test data configs
img_sz = 64
multiscale_lowres_count = 5
sliding_window_flag = False

train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=(img_sz, img_sz), num_channels=2, sliding_window_flag= sliding_window_flag, multiscale_lowres_count = multiscale_lowres_count
)
# setting up MicroSplit parametrization
experiment_params = get_microsplit_parameters(
    algorithm = "musplit",
    img_size=(img_sz, img_sz),
    batch_size=32,
    num_epochs=10,
    multiscale_count=multiscale_lowres_count,
    noise_model_path=NM_PATH,
    target_channels=2,
)

# create the dataset
train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=DATA_PATH,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=test_data_config,
    load_data_func=get_train_val_data,
)

# test_dset.reduce_data([0]) #!comment this out later
# %%


image_size_for_grid_centers = 32
mmse_count = 64
custom_image_size = 64
batch_size = 32
num_workers = 6
eval_datasplit_type = DataSplitType.Test

config = load_config(ckpt_dir)
config = ml_collections.ConfigDict(config)
old_image_size = None
with config.unlocked():
    if 'test_fraction' not in config.training:
        config.training.test_fraction =0.0
        
    if 'datadir' not in config:
        config.datadir = ''
    if 'encoder' not in config.model:
        config.model.encoder = ml_collections.ConfigDict()
        assert 'decoder' not in config.model
        config.model.decoder = ml_collections.ConfigDict()
    
        config.model.encoder.dropout = config.model.dropout
        config.model.decoder.dropout = config.model.dropout
        config.model.encoder.blocks_per_layer = config.model.blocks_per_layer
        config.model.decoder.blocks_per_layer = config.model.blocks_per_layer
        config.model.encoder.n_filters = config.model.n_filters
        config.model.decoder.n_filters = config.model.n_filters
        
    if 'multiscale_retain_spatial_dims' not in config.model.decoder:
        config.model.decoder.multiscale_retain_spatial_dims = False
        
    if 'res_block_kernel' not in config.model.encoder:
        config.model.encoder.res_block_kernel = 3
        assert 'res_block_kernel' not in config.model.decoder
        config.model.decoder.res_block_kernel = 3
    
    if 'res_block_skip_padding' not in config.model.encoder:
        config.model.encoder.res_block_skip_padding = False
        assert 'res_block_skip_padding' not in config.model.decoder
        config.model.decoder.res_block_skip_padding = False
    
    if config.data.data_type == DataType.CustomSinosoid:
        if 'max_vshift_factor' not in config.data:
            config.data.max_vshift_factor = config.data.max_shift_factor
            config.data.max_hshift_factor = 0
        if 'encourage_non_overlap_single_channel' not in config.data:
            config.data.encourage_non_overlap_single_channel = False
            
    if 'skip_bottom_layers_count' in config.model:
        config.model.skip_bottom_layers_count = 0
        
    if 'logvar_lowerbound' not in config.model:
        config.model.logvar_lowerbound = None
    if 'train_aug_rotate' not in config.data:
        config.data.train_aug_rotate = False
    if 'multiscale_lowres_separate_branch' not in config.model:
        config.model.multiscale_lowres_separate_branch = False
    if 'multiscale_retain_spatial_dims' not in config.model:
        config.model.multiscale_retain_spatial_dims = False
    config.data.train_aug_rotate=False
    
    if 'randomized_channels' not in config.data:
        config.data.randomized_channels = False
        
    if 'predict_logvar' not in config.model:
        config.model.predict_logvar=None
    
    if 'batchnorm' in config.model and 'batchnorm' not in config.model.encoder:
        assert 'batchnorm' not in config.model.decoder
        config.model.decoder.batchnorm = config.model.batchnorm
        config.model.encoder.batchnorm = config.model.batchnorm
    if 'conv2d_bias' not in config.model.decoder:
        config.model.decoder.conv2d_bias = True
        
    
    if custom_image_size is not None:
        old_image_size = config.data.image_size
        config.data.image_size = custom_image_size
    if image_size_for_grid_centers is not None:
        old_grid_size = config.data.get('grid_size', "grid_size not present")
        config.data.grid_size = image_size_for_grid_centers
        config.data.val_grid_size = image_size_for_grid_centers

    config.model.mode_pred = True

    config.model.skip_nboundary_pixels_from_loss = None
    if config.model.model_type == ModelType.UNet and 'n_levels' not in config.model:
        config.model.n_levels = 4
    
    if config.model.model_type == ModelType.UNet and 'init_channel_count' not in config.model:
        config.model.init_channel_count = 64
    
    if 'skip_receptive_field_loss_tokens' not in config.loss:
        config.loss.skip_receptive_field_loss_tokens = []
    
    if 'lowres_merge_type' not in config.model.encoder:
        config.model.encoder.lowres_merge_type = 0
print(config)

# %%
def get_best_checkpoint(ckpt_dir):
    output = []
    for filename in glob.glob(ckpt_dir + "/*_best.ckpt"):
        output.append(filename)
    assert len(output) == 1, '\n'.join(output)
    return output[0]

# %%
# train_dset = val
with config.unlocked():
    if old_image_size is not None:
        config.data.image_size = old_image_size

if config.data.target_separate_normalization is True:
    mean_fr_model, std_fr_model = train_dset.compute_individual_mean_std()
else:
    mean_fr_model, std_fr_model = train_dset.get_mean_std()


model = create_model(config, mean_fr_model,std_fr_model)

ckpt_fpath = get_best_checkpoint(ckpt_dir)
checkpoint = torch.load(ckpt_fpath, weights_only=False)

_ = model.load_state_dict(checkpoint['state_dict'])
model.eval()
_= model.cuda()

model.set_params_to_same_device_as(torch.Tensor(1).cuda())

print('Loading from epoch', checkpoint['epoch'])

# %%
from careamics.lvae_training.eval_utils import get_predictions, get_device
mmse_count=64
grid_size=32 
num_workers=6
batch_size= 128*5

stitched_predictions_, stitched_stds_ = get_predictions(
                                            model=model,
                                            dset=test_dset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            mmse_count=mmse_count,
                                            tile_size=(64,64),
                                            grid_size=grid_size,
                                            sliding_window_flag = sliding_window_flag
                                        )

def function_to_process_stitched_predictions(stitched_predictions, stitched_stds, dset, key, TARGET_CHANNEL_IDX_LIST = [0,1]):
            
    stitched_predictions = stitched_predictions[key]
    stitched_stds = stitched_stds[key]

    stitched_predictions = stitched_predictions[..., : len(TARGET_CHANNEL_IDX_LIST)]
    stitched_stds = stitched_stds[..., : len(TARGET_CHANNEL_IDX_LIST)]

    mean_params, std_params = dset.get_mean_std()
    unnorm_stitched_predictions = stitched_predictions * std_params["target"].squeeze().reshape(1, 1, 1, -1) + mean_params["target"].squeeze().reshape(1, 1, 1, -1)
    return unnorm_stitched_predictions, stitched_predictions, stitched_stds

stitched_predictions, norm_stitched_predictions, stitched_stds = function_to_process_stitched_predictions(stitched_predictions_, stitched_stds_, train_dset, dataset, [0,1])


import dill
from usplit.core.tiff_reader import save_tiff
save_tiff(f'/group/jug/aman/results/Results_usplit_64/{dataset}/pred_test_dset_usplit_og.tiff',stitched_predictions.transpose(0,3,1,2))
with open(f'/group/jug/aman/results/Results_usplit_64/{dataset}/pred_test_dset_usplit_og.pkl',"wb") as f:
    dill.dump(stitched_predictions,f)