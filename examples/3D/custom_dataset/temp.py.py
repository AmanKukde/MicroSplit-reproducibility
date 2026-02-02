# %% [markdown]
# # **Step 2:** Using a Trained <nobr>Micro$\mathbb{S}$plit</nobr> Model

# %% [markdown]
# # Introduction - what does this notebook do?
# In this notebook, we will learn how to use a previously trained <nobr>Micro$\mathbb{S}$plit</nobr> model to unmix superimposed cellular structures in fluorescent microscopy data. For this, we will use a held-out portion of your custom dataset, which we have not used during training.
# 
# **More specifically we will:**
# * make full frame predictions and inspect the results,
# * explore the possibility of <nobr>Micro$\mathbb{S}$plit</nobr> to sample predictions from the learned posterior of possible solutions,
# * visually inspect the data uncertainty we can deduce from posterior samples, and
# * quantitatively evaluate the model using several metrics. 

# %% [markdown]
# # Let's do it, let's use <nobr>Micro$\mathbb{S}$plit</nobr>!

# %% [markdown]
# **You are new to Jupyter notebooks?** Don't worry, if you take the time to read all our explanations, we will guide you through them and you will understand a lot. Still, you will likely end up less frustrated, if you do not even start with the ambition to interpret the purpose of every line of code.
# Let's start with a nice example, the imports to enable the remainder of this notebook. Ignore it (unless you know what you are doing) and just click **⇧*Shift* + ⏎*Enter*** to execute this (and all other) code cells. 

# %%
%load_ext autoreload
%autoreload 2

# %%
import os
import platform
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pooch

from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
)
from microsplit_reproducibility.utils.io import load_checkpoint_path
from microsplit_reproducibility.utils.utils import plot_input_patches_3d
from microsplit_reproducibility.datasets import create_train_val_datasets

from careamics.lightning import VAEModule

# Dataset specific imports...
from microsplit_reproducibility.configs.parameters.custom_dataset_3D import get_microsplit_parameters
from microsplit_reproducibility.configs.data.custom_dataset_3D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_3D import get_train_val_data

from torch.utils.data import DataLoader
import numpy as np

# %% [markdown]
# # **Step 2.1:** Data Preparation

# %% [markdown]
# ### Download example data
# Depending on your internet connection, this will take a while...
# 
# Appropriate noise models will only be downloaded if they were not created by executing the notebook `00_noisemodels.ipynb`.

# %%
# DATA = pooch.create(
#     path="./",
#     base_url="https://download.fht.org/jug/msplit/custom3D",
#     registry={f"custom_3Ddata_example.zip": None},
# )
# for fname in DATA.registry:
#     DATA.fetch(fname, processor=pooch.Unzip(), progressbar=True)

# DATA_PATH = DATA.abspath / (DATA.registry_files[0] + ".unzip/data/")

# %% [markdown]
# ### OR set the path to your own data
# Important: the path should end with `data/`

# %%
DATA_PATH = Path("/group/jug/aman/Datasets/Care3D/data/")

# %% [markdown]
# ### Setup the path to the noise models
# This is the path to the noise models that you trained in the notebook **00_noisemodels.ipynb**

# %%
NM_PATH = Path("/group/jug/aman/Datasets/Care3D/noise_models/")

# %% [markdown]
# ### Next, we load the data we selected above...
# 
# As in the training notebook, depending on the amount of GPU memory you have available, you might want to adjust the `BATCH_SIZE`. The default is 32, but you can reduce it to 16 if you run out of memory by changing the <i> batch_size </i> parameter.
# 
# You also need to specify the `NUM_CHANNELS` parameter, which controls the number of channels in the input data depending on how many channels you want to split.
# 
# Finally, ensure that the `PATCH_SIZE` parameter (i.e., the patch size in (`Z`, `Y`, `X`) you want to use to train the model) is properly set given the size of your data and of you GPU. **Note that the patch size you use here MUST be the same you set for training!!!**

# %%
NUM_CHANNELS = 3
"""The number of channels considered for the splitting task."""
NUM_Z_SLICES = 69
"""The number of z slices in the input data."""
BATCH_SIZE = 32
"""The batch size for training."""
PATCH_SIZE = (9, 128, 128)

GRID_SIZE = (9, 64, 64)
"""The size of the patches fed to the network for training in (Z, Y, X)."""
EPOCHS = 3
"""The number of epochs to train the network."""

assert len(PATCH_SIZE) == 3, "Patch size must be a tuple of length 3 (Z, Y, X) since we are using 3D data."
assert PATCH_SIZE[0] <= NUM_Z_SLICES, "Patch size in Z dimension must be smaller than or equal to the number of z slices in the input data."

# %%
# setting up train, validation, and test data configs
train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=PATCH_SIZE,
    grid_size = GRID_SIZE,
    num_channels=NUM_CHANNELS,
    sliding_window_flag = False,
    multiscale_lowres_count= 1
)

# setting up MicroSplit parametrization
experiment_params = get_microsplit_parameters(
    algorithm="denoisplit",
    img_size=(9,64,64),
    batch_size=BATCH_SIZE,
    multiscale_count=1,
    noise_model_path=NM_PATH,
    target_channels=NUM_CHANNELS,
    
)

# %% [markdown]
# ### Configure `num_workers`
# In Windows and MacOS, setting `num_workers > 0` for dataloaders would cause out-of-memory issue and might crash the system.

# %%
# start the download of required files
train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=DATA_PATH,
    train_config=train_data_config,
    val_config=val_data_config,
    test_config=val_data_config,
    load_data_func=get_train_val_data
)

# %%
print(test_dset)
print(test_dset._data.shape)
print(test_dset[0][0].shape)
print(test_dset[0][1].shape)

# %%
def get_num_workers():
    """Utility function to set num_workers based on OS."""
    if platform.system() == "Windows" or platform.system() == "Darwin":
        return 0
    else:
        return 3  # or any other number suitable for your system

experiment_params["num_workers"] = get_num_workers()

# %% [markdown]
# ### 👇 🖼️ pick Validation or Test data to be used! 👇
# **Side Note:** if you want to be well prepared for the next notebook on calibration and error estimation, you will have to do both (ie, after you have run this notebook with <i> evaluate_on_validation_data = False </i> in the cell below, return here, set <i> evaluate_on_validation_data = True </i> and re-run the subsequent cells). This will ensure that the files required in the next notebook for both the test and validation datasets are prepared correctly. More detail is provided in section 2.7
# 

# %%
# by default, we will use the held out test data, 
# but if you want, we can switch to using the 
# validation data instead
evaluate_on_validation_data = False
if evaluate_on_validation_data:
    print('Will use validation data', end='')
    dset = val_dset
else:
    print('Will use test data', end='')
    dset = test_dset
print(f' (containing a total of {dset.get_num_frames()} frames).')

# %% [markdown]
# ### Finally, let's look at bits of the data you chose!
# 

# %%
# plot_input_patches_3d(dataset=dset, num_channels=2, num_samples=3, patch_size=128)

# %% [markdown]
# # **Step 2.2:** Picking <nobr>Micro$\mathbb{S}$plit</nobr> Model to Use

# %% [markdown]
# Let's see what model checkpoints you have available in the `checkpoints` folder, where the model(s) you trained with the notebook `01_train.ipynb` are stored, and the `pretrained_checkpoints` folder, where we just downloaded models into (as long as any such models exist on our servers).

# %%
# # Recursively search for .ckpt files in 'checkpoints' folder
# ckpt_folder = Path("./checkpoints")
# ckpt_folders = set()
# for file in ckpt_folder.rglob("*.ckpt"):
#     ckpt_folders.add(file.parent)
# ckpt_folders = sorted(ckpt_folders)

# def list_available_model_checkpoint_folders():
#     print('These models you have trained have been found:')
#     if len(ckpt_folders)==0:
#         print(' ❌ None!')
#     else:
#         for file in ckpt_folders:
#             print(' 🟢', file)

# %% [markdown]
# ## 👇 🤖 please pick one of the available models! 👇

# %%
# list_available_model_checkpoint_folders()

# %%
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# if you cut&past a path to the chosen ckpt down here,
# we will use that checkpoint, otherwise we pick one automatically.
# user_selected_ckpt_folder = 'checkpoints'
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# %%
# ckpt_folder = "/group/jug/ashesh/training/disentangle/2506/D30-M3-S0-L8/6/" #:Liver
ckpt_folder = "/group/jug/ashesh/training/disentangle/2411/D30-M3-S0-L8/26/" #:LC
ckpt_folder = "/group/jug/ashesh/training/disentangle/2411/D30-M3-S0-L8/8/" #Zebrafish Depth 9

    
if ckpt_folder=='':
    print("🚨 CRITICAL: No model checkpoint seems to be available!")
else:
    selected_ckpt = load_checkpoint_path(str(ckpt_folder+"BaselineVAECL_best.ckpt"), best=True)
    print("✅ Selected model checkpoint:", selected_ckpt)

# %% [markdown]
# # **Step 2.3:** Prepare <nobr>Micro$\mathbb{S}$plit</nobr> Model
# Next, we create all the configs needed to instatiate the selected model. These lines are not very intuitive and if you don't intend to dive really deep into CAREamics and the internals of <nobr>Micro$\mathbb{S}$plit</nobr>, you might just execute these cells and move on.

# %%
# making our data_stas known to the experiment (model) we prepare
experiment_params["data_stats"] = data_stats

# setting up model config (using default parameters)
model_config = get_model_config(**experiment_params)

# NOTE: The creation of the following configs are not strictly necessary for prediction,
#     but they ARE currently expected by the create_algorithm_config function below.
#     They act as a placeholder for now and we will work to remove them in a following release
loss_config = get_loss_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)

# finally, assemble the full set of experiment configurations...
experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
)

# %% [markdown]
# ### Create model and load checkpoint

# %%
model = VAEModule(algorithm_config=experiment_config)

# %%
model.model.reset_for_inference((9,128,128))

# %%
import torch
ckpt_dict = torch.load(selected_ckpt, map_location='cuda', weights_only=True)
model.model.load_state_dict(ckpt_dict["state_dict"], strict=False)

# %% [markdown]
# ### Wanna be done quickly?
# <div class="alert alert-block alert-info">
# <b>Note:</b> Being quick will make the last notebook `03_calibration.ipynb` work less well. Still, if you just want to see some results the selected <nobr>Micro$\mathbb{S}$plit</nobr> model can generate, feel invited to crop down on the evaluation data we loaded above and save some time.
# </div>

# %% [markdown]
# # TODO add desctiption how to reduce hw also

# %%
reduce_data = False
import copy
dset = copy.deepcopy(test_dset)
if reduce_data:
    print("Using REDUCED evaluation data for quick'n'dirty testing!")
    dset.reduce_data(t_list=[0], h_start = -256*2,h_end = -256, w_start=-256*2, w_end = -256)
else:
    print('Using the full set of evaluation data!')
    print(f'(More specifically, I will use {dset.get_num_frames()} frames for evaluations.)') 

# %%
# model.model.reset_for_inference(tile_size = (9,64,64))

# %% [markdown]
# # **Step 2.4:** Predictions on Uncropped Data
# If a single frame has the size of a typical microscopy image, we cannot feed the entire image to <nobr>Micro$\mathbb{S}$plit</nobr> (your GPU would run out of memory). Hence, we predict results for smaller chunks of the full image, so called tiles.
# 
# When we perform tiled predicitons, we use '***inner padding***'. See a detailed explanation in our [µSplit paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Ashesh_uSplit_Image_Decomposition_for_Fluorescence_Microscopy_ICCV_2023_paper.pdf), and the schematic below.
# ![InnerAndOuterPadding.png](attachment:dbe65ad8-8c38-45bd-8301-a7e574a84cf2.png)

# %%
from microsplit_reproducibility.notebook_utils.custom_dataset_3D import get_unnormalized_predictions, get_target, get_input

# Here we use a small helper function that returns the final results
# after performing Inner Padding, as mentioned above.
# Note also that it also returns `stitched_stds`, which is the pixel-wise
# standard deviation (std) between the posterior samples we have averaged
# while computing the MMSE per patch during tiled predictions. These 
# values will become most useful at the end of this notebook and in even 
# more so in `03_calibration.ipynb` for calibration and error estimations.
stitched_predictions, norm_stitched_predictions, stitched_stds = get_unnormalized_predictions(
    model, 
    dset,
    data_key=dset._fpath.name, # path to the data to unpack the dataset dictionary
    mmse_count = 50,
    num_workers=experiment_params['num_workers'],
    batch_size= 32
)

# load inputs and noisy targets (needed for plotting later on)
inp = get_input(dset).sum(axis=-1)
tar = get_target(dset)

# %%
from microsplit_reproducibility.utils.paper_metrics import compute_psnr_only

print("Metric, followed by values for channel 1 and channel 2")
metrics = compute_psnr_only(tar[0,:,32:-32,32:-32], stitched_predictions[0,:,32:-32,32:-32])


# %%
import tifffile as tf 
from pathlib import Path
save_path = Path("/group/jug/aman/MicroSplit_Runs/MSR_13Jan26/Care3D/")
save_path.mkdir(exist_ok= True, parents= True)
tf.imwrite(save_path/"prediction_og.tiff",stitched_predictions)
# tf.imwrite(save_path/"counts_swt.tiff",counts.cpu().numpy())
