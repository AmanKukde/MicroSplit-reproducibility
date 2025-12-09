import pooch
import numpy as np
from pathlib import Path


from microsplit_reproducibility.configs.factory import (
    create_algorithm_config,
    get_likelihood_config,
    get_loss_config,
    get_model_config,
)
from microsplit_reproducibility.datasets import create_train_val_datasets
from microsplit_reproducibility.configs.parameters.HT_H24 import get_microsplit_parameters
from microsplit_reproducibility.configs.data.HT_H24 import get_data_configs
from microsplit_reproducibility.datasets.HT_H24 import get_train_val_data
from microsplit_reproducibility.utils.io import load_checkpoint_path

from careamics.lightning import VAEModule
from microsplit_reproducibility.notebook_utils.HT_H24 import load_pretrained_model
from careamics.lvae_training import swt_stitching as swt
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import logging
import time

from torch.utils.data import DataLoader, Subset
from typing import List, Tuple
# --------------------------------------------------------------------
# LOGGER SETUP
# --------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

log = logging.getLogger(__name__)


ROOT_DIR = Path('/group/jug/aman/Datasets/HT_H24/')

DATA = pooch.create(
    path=ROOT_DIR/"data",
    base_url="https://download.fht.org/jug/msplit/ht_h24/data",
    registry={"ht_h24.zip": None},
)

NOISE_MODELS = pooch.create(
    path=ROOT_DIR/"noise_models",
    base_url="https://download.fht.org/jug/msplit/ht_h24/noise_models/",
    registry={"noise_model_Ch0.npz": None, "noise_model_Ch1.npz": None},
)

for fname in NOISE_MODELS.registry:
    NOISE_MODELS.fetch(fname, progressbar=False)

for fname in DATA.registry:
    DATA.fetch(fname, processor=pooch.Unzip(), progressbar=False)

# --------------------------------------------------------------------
# DATA CONFIG + HARDCODED MEAN/STD (ONLY TEST, TRAIN/VAL SKIPPED)
# --------------------------------------------------------------------
train_cfg, val_cfg, test_cfg = get_data_configs(sliding_window_flag=True)

experiment_params = get_microsplit_parameters(nm_path=NOISE_MODELS.path, batch_size=32)

start_total = time.time()
log.info("Starting inference script...\n\n")
t0 = time.time()

train_dset, val_dset, test_dset, data_stats = create_train_val_datasets(
    datapath=DATA.path / "ht_h24.zip.unzip/ht_h24",
    train_config=train_cfg,
    val_config=val_cfg,
    test_config=val_cfg,
    load_data_func=get_train_val_data,
)
log.info(f"Loaded Data in {time.time() - t0:.2f} sec\n\n")
experiment_params["num_workers"] = 4

print(f"Using test data ({test_dset.get_num_frames()} frames).")


# --------------------------------------------------------------------
# CHECKPOINT SELECTION (NO REDUNDANT LOGIC)
# --------------------------------------------------------------------
CKPT_DIR = Path("/group/jug/aman/Datasets/HT_H24/pretrained_checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

MODEL_CHECKPOINTS = pooch.create(
    path=str(CKPT_DIR),
    base_url="https://download.fht.org/jug/msplit/ht_h24/ckpts/",
    registry={"best.ckpt": None},
)

MODEL_CHECKPOINTS.fetch("best.ckpt", progressbar=False)
selected_ckpt = load_checkpoint_path(str(CKPT_DIR), best=True)
print("Using checkpoint:", selected_ckpt)

# --------------------------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------------------------
experiment_params["data_stats"] = data_stats
model_config = get_model_config(**experiment_params)
loss_config = get_loss_config(**experiment_params)
gaussian_lik_config, noise_model_config, nm_lik_config = get_likelihood_config(
    **experiment_params
)

experiment_config = create_algorithm_config(
    algorithm=experiment_params["algorithm"],
    loss_config=loss_config,
    model_config=model_config,
    gaussian_lik_config=gaussian_lik_config,
    nm_config=noise_model_config,
    nm_lik_config=nm_lik_config,
)

model = VAEModule(algorithm_config=experiment_config)
load_pretrained_model(model, selected_ckpt)
model.model.reset_for_inference(tile_size = (4,64,64))
# --------------------------------------------------------------------
# RUN SLIDING-WINDOW INFERENCE
# --------------------------------------------------------------------
# Compute stitching crop params (same ones that stitcher uses)


def run_inference_sliding(model, test_dset, batch_size=32, num_workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    loader = DataLoader(test_dset, pin_memory=True, num_workers=num_workers,
                        shuffle=False, batch_size=batch_size)

    global_idx = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting tiles"):
            inp = batch[0].to(device)
            rec, _ = model(inp)

            if model.model.predict_logvar is None:
                rec_img = rec
            else:
                rec_img, _ = torch.chunk(rec, chunks=2, dim=1)

            rec_np = rec_img.cpu().numpy()
            for i in range(rec_np.shape[0]):
                global_idx += 1
                yield rec_np[i], (global_idx - 1)


def batched_generator(model, loader, use_gpu_stitch=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    global_idx = 0  # track dataset patch index

    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting batches"):  # batch is just data
            batch = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch
            B = batch.shape[0]

            rec, _ = model(batch)

            if hasattr(model.model, "predict_logvar") and model.model.predict_logvar is not None:
                rec, _ = torch.chunk(rec, chunks=2, dim=1)

            if use_gpu_stitch:
                rec = rec.permute(0, 2, 3, 4, 1).contiguous()
            else:
                rec = rec.permute(0, 2, 3, 4, 1).cpu().numpy()

            # yield batch with computed indices
            indices = np.arange(global_idx, global_idx + B) if not use_gpu_stitch else torch.arange(global_idx, global_idx + B, device=device)
            global_idx += B

            yield rec, indices


# -----------------------------------
# Loader: only valid patches
# -----------------------------------
def create_loader_skip_padding(
    dset,
    inner_fraction: List[float],
    pad_per_side: Tuple[int,int,int] = (3,12,12),
    batch_size: int = 64,
    num_workers: int = 4
):
    """
    Return a DataLoader that only yields patches overlapping the valid (non-padded) region.
    """
    # compute inner crop params
    start_inners, _, inner_tile_sizes = swt._compute_inner_crop_params(
        dset.idx_manager.patch_spatial_dims,
        swt._parse_inner_fractions(inner_fraction, 3)
    )
    t0 = time.time()
    # get dataset indices that overlap valid region
    needed_indices = swt.compute_needed_patch_indices((15,1608,1608),
        dset.idx_manager.patch_shape[1:-1],test_dset.get_stride((4,32,32)), pad_per_side
    )
    log.info(f"needed_indices computerd in {time.time() - t0:.2f} sec\n\n")
    # subset dataset (needed_indices are original dataset indices)
    subset = Subset(dset, needed_indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader, needed_indices  # return original dataset indices for stitching


# -----------------------------------
# Batched generator that yields original dataset indices
# -----------------------------------
def batched_generator(
    model,
    loader,
    original_indices: List[int],
    use_gpu_stitch: bool = False
):
    """
    Yields batches of model outputs along with original dataset indices
    so stitching works correctly.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    with torch.no_grad():
        start_idx = 0
        for batch in loader:
            batch_data = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch
            B = batch_data.shape[0]

            rec, _ = model(batch_data)
            if hasattr(model.model, "predict_logvar") and model.model.predict_logvar is not None:
                rec, _ = torch.chunk(rec, chunks=2, dim=1)

            if use_gpu_stitch:
                rec = rec.permute(0,2,3,4,1).contiguous()
                indices = torch.tensor(original_indices[start_idx:start_idx+B], device=device)
            else:
                rec = rec.permute(0,2,3,4,1).cpu().numpy()
                indices = np.array(original_indices[start_idx:start_idx+B])

            start_idx += B
            yield rec, indices


# -----------------------------------
# Usage example
# -----------------------------------
pad_per_side = (5,48,48)
inner_fraction = [0.5, 0.5, 0.5]
batch_size = 32
num_workers = 4
use_gpu_stitch = False  # or True

# create loader skipping padded tiles
loader, needed_indices = create_loader_skip_padding(
    test_dset,
    inner_fraction=inner_fraction,
    pad_per_side=pad_per_side,
    batch_size=batch_size,
    num_workers=num_workers
)

# generator yields outputs + original dataset indices
gen = batched_generator(
    model,
    loader,
    needed_indices,
    use_gpu_stitch=use_gpu_stitch
)
t_stitch = time.time()
# stitch predictions deterministically
stitched, counts = swt.stitch_predictions_windowed(
    gen,
    test_dset,
    debug=True,
    inner_fraction=inner_fraction,
    use_gpu=use_gpu_stitch,
    vectorized=True
)

log.info(f"Stitching completed in {time.time() - t_stitch:.2f} sec\n\n")

mean_params, std_params = test_dset.get_mean_std()
stitched_predictions = stitched * std_params[
    "target"
].squeeze().reshape(1, 1, 1, 1, -1) + mean_params["target"].squeeze().reshape(
    1, 1, 1, 1, -1
)

import tifffile as tf
t_save = time.time()
tf.imwrite("prediction_microsplit_swt.tiff", stitched_predictions.transpose(0,4,1,2,3))
tf.imwrite("counts_microsplit_swt.tiff", counts.transpose(0,4,1,2,3))
log.info(f"TIFF writing completed in {time.time() - t_save:.2f} sec\n\n")
log.info(f"TOTAL RUNTIME = {time.time() - start_total:.2f} sec\n\n")
log.info("Inference + stitching complete.\n\n")

print("Inference + stitching complete.")
