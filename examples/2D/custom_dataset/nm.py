


import platform
from pathlib import Path

import pooch
import numpy as np
import matplotlib.pyplot as plt

from careamics import CAREamist
from careamics.models.lvae.noise_models import (
    GaussianMixtureNoiseModel,
    create_histogram,
)
from careamics.lvae_training.dataset import DataSplitType
from careamics.config import GaussianMixtureNMConfig
from careamics.config import create_n2v_configuration

from microsplit_reproducibility.configs.data.custom_dataset_2D import get_data_configs
from microsplit_reproducibility.datasets.custom_dataset_2D import get_train_val_data
from microsplit_reproducibility.utils.utils import plot_probability_distribution
DATA_PATH = Path("/group/jug/aman/Datasets/Chicago/data/")


train_data_config, val_data_config, test_data_config = get_data_configs(
    image_size=(64, 64), num_channels=2
)

input_data = get_train_val_data(
    data_config=train_data_config,
    datadir=DATA_PATH,
    datasplit_type=DataSplitType.All,
    val_fraction=0.1,
    test_fraction=1,
)
print(input_data.shape)
train_data = input_data[0:-1:10, :, :, :].squeeze()
print(train_data.shape)


_, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(train_data[0, ..., 0])
ax[0].set_title("Input channel 1")
ax[1].imshow(train_data[0, ..., 1])
ax[1].set_title("Input channel 2")
plt.show()

def get_num_workers():
    """Utility function to set num_workers based on OS."""
    if platform.system() == "Windows" or platform.system() == "Darwin":
        return 0
    else:
        return 3 


config = create_n2v_configuration(
    experiment_name="my_data_noise_models_n2v",
    data_type="array",
    axes="SYXC",
    n_channels=2,
    patch_size=(64, 64),
    batch_size=64,
    num_epochs=500, 
    train_dataloader_params={"num_workers": get_num_workers()},
    val_dataloader_params={"num_workers": get_num_workers()},
)

print("N2V configuration generated.")


careamist = CAREamist(source=config, work_dir="/group/jug/aman/Datasets/Chicago/noise_models/")
careamist.train(train_source=train_data, val_minimum_split=5)


prediction = careamist.predict(train_data, tile_size=(256, 256))


do_crop = True

xfrom = yfrom = 0
xto = yto = -1
strcrop = ""
if do_crop:
    strcrop = " (crop)"
    yfrom = 200
    yto = 600
    xfrom = 800
    xto = 1200

_, ax = plt.subplots(2, 2, figsize=(10, 10))
ax[0][0].imshow(train_data[0, ..., 0][yfrom:yto, xfrom:xto])
ax[0][0].set_title("Input channel 1" + strcrop)
ax[0][1].imshow(prediction[0].squeeze()[0][yfrom:yto, xfrom:xto])
ax[0][1].set_title("Denoised channel 1" + strcrop)
ax[1][0].imshow(train_data[0, ..., 1][yfrom:yto, xfrom:xto])
ax[1][0].set_title("Input channel 2" + strcrop)
ax[1][1].imshow(prediction[0].squeeze()[1][yfrom:yto, xfrom:xto])
ax[1][1].set_title("Denoised channel 2" + strcrop)
plt.show()


for channel_idx in range(train_data.shape[-1]):
    print(f"Training noise model for channel {channel_idx}")
    channel_data = train_data[..., channel_idx]
    channel_prediction = np.concatenate(prediction)[:, channel_idx]
    noise_model_config = GaussianMixtureNMConfig(
        model_type="GaussianMixtureNoiseModel",
        min_signal=channel_data.min(),
        max_signal=channel_data.max(),
        n_coeff=3,
        n_gaussian=3,
    )
    noise_model = GaussianMixtureNoiseModel(noise_model_config)

    noise_model.fit(signal=channel_data, observation=channel_prediction, n_epochs=1000)
    noise_model.save(path=f"noise_models/", name=f"noise_model_Ch{channel_idx}")
    histogram = create_histogram(
        bins=100,
        min_val=channel_data.min(),
        max_val=channel_data.max(),
        signal=channel_data,
        observation=channel_prediction,
    )
    plot_probability_distribution(
        noise_model, signalBinIndex=50, histogram=histogram[0], channel=channel_idx
    )


