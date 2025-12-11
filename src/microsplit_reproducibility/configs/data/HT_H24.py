from careamics.lvae_training.dataset import DatasetConfig, DataSplitType, DataType


class HT_H24Config(DatasetConfig):
    channel_idx_list: list[int]
    z_start: int
    z_stop: int


def get_data_configs(sliding_window_flag=False) -> tuple[HT_H24Config, HT_H24Config]:
    train_data_config = HT_H24Config(
        datasplit_type=DataSplitType.Train,
        image_size=(8, 128, 128),
        grid_size=(4, 64, 64),
        multiscale_lowres_count=1,
        data_type=DataType.HTH24Data,
        depth3D=8,
        mode_3D=True,
        poisson_noise_factor=-1,
        enable_gaussian_noise=False,
        synthetic_gaussian_scale=228,
        input_has_dependant_noise=True,
        use_one_mu_std=True,
        train_aug_rotate=True,
        target_separate_normalization=True,
        input_is_sum=False,
        channel_idx_list=[0, 1],
        z_start=25,
        z_stop=40,
        padding_kwargs={"mode": "reflect"},
        overlapping_padding_kwargs={"mode": "reflect"},
        sliding_window_flag=sliding_window_flag,
    )

    val_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Val,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )
    
    test_data_config = train_data_config.model_copy(
        update=dict(
            datasplit_type=DataSplitType.Test,
            allow_generation=False,  # No generation during validation
            enable_random_cropping=False,  # No random cropping on validation.
        )
    )

    return train_data_config, val_data_config, test_data_config