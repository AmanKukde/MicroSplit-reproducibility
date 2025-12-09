import os
from ._base import SplittingParameters

def _get_nm_paths(
    nm_path: str, 
    channel_idx_list: list[int],
) -> list[str]:
    nm_paths = []
    for channel_idx in channel_idx_list:
        fname = f"noise_model_Ch{channel_idx}.npz"
        nm_paths.append(os.path.join(nm_path,fname))
    return nm_paths


def get_microsplit_parameters(
    algorithm: str,
    img_size: tuple[int, int],
    noise_model_path: str,
    target_channels: int = 2,
    multiscale_count: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    lr_scheduler_patience: int = 10,
    earlystop_patience: int = 200,
    num_epochs: int = 50,
    num_workers: int = 4,
    mmse_count: int = 64,
    grid_size: int = 32,
) -> dict:
    nm_paths = _get_nm_paths(noise_model_path, channel_idx_list=list(range(target_channels)))
    return SplittingParameters(
        algorithm=algorithm,
        img_size=img_size,
        target_channels=target_channels,
        multiscale_count=multiscale_count,
        predict_logvar="pixelwise",
        loss_type="denoisplit_musplit" if algorithm == "denoisplit" else "musplit",
        kl_type="kl_restricted",
        batch_size=batch_size,
        lr=lr,
        num_epochs=num_epochs,
        num_workers=num_workers,
        lr_scheduler_patience=lr_scheduler_patience,
        earlystop_patience=earlystop_patience,
        mmse_count=mmse_count,
        grid_size=grid_size,
        nm_paths=nm_paths
    ).model_dump()


def get_eval_params() -> dict:
    raise NotImplementedError("Evaluation parameters not implemented for cusstom dataset.")
    return SplittingParameters().model_dump()
