import traceback
import numpy as np
import torch
from PIL import Image, ImageFilter


def dilate_image(img):
    radius = 20
    pil_image = Image.fromarray(np.asarray(img))
    size = radius * 2 + 1
    return np.array(pil_image.filter(ImageFilter.MaxFilter(size)))

def compute_variance_schedule(
    steps: int,
    scheduler_type: str = "cosine",
    beta_1: float = 1e-4,
    beta_t: float = 0.2,
    s: float = 0.008,
    max_beta: float = 0.999,
):
    """
    Parameters
    ----------
    steps : int
        Number of steps
    scheduler_type : str, default="cosine"
        Type of scheduler
    beta_1 : float, default=1e-4
        Beta at time 1
    beta_t : float, default=0.2 
        Beta at time t
    s : float, default=0.008    
    max_beta : float, default=0.999

    Notes
    -----
    1. Refer to Improved DDPM paper (https://arxiv.org/pdf/2102.09672)
    """
    if scheduler_type == "linear":
        beta = torch.linspace(beta_1, beta_t, steps)
        alpha = 1 - beta
        alpha_bar = torch.cumprod(alpha, 0)
        return alpha_bar, beta

    if scheduler_type == "cosine":
        times = torch.arange(0, steps + 1, 1)
        f = torch.cos((times / steps + s) / (1 + s) * torch.pi / 2)
        f = torch.pow(f, 2)
        alpha_bar = f / f[0]
        alpha_bar_shifted = torch.roll(alpha_bar, 1)
        beta = 1 - alpha_bar / alpha_bar_shifted
        beta = torch.clip(beta, 1 - alpha_bar / alpha_bar_shifted, torch.tensor(max_beta))
        return alpha_bar, beta

    raise ValueError(
        f"Unknown scheduler type {scheduler_type}. Supported schedulers are linear and cosine."
    )

def compute_timesteps(num_steps: int, train_steps: int = 50):
    """
    Build the decreasing timestep sequence walked during sampling.

    The variance schedule always spans `train_steps` (it must match training), but
    sampling may skip timesteps. The returned sequence always starts at
    `train_steps` and ends at 0, is strictly decreasing, and yields exactly
    `num_steps` transitions (one per consecutive pair).

    E.g. num_steps=10, train_steps=50 -> [50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]
    """
    num_steps = int(max(1, min(num_steps, train_steps)))
    ts = np.rint(np.linspace(train_steps, 0, num_steps + 1)).astype(int).tolist()
    # Defensive: keep it strictly decreasing in case rounding collapses neighbours
    deduped = [ts[0]]
    for t in ts[1:]:
        if t < deduped[-1]:
            deduped.append(t)
    if deduped[-1] != 0:
        deduped.append(0)
    return deduped


def diffusion_process(gt_pred_image: Image.Image, decoder, scribble_class: int = 0, num_steps: int = 50, progress_callback=None) -> Image.Image:
    """
    Processes an image using a diffusion loop.
    - Resizes image from 256x256 to 128x128.
    - Separates blue (ground truth) and green (prediction) channels.
    - Runs a diffusion process for `steps` steps, adding noise and decoding.
    - Returns a new image with the result in the red channel, and original green and blue channels.
    
    Args:
        gt_pred_image: Input PIL Image
        decoder: Decoder function for the diffusion process
        scribble_class: Scribble class to pass to the model (0=Background, 1=Foreground)
        num_steps: Number of sampling steps (10-50). Fewer steps skip timesteps,
            walking t from 50 down to 0 in evenly spaced strides.
        progress_callback: Optional callback function(current_step, total_steps) to report progress
    """
    try:
        device = "cpu"

        print("Starting diffusion process...")

        # NOTE: these need to be the same as during training!
        train_steps = 50
        scheduler_args = {
            "beta_1": 1e-4,
            "beta_t": 0.2,
            "s": 0.008,
            "max_beta": 0.999
        }
        scheduler_type = "cosine"
        input_size = 128

        # Compute variance schedule
        var_scheduler = compute_variance_schedule(train_steps, scheduler_type, **scheduler_args)
        alpha_bar = var_scheduler[0]
        beta = var_scheduler[1]

        # Convert PIL Image to NumPy array and resize
        gt_pred_np = np.array(
            gt_pred_image.resize((input_size, input_size), Image.Resampling.NEAREST)
        )

        # make sure in each channel the value is either 255 or 0.
        pred_channel = gt_pred_np[:, :, 0]
        gt_channel = gt_pred_np[:, :, 1]
        pred_channel[pred_channel > 0] = 255
        gt_channel[gt_channel > 0] = 255
        gt_pred_np[:, :, 0] = pred_channel
        gt_pred_np[:, :, 1] = gt_channel
        gt_pred_np[:, :, 2] = 0

        # RGB images with R=Pred, G=GT, B=Scribble
        pred  = torch.tensor(gt_pred_np[:, :, 0], dtype=torch.uint8)
        gt  = torch.tensor(gt_pred_np[:, :, 1], dtype=torch.uint8)
        pred_r = pred
        gt_r = gt
        pred = pred.numpy()
        gt = gt.numpy()

        # Add batch and channel dimensions to pred_dilated and gt_dilated
        pred = torch.tensor(pred, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
        gt = torch.tensor(gt, dtype=torch.uint8).unsqueeze(0).unsqueeze(0)

        # Convert dilated images back to torch tensors and add channel dimension
        conditionGt = gt.float().to(device) / 255.0
        conditionPred = pred.float().to(device) / 255.0 
        conditionsGt = conditionGt.numpy()
        conditionsPred = conditionPred.numpy()
        
        # Initialize noise, as we only handle on1y 1 image and 1 channel
        batch_size = 1
        # torch.manual_seed(torch.randint(0, 100000, (1,)).item())
        xt = torch.randn(batch_size, 1, input_size, input_size, device=device)
        print(f"Initialized noise xt. {xt.sum():.5f}")
        # Timesteps walked during sampling: always 50 -> 0, skipping steps when
        # num_steps < train_steps. Each consecutive pair is one denoising step.
        timesteps = compute_timesteps(num_steps, train_steps)
        total_steps = len(timesteps) - 1
        min_alpha = 1 - scheduler_args["max_beta"]
        print(f"Sampling timesteps ({total_steps} steps): {timesteps}")

        # Diffusion loop: do backward diffusion (by predicting noise t0 be removed)
        for idx, (t, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            print(f"Diffusion step {idx + 1}/{total_steps} (t={t} -> {t_prev})")

            # Report progress if callback provided
            if progress_callback:
                progress_callback(idx + 1, total_steps)

            z = torch.randn_like(xt, device=device) if t_prev > 0 else 0
            t_tensor = torch.tensor([t], device=device)

#            noise = decoder(xt.numpy(), t_tensor.numpy(), conditions, scribble_class)[0]
            noise = decoder(xt.numpy(), t_tensor.numpy(), conditionsGt, conditionsPred, scribble_class)[0]

            # Effective alpha over the (possibly skipped) interval t -> t_prev.
            # For t_prev == t - 1 this reduces to 1 - beta[t]; the clamp mirrors the
            # max_beta clipping in the variance schedule (alpha_bar[train_steps] is
            # effectively 0, so the first ratio would otherwise explode).
            alpha_bar_ = alpha_bar[t]
            alpha = torch.clamp(alpha_bar_ / alpha_bar[t_prev], min=min_alpha)
            sigma = torch.sqrt(1 - alpha)  # DDPM: sigma_t^2 = beta_t

            xt = (
                (1/ torch.sqrt(alpha))
                * ((xt - ((1 - alpha) / torch.sqrt(1 - alpha_bar_) * noise[0])))
            )

            if t_prev > 0:
                xt += (sigma * z)

        print("Completed diffusion loop.")

        # Binarise at 0.5, matching the training-time evaluation threshold
        xt_clipped = (torch.clamp(xt, 0.0, 1.0) > 0.5).to(torch.float32)
        # return an image with the result in the blue channel and using the gt_pred_image green and red channels    
        result_img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        result_img[:, :, 0] = pred_r.numpy()  # Red (prediction)
        result_img[:, :, 1] = gt_r.numpy()    # Green (ground truth)
        result_img[:, :, 2] = (xt_clipped.squeeze(0).squeeze(0).numpy() * 255).astype(np.uint8)  # Blue (scribble)
       
        print("Prepared result image.")

        print(result_img.shape)
        print("Diffusion process completed successfully.")

        return Image.fromarray(result_img, mode='RGB')
    except Exception as e:
        traceback.print_exc()
        print(f"Error during diffusion process: {e}")
        raise  # re-raise so callers can handle and display the error
