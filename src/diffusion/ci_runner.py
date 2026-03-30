"""
Phase 1: CI Prediction Test (Baseline Only)
===========================================
Runs stable-diffusion-v1-5 with DDIM (50 steps) for seeds 0–99.
At every step computes:
  - thin_pixel_count  (Canny edges on latent-proxy image, dilated)
  - CI                (Kendall-τ_thin − Kendall-τ_fat, sliding window=10)
Saves per-seed .npy trajectories plus a summary .npz.
"""

import argparse
import os
import sys
import warnings

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, DiffusionPipeline
from scipy.stats import kendalltau

# ─────────────────────────── Configuration ───────────────────────────────────
MODEL_ID       = "runwayml/stable-diffusion-v1-5"
PROMPT         = (
    "A highly detailed macro photograph of a bare winter tree with many thin branches, "
    "intricate twig structures, natural lighting, ultra high resolution, sharp focus, "
    "realistic texture, neutral background"
)
NEGATIVE_PROMPT = (
    "unreal engine, hyper realistic, low quality, bad quality, blurry, pixelated, "
    "noisy, low resolution, cropping, out of frame, out of focus, duplicate, error, "
    "bad anatomy, deformed"
)
NUM_STEPS      = 50
GUIDANCE_SCALE = 7.5   # overridden by --guidance at runtime
WINDOW_SIZE    = 10       # sliding window for CI computation
CANNY_LO       = 50
CANNY_HI       = 150
DILATE_KERNEL  = 3
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ─────────────────────────── Helpers ─────────────────────────────────────────

def latent_to_proxy_image(latent: torch.Tensor, size: int = 256) -> np.ndarray:
    """
    Cheap proxy: upscale the first latent channel to `size × size`,
    normalise to [0,255] uint8.  Used for thin_pixel_count at steps < 49.
    latent: [1, 4, H, W] fp16/fp32 on DEVICE.
    """
    ch = latent[0, :3].float()                              # [3, H, W]
    ch = F.interpolate(ch.unsqueeze(0), size=(size, size),
                       mode="bilinear", align_corners=False)[0]  # [3, sz, sz]
    lo, hi = ch.min(), ch.max()
    if hi > lo:
        ch = (ch - lo) / (hi - lo)
    else:
        ch = ch * 0
    img = (ch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


def compute_thin_pixel_count(rgb_img: np.ndarray) -> int:
    """Canny edges on gray, dilate 3×3, count non-zero."""
    gray  = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, CANNY_LO, CANNY_HI)
    kernel = np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    return int(np.count_nonzero(dilated))


def compute_ci(noise_history: list, thin_mask: np.ndarray) -> float:
    """
    Sliding-window CI = Kendall-τ_thin − Kendall-τ_fat.
    noise_history: last WINDOW_SIZE frames, each [H,W] fp32.
    thin_mask: bool [H,W] at the latent resolution (64×64).
    Returns NaN if not enough data or too few pixels.
    """
    window = np.array(noise_history[-WINDOW_SIZE:])   # [K, 64, 64]
    
    if window.ndim != 3:
        print(f"DEBUG: window.shape={window.shape}, len(noise_history)={len(noise_history)}")
        if len(noise_history) > 0:
            print(f"DEBUG: first element shape: {noise_history[0].shape}")
        return np.nan
        
    K, H, W = window.shape

    # Resize thin_mask to latent resolution if needed
    if thin_mask.shape != (H, W):
        thin_mask = cv2.resize(
            thin_mask.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
    fat_mask = ~thin_mask

    thin_pix = window[:, thin_mask].T   # [N_thin, K]
    fat_pix  = window[:, fat_mask ].T   # [N_fat,  K]

    if thin_pix.shape[0] < 2 or fat_pix.shape[0] < 2:
        return np.nan

    # Sub-sample for speed
    rng = np.random.default_rng(seed=0)
    if thin_pix.shape[0] > 500:
        idx = rng.choice(thin_pix.shape[0], 500, replace=False)
        thin_pix = thin_pix[idx]
    if fat_pix.shape[0] > 500:
        idx = rng.choice(fat_pix.shape[0], 500, replace=False)
        fat_pix = fat_pix[idx]

    t_axis = np.arange(K, 0, -1)   # descending time-step order

    def mean_tau(pixels):
        taus = [kendalltau(t_axis, p).statistic for p in pixels
                if not np.all(p == p[0])]
        return float(np.mean(taus)) if taus else np.nan

    tau_thin = mean_tau(thin_pix)
    tau_fat  = mean_tau(fat_pix)

    if np.isnan(tau_thin) or np.isnan(tau_fat):
        return np.nan
    return tau_thin - tau_fat


# ─────────────────────────── Main Runner ─────────────────────────────────────

class CIRunner:
    def __init__(self, model_id=MODEL_ID):
        self.model_id = model_id
        print(f"Loading {model_id} …")
        if "mvdream" in model_id.lower():
            try:
                import sys
                project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                lgm_path = os.path.join(project_root, "LGM")
                if lgm_path not in sys.path:
                    sys.path.append(lgm_path)
                from mvdream.pipeline_mvdream import MVDreamPipeline
                self.pipe = MVDreamPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    trust_remote_code=True
                ).to(DEVICE)
            except ImportError:
                print("LGM module not found. Falling back to StableDiffusionPipeline for MVDream...")
                from diffusers import StableDiffusionPipeline
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    trust_remote_code=True
                ).to(DEVICE)
                
                # Monkey-patch missing config attributes via register_to_config
                kwargs = dict(self.pipe.unet.config)
                if "in_channels" not in kwargs: kwargs["in_channels"] = 4
                if "sample_size" not in kwargs: kwargs["sample_size"] = 32
                if "time_cond_proj_dim" not in kwargs: kwargs["time_cond_proj_dim"] = None
                self.pipe.unet.register_to_config(**kwargs)
        else:
            self.pipe = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                trust_remote_code=True
            ).to(DEVICE)
        self.pipe.scheduler = DDIMScheduler.from_config(
            self.pipe.scheduler.config,
            # ensure deterministic ordering
        )
        self.pipe.set_progress_bar_config(disable=True)
        self.out_name = "summary"
        self.guidance_scale = GUIDANCE_SCALE
        self.intervention_step  = None   # int or None
        self.intervention_boost = 0.0    # fractional latent boost
        print("Pipeline ready.")

    def run_seed(self, seed: int):
        """Run one seed, return (ci_traj, thin_traj, final_img_pil)."""
        generator = torch.Generator(DEVICE).manual_seed(seed)

        # Per-step accumulators
        noise_history: list[np.ndarray] = []   # [H,W] fp32 noise magnitude
        ci_traj   = np.full(NUM_STEPS, np.nan, dtype=np.float32)
        thin_traj = np.zeros(NUM_STEPS, dtype=np.int32)
        final_img_pil = [None]

        # ── UNet hook: capture noise magnitude ──────────────────────────────
        original_forward = self.pipe.unet.forward

        def wrapped_forward(*args, **kwargs):
            args_list = list(args)
            # Ensure timestep is 1D tensor for MVDream UNet compatibility
            if len(args_list) > 1:
                t = args_list[1]
                if isinstance(t, (int, float)):
                    args_list[1] = torch.tensor([t], device=DEVICE)
                elif isinstance(t, torch.Tensor) and t.ndim == 0:
                    args_list[1] = t.reshape((1,))
            elif 'timestep' in kwargs:
                t = kwargs['timestep']
                if isinstance(t, (int, float)):
                    kwargs['timestep'] = torch.tensor([t], device=DEVICE)
                elif isinstance(t, torch.Tensor) and t.ndim == 0:
                    kwargs['timestep'] = t.reshape((1,))
            
            # MVDream UNet uses 'context' instead of 'encoder_hidden_states'
            if 'encoder_hidden_states' in kwargs:
                kwargs['context'] = kwargs.pop('encoder_hidden_states')

            out = original_forward(*args_list, **kwargs)
            
            # If original_forward returned a raw tensor, Diffusers SD pipeline __call__ 
            # will do out[0], dropping the batch dimension and causing a crash pipeline-side. 
            # Wrap it in a tuple!
            if isinstance(out, torch.Tensor):
                out = (out,)
                
            noise_pred = out.sample if hasattr(out, "sample") else out[0]
            
            # CFG combine: batch is [uncond, text]
            n_uc, n_tx = noise_pred.detach().float().chunk(2)
            combined   = n_uc + self.guidance_scale * (n_tx - n_uc)
            mag = torch.norm(combined, dim=1)[0].cpu().numpy()  # [H,W]
            noise_history.append(mag)
            return out

        self.pipe.unet.forward = wrapped_forward

        # ── Callback: thin_pixel_count + CI ─────────────────────────────────
        def callback_fn(pipe_cb, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]

            # ---- proxy image for thin_pixel_count ----
            proxy_img = latent_to_proxy_image(latents, size=256)
            tpc = compute_thin_pixel_count(proxy_img)
            thin_traj[step] = tpc

            # thin mask at latent resolution (64×64) for CI
            gray   = cv2.cvtColor(proxy_img, cv2.COLOR_RGB2GRAY)
            edges  = cv2.Canny(gray, CANNY_LO, CANNY_HI)
            kernel = np.ones((DILATE_KERNEL, DILATE_KERNEL), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)
            # downscale mask to latent h/w
            lat_h, lat_w = latents.shape[2], latents.shape[3]
            thin_mask_lat = cv2.resize(
                (dilated > 0).astype(np.uint8), (lat_w, lat_h),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)

            # ---- CI ----
            if len(noise_history) >= WINDOW_SIZE:
                ci = compute_ci(noise_history, thin_mask_lat)
                ci_traj[step] = ci

            # ---- Final decode ----
            if step == NUM_STEPS - 1:
                with torch.no_grad():
                    ls = latents / pipe_cb.vae.config.scaling_factor
                    img = pipe_cb.vae.decode(ls, return_dict=False)[0]
                    img = (img / 2 + 0.5).clamp(0, 1)
                    img = img.cpu().permute(0, 2, 3, 1).float().numpy()
                    img_u8 = (img * 255).round().astype("uint8")
                    from PIL import Image
                    if img_u8.shape[0] > 1:
                        # MVDream batch size is 4, concatenate horizontally
                        imgs = [Image.fromarray(img_u8[i]) for i in range(img_u8.shape[0])]
                        total_width = sum(i.width for i in imgs)
                        max_height = max(i.height for i in imgs)
                        combined = Image.new("RGB", (total_width, max_height))
                        x_offset = 0
                        for i in imgs:
                            combined.paste(i, (x_offset, 0))
                            x_offset += i.width
                        final_img_pil[0] = combined
                    else:
                        final_img_pil[0] = Image.fromarray(img_u8[0])

            # ---- Intervention at specified step ----
            if (self.intervention_step is not None
                    and step == self.intervention_step
                    and self.intervention_boost != 0.0):
                # Amplify latents in the denoised direction:
                # boosts structure signal without altering noise field
                latents = latents * (1.0 + self.intervention_boost)
                callback_kwargs["latents"] = latents
                # mark that the intervention fired
                if not hasattr(self, "_intervention_fired"):
                    self._intervention_fired = True

            return {"latents": latents} if (self.intervention_step is not None
                                            and step == self.intervention_step
                                            and self.intervention_boost != 0.0) else {}

        kwargs = {}
        is_mvdream = getattr(self, "model_id", "") and "mvdream" in self.model_id.lower()
        if is_mvdream:
            kwargs["height"] = 256
            kwargs["width"] = 256
            
            def mvd_callback_wrapper(step_idx, t, latents):
                cb_kwargs = {"latents": latents}
                ret = callback_fn(self.pipe, step_idx, t, cb_kwargs)
                if ret and "latents" in ret:
                    latents.copy_(ret["latents"])
            
            kwargs["callback"] = mvd_callback_wrapper
            kwargs["callback_steps"] = 1
        else:
            kwargs["callback_on_step_end"] = callback_fn
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

        try:
            pipe_kwargs = dict(
                prompt=PROMPT,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=NUM_STEPS,
                guidance_scale=self.guidance_scale,
                generator=generator,
                **kwargs
            )
            if is_mvdream:
                pipe_kwargs["num_images_per_prompt"] = 4

            res = self.pipe(**pipe_kwargs)
        finally:
            self.pipe.unet.forward = original_forward

        return ci_traj, thin_traj, final_img_pil[0]

    def run_all(self, seed_start: int, seed_end: int):
        seeds = list(range(seed_start, seed_end + 1))
        n = len(seeds)
        all_ci   = np.full((n, NUM_STEPS), np.nan, dtype=np.float32)
        all_thin = np.zeros((n, NUM_STEPS), dtype=np.int32)
        final_tpc = np.zeros(n, dtype=np.int32)

        for i, seed in enumerate(seeds):
            print(f"\n{'='*60}")
            print(f"Seed {seed}  ({i+1}/{n})")
            print(f"{'='*60}")

            ci_traj, thin_traj, img = self.run_seed(seed)

            all_ci[i]    = ci_traj
            all_thin[i]  = thin_traj
            final_tpc[i] = thin_traj[-1]

            # Save per-seed
            out_dir = getattr(self, "out_dir", DATA_DIR)
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, f"ci_traj_{seed}.npy"),   ci_traj)
            np.save(os.path.join(out_dir, f"thin_traj_{seed}.npy"), thin_traj)
            np.save(os.path.join(out_dir, f"final_tpc_{seed}.npy"), np.array([thin_traj[-1]]))

            if img is not None:
                img.save(os.path.join(IMAGES_DIR, f"final_{seed}.png"))

            print(f"  CI_max={np.nanmax(ci_traj):.3f}  "
                  f"final_tpc={thin_traj[-1]}")

        # Save summary
        out_dir = getattr(self, "out_dir", DATA_DIR)
        os.makedirs(out_dir, exist_ok=True)
        out_name = getattr(self, "out_name", "summary")
        out_path = os.path.join(out_dir, f"{out_name}.npz")
        np.savez_compressed(
            out_path,
            seeds      = np.array(seeds),
            ci_trajs   = all_ci,
            thin_trajs = all_thin,
            final_tpc  = final_tpc,
        )
        print(f"\nDone. Summary saved to {out_path}")


# ─────────────────────────── Entry Point ─────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs=2, type=int, default=None,
                        metavar=("START", "END"),
                        help="Inclusive seed range")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run a single seed")
    parser.add_argument("--mode", type=str, default="control")
    parser.add_argument("--freeze-threshold", action="store_true")
    parser.add_argument("--no-refit", action="store_true")
    parser.add_argument("--strict-integrity", action="store_true")
    parser.add_argument("--out-name", type=str, default="summary",
                        help="Stem for summary .npz file (default: summary)")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="CFG guidance scale (default: 7.5)")
    parser.add_argument("--intervention-step", type=int, default=None,
                        help="Step index at which to apply intervention (default: None)")
    parser.add_argument("--intervention-boost", type=float, default=0.0,
                        help="Fractional latent boost at intervention step (default: 0.0)")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Output directory for data files")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Model ID to load")
    args = parser.parse_args()
    runner = CIRunner(model_id=args.model_name)
    if args.out_dir:
        runner.out_dir = args.out_dir
    runner.out_name          = args.out_name
    runner.guidance_scale    = args.guidance
    runner.intervention_step  = args.intervention_step
    runner.intervention_boost = args.intervention_boost
    print(f"[CONFIG] guidance_scale      = {runner.guidance_scale}")
    print(f"[CONFIG] out_name            = {runner.out_name}")
    print(f"[CONFIG] intervention_step   = {runner.intervention_step}")
    print(f"[CONFIG] intervention_boost  = {runner.intervention_boost}")
    if args.seed is not None:
        print(f"[CONFIG] seed                = {args.seed}")
        runner.run_all(args.seed, args.seed)
    else:
        seeds = args.seeds if args.seeds else [0, 99]
        print(f"[CONFIG] seeds               = {seeds[0]}–{seeds[1]}")
        runner.run_all(seeds[0], seeds[1])
