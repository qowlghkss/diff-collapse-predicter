import sys
import torch
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "diffusion"))
from diffusers import StableDiffusionPipeline
import logging
logging.basicConfig(level=logging.ERROR)

model_id = "ashawkey/mvdream-sd2.1-diffusers"
print("Loading pipe...")
try:
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None,
        trust_remote_code=True
    ).to("cuda")
except Exception as e:
    print(f"Failed to load: {e}")
    sys.exit(1)

# Monkey-patch config
kwargs = dict(pipe.unet.config)
if "in_channels" not in kwargs: kwargs["in_channels"] = 4
if "sample_size" not in kwargs: kwargs["sample_size"] = 32
if "time_cond_proj_dim" not in kwargs: kwargs["time_cond_proj_dim"] = None
pipe.unet.register_to_config(**kwargs)

original_forward = pipe.unet.forward
def wrapped_forward(*args, **kwargs):
    args_list = list(args)
    if len(args_list) > 1:
        t = args_list[1]
        if isinstance(t, (int, float)): args_list[1] = torch.tensor([t], device="cuda")
        elif isinstance(t, torch.Tensor) and t.ndim == 0: args_list[1] = t.reshape((1,))
    elif 'timestep' in kwargs:
        t = kwargs['timestep']
        if isinstance(t, (int, float)): kwargs['timestep'] = torch.tensor([t], device="cuda")
        elif isinstance(t, torch.Tensor) and t.ndim == 0: kwargs['timestep'] = t.reshape((1,))
    if 'encoder_hidden_states' in kwargs:
        kwargs['context'] = kwargs.pop('encoder_hidden_states')
    
    out = original_forward(*args_list, **kwargs)
    noise_pred = out.sample if hasattr(out, "sample") else out[0]
    print(f"UNET FORWARD: input={args_list[0].shape}, output={noise_pred.shape}")
    return out

pipe.unet.forward = wrapped_forward

original_step = pipe.scheduler.step
def wrapped_step(model_output, timestep, sample, **kwargs):
    print("INSIDE WRAPPED STEP:")
    try:
        if type(timestep) == torch.Tensor:
            alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep.to(torch.int32).item()]
        else:
            alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep]
            
        beta_prod_t = 1 - alpha_prod_t
        print(f"sample: type={type(sample)}, shape={sample.shape}")
        print(f"model_output: type={type(model_output)}, shape={model_output.shape}")
        print(f"alpha_prod_t: type={type(alpha_prod_t)}, shape={alpha_prod_t.shape}")
        print(f"beta_prod_t: type={type(beta_prod_t)}, shape={beta_prod_t.shape}")
        
        # Test the math
        b0 = beta_prod_t ** 0.5
        m0 = model_output
        print(f"b0 shape = {b0.shape}, m0 shape = {m0.shape}")
        term = b0 * m0
        print(f"term shape = {term.shape}")
        div = sub / (alpha_prod_t ** 0.5)
        print("Math succeeded in wrapper!")
    except Exception as e:
        print("Math failed in wrapper! Error:", str(e))
        
    try:
        return original_step(model_output, timestep, sample, **kwargs)
    except Exception as e:
        print("!!! ERROR IN ORIGINAL STEP !!")
        print(f"Exception: {e}")
        # Let's inspect the shapes passed in one more time
        print(f"model_output: {model_output.shape}")
        print(f"timestep: {timestep}")
        print(f"sample: {sample.shape}")
        sys.exit(1)

pipe.scheduler.step = wrapped_step

# test 1 step
prompt = "A 3D object of a mechanical gear system"
generator = torch.Generator("cuda").manual_seed(42)

print("Starting generation...")
try:
    res = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=9.0, generator=generator, num_images_per_prompt=4)
    print("Generation Success!")
except Exception as e:
    import traceback
    traceback.print_exc()

