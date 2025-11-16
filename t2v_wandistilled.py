"""
FastWan2.2 Distilled Model - Text-to-Video Generation with Sparse Attention

USAGE:
  1. First run - Profile the model:
     - Set PROFILE_MODE = True (line ~100)
     - Run: python t2v_wandistilled.py
     - Check output file: FastWan2.2_Distilled_profile.md
     
  2. Update config:
     - Copy the recommended num_layers value from profile
     - Update config["attnprocessor_kwargs"]["num_layers"]
     
  3. Normal inference:
     - Set PROFILE_MODE = False
     - Run: python t2v_wandistilled.py
     - Video will be saved to zeroseed_vids/
"""

# import accelerate  # pylint: disable=unused-import
import diffusers  # pylint: disable=unused-import
import torch
import torch.nn.functional as F
# import transformers  # pylint: disable=unused-import
import t2v_wandistilled_processors  # Custom processor for WAN distilled (supports SDPA Top‑CDF)
import numpy as np
import random
import pathlib


AutoencoderKLWan = diffusers.AutoencoderKLWan
WanPipeline = diffusers.WanPipeline
from diffusers import UniPCMultistepScheduler


def set_wandistilled_attention(pipe, *args, **kwargs):
  """Sets custom attention processor for FastWan2.2 distilled model.
  
  Compatible with diffusers 0.35.1+
  
  Args:
    pipe: The WanDMDPipeline instance.
    **kwargs: Arguments to pass to the custom processor.
        - processor (str): "baseline", "cached", "bitmask", "topk", "2x"
        - num_layers (int): Number of transformer layers (auto-detected if not provided)
        - compute_cache_at (int): Step to compute cache (for cached/bitmask)
        - compress (bool): Compress cached masks
        - offload (bool): Offload cache to CPU
  """
  # Auto-count attention layers if not provided
  if "num_layers" not in kwargs:
    num_attn_layers = 0
    for name, module in pipe.transformer.named_modules():
      if 'attn1' in name:
        if hasattr(module, 'set_processor'):
          num_attn_layers += 1
    print(f"Auto-detected num_layers: {num_attn_layers}")
    kwargs["num_layers"] = num_attn_layers
  
  # Step 1: Create processor (no kwargs - required for diffusers 0.35.1+)
  attn_processor = t2v_wandistilled_processors.MyCustomProcessor()
  
  # Step 2: Configure it with sparse attention parameters
  attn_processor.configure(**kwargs)
  
  # Step 3: Install into all attention modules
  for name, module in pipe.transformer.named_modules():
    if 'attn1' in name:
        if hasattr(module, 'set_processor'):
            print('replacing attention processor of: ', name)
            module.set_processor(attn_processor)


def get_wandistilled_pipeline(model_path, *args, **kwargs):
  """Loads the FastWan2.2 pipeline with sparse attention.
  
  Args:
    model_path: Path to the model on HuggingFace or local path.
    
  Returns:
    WanPipeline instance.
  """
  print("Loading FastWan2.2 model from", model_path)
  
  # Load text encoder separately with low_cpu_mem_usage=False to avoid offload_state_dict issue
  from transformers import UMT5EncoderModel
  print("Loading text encoder...")
  text_encoder = UMT5EncoderModel.from_pretrained(
      model_path,
      subfolder="text_encoder",
      low_cpu_mem_usage=False,
      torch_dtype=torch.bfloat16
  )
  
  # Load VAE in float32 to avoid color artifacts (bf16 VAE causes posterization/banding)
  print("Loading VAE in float32...")
  vae = AutoencoderKLWan.from_pretrained(
      model_path,
      subfolder="vae",
      torch_dtype=torch.float32
  )
  
  # Load pipeline with pre-loaded text encoder and VAE
  pipe = WanPipeline.from_pretrained(
      model_path,
      text_encoder=text_encoder,
      vae=vae,
      torch_dtype=torch.bfloat16,  # Transformer stays in bf16
      # device_map="balanced"
  )
  
  # Multi-step scheduler for non-DMD sampling
  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  
  return pipe


def run_wandistilled(
    pipe,
    prompt,
    negative_prompt="",
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=25,  # Non-DMD model uses 12-20 steps
    *args,
    **kwargs
):
  """Runs the FastWan2.2 model to generate a video.

  Args:
    pipe: The WanPipeline instance.
    prompt: The text prompt.
    negative_prompt: The negative text prompt.
    height: The height of the video.
    width: The width of the video.
    num_frames: The number of frames in the video.
    guidance_scale: The guidance scale.
    num_inference_steps: The number of inference steps (default 20 for non-DMD).

  Returns:
    The generated video frames as a tensor.
  """
  output = pipe(
      prompt=prompt,
      negative_prompt=negative_prompt,
      height=height,
      width=width,
      num_frames=num_frames,
      guidance_scale=guidance_scale,
      num_inference_steps=num_inference_steps,
  ).frames[0]
  return output



if __name__ == '__main__':

  import argparse

  # ═══════════════════════════════════════════════════════════════════════════
  # PROFILING MODE: Set to True to profile model architecture, False to run inference
  # ═══════════════════════════════════════════════════════════════════════════
  PROFILE_MODE = False  # Set to True on first run to get num_layers, then set to False
  
  # Parse command line arguments
  prompt_base = "a horse bending down to drink water from a river"
  
  argparser = argparse.ArgumentParser(description="FastWan2.2 Distilled Model Inference")
  argparser.add_argument("--processor", type=str, default="bitmask", 
                         choices=["baseline", "cached", "bitmask", "sdpa_topcdf", "sdpa_cached"], # works
                         help="Attention processor type: 'baseline' (dense), 'cached' (index cache), 'bitmask' (bitmap cache), 'sdpa_topcdf' (Top‑CDF SDPA), 'sdpa_cached' (Naive cache SDPA)")
  argparser.add_argument("--prompt", type=str, default=prompt_base, help="Text prompt for video generation")
  argparser.add_argument("--filepath", type=str, default=None, help="Output video file path")
  args = argparser.parse_args()
  processor_type = args.processor
  prompt = args.prompt
  filepath = args.filepath

  # Configure attention processor based on user choice
  if processor_type == "baseline":
      print("Using BASELINE processor (standard PyTorch attention)")
      
      attnprocessor_kwargs = {
          "processor": "baseline",
          # num_layers will be auto-detected
      }
      model_name = "wandistilled_5b_480x832x81_baseline"
      
  elif processor_type == "cached":
      print("Using CACHED processor (sparse attention with index cache)")
      
      attnprocessor_kwargs = {
          "processor": "cached",
          # num_layers will be auto-detected
          "compute_cache_at": [0],  # Compute cache at step 0 (only once for distilled model)
          "compress": True,  # Compress cache to save memory using nvcomp
          "offload": False,  # Keep cache on GPU for speed
          "thresh": 0.5/8190,  # Threshold: 0.5/seq_len (seq=8190 for 480x832x81)
          "blocksz": 128,  # Block size for mask generation
      }
      model_name = "wandistilled_5b_480x832x81_cached"
      
  else:  # bitmask
      print("Using BITMASK processor (sparse attention with bitmap cache)")
      
      attnprocessor_kwargs = {
          "processor": "bitmask",
          # num_layers will be auto-detected
          "compute_cache_at": [0],  # Compute cache at step 0 (only once for distilled model)
          "compress": True,  # Compress cached masks to save memory
          "offload": False,  # Keep cache on GPU for speed
          "thresh": 0.5/8190,  # Threshold: 0.5/seq_len (seq=8190 for 480x832x81)
          "blocksz": 128,  # Block size for mask generation
      }
      model_name = "wandistilled_5b_480x832x81_bitmask"
  
  if processor_type == "sdpa_topcdf":
      print("Using SDPA Top-CDF processor (pure PyTorch mask injected into SDPA)")
      attnprocessor_kwargs = {
          "processor": "sdpa_topcdf",
          # num_layers will be auto-detected
          "blocksz": 128,
          # Global thresholds
          "tau": 0.95,
          "gamma_q": 0.5,
          "gamma_k": 0.5,
      }
      model_name = "wandistilled_5b_480x832x81_sdpa_topcdf"
  
  elif processor_type == "sdpa_cached":
      print("Using SDPA Naive Cache processor (threshold-based, pure PyTorch)")
      attnprocessor_kwargs = {
          "processor": "sdpa_cached",
          # num_layers will be auto-detected
          "blocksz": 128,
          "thresh": 0.5/8190,  # Threshold: 0.5/seq_len (seq=8190 for 480x832x81)
          "compute_cache_at": [0],  # Compute cache at step 0 (only once for distilled model)
      }
      model_name = "wandistilled_5b_480x832x81_sdpa_cached"

  # Build configuration
  config = {
      "model_name": model_name,
      "generated_vids_dir": "zeroseed_vids",
      "num_samples": 1,
      "init_fn": get_wandistilled_pipeline,
      "init_fn_kwargs": {
          "model_path": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",
      },
      "set_attnprocessor_fn": set_wandistilled_attention,
      "attnprocessor_kwargs": attnprocessor_kwargs,
      "run_fn": run_wandistilled,
      "run_fn_kwargs": {
          "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
          "height": 480,
          "width": 832,
          "num_frames": 81,
          "guidance_scale": 5.0,
          "num_inference_steps": 50, 
      },
      "fps": 24,  # FastWan2.2 runs at 24fps
  }

  output_dir_base = pathlib.Path(config["generated_vids_dir"]) / config["model_name"]
  # Include block size in filename when available
  _blk = None
  try:
    _blk = config.get("attnprocessor_kwargs", {}).get("blocksz", None)
  except Exception:
    _blk = None
  _fname = f"test_{_blk}.mp4" if _blk is not None else "test.mp4"
  filepath_base = output_dir_base / _fname
  if filepath is None:
    print(filepath)
    filepath = filepath_base
    output_dir_base.mkdir(parents=True, exist_ok=True)
  
  # Convert to absolute path and ensure parent directory exists
  filepath = pathlib.Path(filepath).resolve()
  filepath.parent.mkdir(parents=True, exist_ok=True)

  pipe = config['init_fn'](**config["init_fn_kwargs"])
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PROFILING: Run once to detect correct num_layers and model parameters
  # ═══════════════════════════════════════════════════════════════════════════
  if PROFILE_MODE:
    try:
      from profile_model_params import profile_pipeline_parameters
    except ImportError:
      print("ERROR: profile_model_params module not found!")
      print("Profiling mode requires the profile_model_params.py utility.")
      import sys
      sys.exit(1)
    
    print("\n" + "="*80)
    print("PROFILING MODE: Analyzing model architecture...")
    print("="*80 + "\n")
    
    profile_pipeline_parameters(
        pipe,
        output_file="FastWan2.2_Distilled_profile.md",
        model_name="FastWan2.2-TI2V-5B-FullAttn-Diffusers"
    )
    
    print("\n" + "="*80)
    print("Profiling complete! Check 'FastWan2.2_Distilled_profile.md'")
    print("Update config with the recommended num_layers value")
    print("Then set PROFILE_MODE = False and run again")
    print("="*80 + "\n")
    
    import sys
    sys.exit(0)  # Exit after profiling
  # ═══════════════════════════════════════════════════════════════════════════
  
  pipe.to('cuda')
  pipe.enable_model_cpu_offload()
  # pipe.transformer.compile(mode='reduce-overhead', dynamic=True)
  config["set_attnprocessor_fn"](pipe, **config["attnprocessor_kwargs"])
  # torch.manual_seed(0)
  # np.random.seed(0)
  # random.seed(0)
  frames = config["run_fn"](
    pipe,
    prompt,
    **config["run_fn_kwargs"]
  )
  diffusers.utils.export_to_video(frames, str(filepath), fps=config["fps"])




