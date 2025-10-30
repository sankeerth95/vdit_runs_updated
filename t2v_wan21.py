# import accelerate  # pylint: disable=unused-import
import diffusers  # pylint: disable=unused-import
import torch
import torch.nn.functional as F
# import transformers  # pylint: disable=unused-import
import t2v_wan21_processors
import numpy as np
import random
import pathlib


AutoencoderKLWan = diffusers.AutoencoderKLWan
WanPipeline = diffusers.WanPipeline


def set_wan21_attention(pipe, *args, **kwargs):
  attn_processor = t2v_wan21_processors.MyCustomProcessor(**kwargs)
  for name, module in pipe.transformer.named_modules():
    if 'attn1' in name:
        if hasattr(module, 'set_processor'):
            print('replacing attention processor of: ', name)
            module.set_processor(attn_processor)


def get_wan21_pipeline(model_path, *args, **kwargs):
  print("Loading WAN2.1 model from", model_path)
  vae = AutoencoderKLWan.from_pretrained(
      model_path, subfolder="vae", torch_dtype=torch.float32
  )
  pipe = WanPipeline.from_pretrained(
      model_path, vae=vae, torch_dtype=torch.bfloat16,
      # device_map="balanced"
  )
  return pipe


def run_wan21(
    pipe,
    prompt,
    negative_prompt="",
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=50,
    *args,
    **kwargs
):
  """Runs the WAN2.1 model to generate a video.

  Args:
    pipe: The WAN2.1 pipeline.
    prompt: The text prompt.
    negative_prompt: The negative text prompt.
    height: The height of the video.
    width: The width of the video.
    num_frames: The number of frames in the video.
    guidance_scale: The guidance scale.
    num_inference_steps: The number of inference steps.

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

  import distributedrunconfig
  import argparse

  # ═══════════════════════════════════════════════════════════════════════════
  # PROFILING MODE: Set to True to profile model architecture, False to run inference
  # ═══════════════════════════════════════════════════════════════════════════
  PROFILE_MODE = False  # Set to True on first run to get num_layers and num_heads

  # prompt = "A beautiful coastal beach in spring, waves lapping on sand by Vincent van Gogh"
  # prompt = "An oil painting of a couple in formal evening wear going home get caught in a heavy downpour with umbrellas"
  # config = distributedrunconfig.get_wan21_14b_720x1280x81_baseline_config()
  # config = distributedrunconfig.get_wan21_14b_720x1280x81_bitmaskcached_config()
  # config = distributedrunconfig.get_wan21_1_3b_720x1280x81_baseline_config()
  # config = distributedrunconfig.get_wan21_1_3b_720x1280x81_bitmaskcached_config()


  # config = distributedrunconfig.get_wan21_14b_480x832x81_bitmaskcached_config()
  # config = distributedrunconfig.get_wan21_1_3b_720x1280x81_bitmaskcached_config()
  # config = distributedrunconfig.get_wan21_14b_720x1280x81_baseline_config()
  # config = distributedrunconfig.get_wan21_14b_720x1280x81_bitmaskcached_config()
  # config = distributedrunconfig.get_wan21_1_3b_720x1280x81_baseline_config()
  # config = distributedrunconfig.get_wan21_1_3b_480x832x81_baseline_config()
  # config = distributedrunconfig.get_wan21_1_3b_480x832x81_topcdf_config()
  config = distributedrunconfig.get_wan21_1_3b_480x832x81_sdpa_topcdf16_config()
  # config = distributedrunconfig.get_wan21_1_3b_720x1280x81_topcdf_config()
  # config = distributedrunconfig.get_wan21_14b_480x832x81_topcdf_config()
  # config = distributedrunconfig.get_wan21_14b_720x1280x81_topcdf_config()
  # config = distributedrunconfig.get_wan21_14b_720x1280x81_2x_config()
  # config = distributedrunconfig.get_wan21_1_3b_720x1280x81_2x_config()
  # config = distributedrunconfig.get_wan21_1_3b_480x832x81_2x_config()

  # config = distributedrunconfig.get_wan21_14b_720x1280x81_bitmaskcached_config()
  # config = distributedrunconfig.get_wan21_14b_720x1280x81_bitmaskcached_config()

  prompt_base = "a horse bending down to drink water from a river"
  output_dir_base = pathlib.Path(config["generated_vids_dir"]) / config["model_name"]
  filepath_base = output_dir_base / "test.mp4"

  argparser = argparse.ArgumentParser()
  argparser.add_argument("--prompt", type=str, default=prompt_base, )
  argparser.add_argument("--filepath", type=str, default=None, )
  argparser.add_argument("--profile", action="store_true", help="Profile model architecture")
  args = argparser.parse_args()
  prompt = args.prompt
  filepath = args.filepath
  if args.profile:
    PROFILE_MODE = True
  
  if filepath is None:
    print(filepath)
    filepath = filepath_base
    output_dir_base.mkdir(parents=True, exist_ok=True)

  pipe = config['init_fn'](**config["init_fn_kwargs"])
  
  # ═══════════════════════════════════════════════════════════════════════════
  # PROFILING: Run once to detect correct num_layers and num_heads
  # ═══════════════════════════════════════════════════════════════════════════
  if PROFILE_MODE:
    print("\n" + "="*80)
    print("PROFILING MODE: Analyzing model architecture...")
    print("="*80 + "\n")
    
    # Count attention layers and detect number of heads
    num_layers = 0
    num_heads = None
    sample_attn_module = None
    
    for name, module in pipe.transformer.named_modules():
        if 'attn1' in name and hasattr(module, 'set_processor'):
            num_layers += 1
            if sample_attn_module is None:
                sample_attn_module = module
    
    # Try to detect number of heads from the first attention module
    if sample_attn_module is not None:
        if hasattr(sample_attn_module, 'heads'):
            num_heads = sample_attn_module.heads
        elif hasattr(sample_attn_module, 'num_heads'):
            num_heads = sample_attn_module.num_heads
        elif hasattr(sample_attn_module.processor, 'num_heads'):
            num_heads = sample_attn_module.processor.num_heads
        # Try to infer from to_q weight shape
        elif hasattr(sample_attn_module, 'to_q'):
            hidden_size = sample_attn_module.to_q.out_features
            head_dim = getattr(sample_attn_module, 'head_dim', 128)  # default 128
            num_heads = hidden_size // head_dim
    
    # Write profile to markdown
    model_name = config.get("model_name", "WAN2.1")
    output_file = f"{model_name}_profile.md"
    with open(output_file, 'w') as f:
        f.write(f"# {model_name} Model Profile\n\n")
        f.write(f"**Model Path**: `{config['init_fn_kwargs'].get('model_path', 'Unknown')}`\n\n")
        f.write("## Architecture Summary\n\n")
        f.write(f"- **Number of Attention Layers**: {num_layers}\n")
        f.write(f"- **Number of Attention Heads**: {num_heads if num_heads else 'Unknown (check manually)'}\n\n")
        f.write("## Configuration Recommendation\n\n")
        f.write("```python\n")
        f.write(f'"num_layers": {num_layers},\n')
        if num_heads:
            f.write(f'"num_heads": {num_heads},  # For per-head thresholds\n')
        f.write("```\n\n")
        f.write("## Sample Attention Module\n\n")
        if sample_attn_module:
            f.write(f"**Module Type**: `{type(sample_attn_module).__name__}`\n\n")
            f.write("**Attributes**:\n")
            for attr in ['heads', 'num_heads', 'head_dim', 'inner_dim']:
                if hasattr(sample_attn_module, attr):
                    f.write(f"- `{attr}`: {getattr(sample_attn_module, attr)}\n")
    
    print(f"\nProfile saved to: {output_file}")
    print(f"  - Layers: {num_layers}")
    print(f"  - Heads: {num_heads if num_heads else 'Unknown'}")
    print("\n" + "="*80)
    print(f"Profiling complete! Check '{output_file}'")
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
  diffusers.utils.export_to_video(frames, filepath, fps=config["fps"])


