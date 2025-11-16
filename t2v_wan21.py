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
  # 
  #config = distributedrunconfig.get_wan21_1_3b_480x832x81_sdpa_topcdf16_global_config()# work, complete
  #config = distributedrunconfig.get_wan21_1_3b_480x832x81_sdpa_cached_config() # work, complete
  #config = distributedrunconfig.get_wan21_14b_480x832x81_sdpa_topcdf128_global_config()  # work, complete
  #config = distributedrunconfig.get_wan21_14b_480x832x81_sdpa_cached_config()  # work, complete


  #config = distributedrunconfig.get_wan21_1_3b_720x1280x81_sdpa_topcdf_config() # work, complete
  #config = distributedrunconfig.get_wan21_1_3b_720x1280x81_sdpa_cached_config() # work, complete
  config = distributedrunconfig.get_wan21_14b_720x1280x81_sdpa_topcdf_config()  # work
  #config = distributedrunconfig.get_wan21_14b_720x1280x81_sdpa_cached_config()  # work, need to export SDPA_CHUNK=512


  prompt_base = "a horse bending down to drink water from a river"
  output_dir_base = pathlib.Path(config["generated_vids_dir"]) / config["model_name"]
  # Include block size in filename when available
  _blk = None
  try:
    _blk = config.get("attnprocessor_kwargs", {}).get("blocksz", None)
  except Exception:
    _blk = None
  _fname = f"test_{_blk}.mp4" if _blk is not None else "test.mp4"
  filepath_base = output_dir_base / _fname

  argparser = argparse.ArgumentParser()
  argparser.add_argument("--prompt", type=str, default=prompt_base, )
  argparser.add_argument("--filepath", type=str, default=None, )
  args = argparser.parse_args()
  prompt = args.prompt
  filepath = args.filepath
  if filepath is None:
    print(filepath)
    filepath = filepath_base
    output_dir_base.mkdir(parents=True, exist_ok=True)

  pipe = config['init_fn'](**config["init_fn_kwargs"])
  # Offload all large components to CPU to avoid GPU OOM for 14B,
  # but keep the VAE on GPU and unchanged (float32 as loaded above).
  pipe.enable_model_cpu_offload()
  if hasattr(pipe, 'vae'):
    pipe.vae.to('cuda', dtype=torch.float32)
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

