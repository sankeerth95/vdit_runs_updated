# import accelerate  # pylint: disable=unused-import
import diffusers  # pylint: disable=unused-import
import torch

import torch.nn.functional as F
# import transformers  # pylint: disable=unused-import
import t2v_hunyuan_processor
import numpy as np
import random
import os
import pathlib


HunyuanVideoTransformer3DModel = diffusers.HunyuanVideoTransformer3DModel
AutoencoderKLHunyuanVideo = diffusers.AutoencoderKLHunyuanVideo
HunyuanVideoPipeline = diffusers.HunyuanVideoPipeline


def set_hunyuan_attention(pipe, *args, **kwargs):
  # count attn layers
  num_attn_layers = 0
  for name, module in pipe.transformer.named_modules():
    # print(name)
    if 'attn' in name and 'token_refiner' not in name:
        if hasattr(module, 'set_processor'):
            num_attn_layers += 1
  
  if "num_layers" not in kwargs:
     print("num_layers key not found. Setting it to counted value: ", num_attn_layers)
     kwargs["num_layers"] = num_attn_layers

  attn_processor = t2v_hunyuan_processor.CustomProcessor(**kwargs)
  for name, module in pipe.transformer.named_modules():
    if 'attn' in name and 'token_refiner' not in name:
        if hasattr(module, 'set_processor'):
            print('replacing attention processor of: ', name)
            module.set_processor(attn_processor)


def get_hunyuan_pipeline(model_path, *args, **kwargs):
  print("Loading HunyuanVideo model from", model_path)
  pipe = HunyuanVideoPipeline.from_pretrained(
      model_path,
      torch_dtype=torch.bfloat16,
      # device_map='balanced' 
  )
  return pipe


def run_hunyuan(
    pipe,
    prompt,
    negative_prompt="",
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=50):
  """Runs the HunyuanVideo model to generate a video.

  Args:
    pipe: The HunyuanVideo pipeline.
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


  # prompt = "A beautiful coastal beach in spring, waves lapping on sand by Vincent van Gogh"
  # prompt = "An oil painting of a couple in formal evening wear going home get caught in a heavy downpour with umbrellas"
  # config = distributedrunconfig.get_hunyuan_720x1280x81_baseline_config()
  config = distributedrunconfig.get_hunyuan_720x1280x81_bitmaskcached_config()
  prompt_base = "a horse bending down to drink water from a river"
  output_dir_base = pathlib.Path(config["generated_vids_dir"]) / config["model_name"]
  filepath_base = output_dir_base / "test.mp4"

  argparser = argparse.ArgumentParser()
  argparser.add_argument("--prompt", type=str, default=prompt_base)
  argparser.add_argument("--filepath", type=str, default=None)
  args = argparser.parse_args()
  prompt = args.prompt
  filepath = args.filepath

  if filepath is None:
    filepath = filepath_base
    output_dir_base.mkdir(parents=True, exist_ok=True)
    print(filepath)

  pipe = config['init_fn'](**config["init_fn_kwargs"])
  # pipe.to('cuda')
  pipe.enable_model_cpu_offload()
  config["set_attnprocessor_fn"](pipe, **config["attnprocessor_kwargs"])
  # pipe.transformer.compile(mode='reduce-overhead', dynamic=True)
  # torch.manual_seed(0)
  # np.random.seed(0)
  # random.seed(0)
  frames = config["run_fn"](
    pipe,
    prompt,
    **config["run_fn_kwargs"]
  )
  diffusers.utils.export_to_video(frames, filepath, fps=config["fps"])



