import accelerate  # pylint: disable=unused-import
import diffusers  # pylint: disable=unused-import
import torch
import torch.nn.functional as F
import transformers  # pylint: disable=unused-import
import t2v_wan21_processors
import numpy as np
import random
import os
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
  )
  pipe.to('cuda')
  # pipe.transformer.compile(mode='reduce-overhead', dynamic=True)
  return pipe


def run_wan21(
    pipe,
    prompt,
    negative_prompt="",
    height=480,
    width=832,
    num_frames=81,
    guidance_scale=5.0,
    num_inference_steps=50):
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

  torch.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  prompt = "a horse bending down to drink water from a river"
  # prompt = "A beautiful coastal beach in spring, waves lapping on sand by Vincent van Gogh"
  # prompt = "An oil painting of a couple in formal evening wear going home get caught in a heavy downpour with umbrellas"
  config = distributedrunconfig.test_config()

  pipe = config['init_fn'](**config["init_fn_kwargs"])
  config["set_attnprocessor_fn"](pipe, **config["attnprocessor_kwargs"])
  frames = config["run_fn"](
    pipe,
    prompt,
    **config["run_fn_kwargs"]
  )

  output_dir = pathlib.Path(config["generated_vids_dir"]) / config["model_name"]
  output_dir.mkdir(parents=True, exist_ok=True)
  filepath = output_dir / f'test.mp4'
  diffusers.utils.export_to_video(frames, filepath, fps=config["fps"])
  torch.cuda.empty_cache()




