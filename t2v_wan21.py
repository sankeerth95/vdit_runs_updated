import accelerate  # pylint: disable=unused-import
import diffusers  # pylint: disable=unused-import
import torch
import torch.nn.functional as F
import transformers  # pylint: disable=unused-import
import t2v_wan21_processors
import numpy as np
import random


AutoencoderKLWan = diffusers.AutoencoderKLWan
WanPipeline = diffusers.WanPipeline


def set_wan21_attention(pipe, CustomProcessor): 
  for name, module in pipe.transformer.named_modules():
    if 'attn1' in name:
        if hasattr(module, 'set_processor'):
            print('replacing attention processor of: ', name)
            module.set_processor(CustomProcessor)


def get_wan21_pipeline(model_path, *args, **kwargs):
  print("Loading WAN2.1 model from", model_path)
  vae = AutoencoderKLWan.from_pretrained(
      model_path, subfolder="vae", torch_dtype=torch.float32
  )
  pipe = WanPipeline.from_pretrained(
      model_path, vae=vae, torch_dtype=torch.bfloat16,
  )
  pipe.to('cuda')
  attnprocessor = t2v_wan21_processors.MyCustomProcessor(
     thresh=kwargs["thresh"], blocksz=kwargs["blocksz"], select_k=kwargs["select_k"])
  set_wan21_attention(pipe, attnprocessor)
  return pipe


def run_wan21(
    pipe,
    prompt,
    negative_prompt,
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

  prompt = "a horse bending down to drink water from a river"
  # prompt = "A beautiful coastal beach in spring, waves lapping on sand by Vincent van Gogh"
  # prompt = "An oil painting of a couple in formal evening wear going home get caught in a heavy downpour with umbrellas"
  negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
  wan_pipeline_path = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'

  # for factor in [1., 10., 20., 40.,80.,]:
  torch.manual_seed(0)
  np.random.seed(0)
  random.seed(0)

  pipe = get_wan21_pipeline(wan_pipeline_path)
  set_wan21_attention(pipe, t2v_wan21_processors.MyCustomProcessor(thresh=0.5/75600))
  pipe.to('cuda')

  # sequence length is either 32760 or 75600
  # pipe.transformer.compile(mode='reduce-overhead', dynamic=True)
  frames = run_wan21(
    pipe,
    prompt,
    negative_prompt,
    height=720, #480,
    width=1280, #832
    # height=480,
    # width=832,
    num_frames=81,
    guidance_scale=6.0,
    num_inference_steps=50,
  )
  diffusers.utils.export_to_video(frames, f'sp_nocache_attn0.5_14b_720p.mp4', fps=16)
  torch.cuda.empty_cache()




