import t2v_wan21
import t2v_hunyuan
import t2v_cogvideox


def test_config_720():
    return {
        "model_name": "wan21_14b_720x1280x81_test",
        "generated_vids_dir": "generated_vids",
        "num_samples": 5,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "bitmask",
            "num_layers": 40,
            "thresh": 0.5/75600,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": True,
            "offload": False,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def test_config():
    return {
        "model_name": "wan21_14b_480x832x81_test",
        "generated_vids_dir": "generated_vids",
        "num_samples": 5,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "bitmask",
            "num_layers": 30,
            "thresh": 0.5/32670,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": True,
            "offload": False,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def get_wan21_1_3b_480x832x81_baseline_config():
    return {
        "model_name": "wan21_1.3b_480x832x81_baseline",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 30,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }
    

def get_wan21_1_3b_480x832x81_cached_config():
    config = get_wan21_1_3b_480x832x81_baseline_config()
    config["model_name"] = "wan21_1.3b_480x832x81_cached"
    config["attnprocessor_kwargs"] = {
            "processor": "cached",
            "num_layers": 30,
            "thresh": 0.5/32670,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": False,
            "offload": False,
        }
    return config


def get_wan21_1_3b_480x832x81_topk_config():
    config = get_wan21_1_3b_480x832x81_baseline_config()
    config["model_name"] = "wan21_1.3b_480x832x81_topk_24k"
    config["attnprocessor_kwargs"] = {
            "processor": "topk",
            "num_layers": 30,
            "select_k": 4096*6,
            "blocksz": 128,
        },
    return config


def get_wan21_1_3b_480x832x81_topcdf_config():
    config = get_wan21_1_3b_480x832x81_baseline_config()
    config["model_name"] = "wan21_1.3b_480x832x81_topcdf"
    config["attnprocessor_kwargs"] = {
            "processor": "topcdf",
            "num_layers": 30,
            "tau": 0.90,
            "gamma_q": 0.6,
            "gamma_k": 0.6,
            "blocksz": 128,
        }
    return config


def get_wan21_1_3b_720x1280x81_baseline_config():
    return {
        "model_name": "wan21_1.3b_720x1280x81_baseline",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 30,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def get_wan21_1_3b_720x1280x81_bitmaskcached_config():
    return {
        "model_name": "wan21_1.3b_720x1280x81_bitmaskcached",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "bitmask",
            "num_layers": 30,
            "thresh": 0.5/75600,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": True,
            "offload": False,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def get_wan21_1_3b_720x1280x81_topcdf_config():
    config = get_wan21_1_3b_720x1280x81_baseline_config()
    config["model_name"] = "wan21_1.3b_720x1280x81_topcdf"
    config["attnprocessor_kwargs"] = {
            "processor": "topcdf",
            "num_layers": 30,
            "tau": 0.90,
            "gamma_q": 0.6,
            "gamma_k": 0.6,
            "blocksz": 128,
        }
    return config


def get_wan21_14b_480x832x81_baseline_config():
    return {
        "model_name": "wan21_14b_480x832x81_baseline",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 40,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }



def get_wan21_14b_480x832x81_bitmaskcached_config():
    return {
        "model_name": "wan21_14b_480x832x81_bitmaskcached",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "bitmask",
            "num_layers": 40,
            "thresh": 0.6/32760,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": True,
            "offload": False,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def get_wan21_14b_480x832x81_topcdf_config():
    config = get_wan21_14b_480x832x81_baseline_config()
    config["model_name"] = "wan21_14b_480x832x81_topcdf"
    config["attnprocessor_kwargs"] = {
            "processor": "topcdf",
            "num_layers": 40,
            "tau": 0.90,
            "gamma_q": 0.6,
            "gamma_k": 0.6,
            "blocksz": 128,
        }
    return config



def get_wan21_14b_720x1280x81_baseline_config():
    return {
        "model_name": "wan21_14b_720x1280x81_baseline",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 40,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def get_wan21_1_3b_720x1280x81_2x_config():
    return {
        "model_name": "wan21_1_3b_720x1280x81_2x",
        "generated_vids_dir": "testvids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "2x",
            "num_layers": 30,
            "blocksz": 128
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def get_wan21_14b_720x1280x81_2x_config():
    return {
        "model_name": "wan21_14b_720x1280x81_2x",
        "generated_vids_dir": "testvids",
        "num_samples": 1,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "2x",
            "num_layers": 40,
            "blocksz": 128
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }

def get_wan21_14b_720x1280x81_topcdf_config():
    config = get_wan21_14b_720x1280x81_baseline_config()
    config["model_name"] = "wan21_14b_720x1280x81_topcdf"
    config["attnprocessor_kwargs"] = {
            "processor": "topcdf",
            "num_layers": 40,
            "tau": 0.90,
            "gamma_q": 0.6,
            "gamma_k": 0.6,
            "blocksz": 128,
        }
    return config


def get_wan21_14b_720x1280x81_bitmaskcached_config():
    return {
        "model_name": "wan21_14b_720x1280x81_bitmaskcached",
        "generated_vids_dir": "new_generated_vids",
        "num_samples": 5,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "bitmask",
            "num_layers": 40,
            "thresh": 0.5/75600,
            "blocksz": 128,
            "compute_cache_at": [0, 20, 60, ],
            "compress": True,
            "offload": False,
        },
        "run_fn": t2v_wan21.run_wan21,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


def get_hunyuan_720x1280x81_baseline_config():
    return {
        "model_name": "hunyuan_720x1280x81_baseline",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_hunyuan.get_hunyuan_pipeline,
        "init_fn_kwargs": {
            "model_path": "hunyuanvideo-community/HunyuanVideo",
        },
        "set_attnprocessor_fn": t2v_hunyuan.set_hunyuan_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 60,
        },
        "run_fn": t2v_hunyuan.run_hunyuan,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }

def get_hunyuan_720x1280x81_bitmaskcached_config():
    return {
        "model_name": "hunyuan_720x1280x81_bitmaskcached",
        "generated_vids_dir": "zeroseed_vids",
        "num_samples": 1,
        "init_fn": t2v_hunyuan.get_hunyuan_pipeline,
        "init_fn_kwargs": {
            "model_path": "hunyuanvideo-community/HunyuanVideo",
        },
        "set_attnprocessor_fn": t2v_hunyuan.set_hunyuan_attention,
        "attnprocessor_kwargs": {
            "processor": "bitmask",
            "num_layers": 60,
            "thresh": 0.5/75856,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": True,
            "offload": False,
        },
        "run_fn": t2v_hunyuan.run_hunyuan,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }


# https://huggingface.co/docs/diffusers/en/api/pipelines/cogvideox
def get_cogvideox_480x720x49_baseline_config():
    return {
        "model_name": "cogvideox_480x720x49_baseline",
        "generated_vids_dir": "generated_vids",
        "num_samples": 5,
        "init_fn": t2v_cogvideox.get_cogvideox_pipeline,
        "init_fn_kwargs": {
            "model_path": "THUDM/CogVideoX-5b",            
        },
        "set_attnprocessor_fn": t2v_cogvideox.set_cogvideox_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 42,
        },
        "run_fn": t2v_cogvideox.run_cogvideox,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 480,
            "width": 720,
            "num_frames": 49,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 8,
    }

def get_cogvideox_480x720x49_cached_config():
    return {
        "model_name": "cogvideox_480x720x49_cached",
        "generated_vids_dir": "generated_vids",
        "num_samples": 5,
        "init_fn": t2v_cogvideox.get_cogvideox_pipeline,
        "init_fn_kwargs": {
            "model_path": "THUDM/CogVideoX-5b",            
        },
        "set_attnprocessor_fn": t2v_cogvideox.set_cogvideox_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 42,
            "thresh": 0.5/75600,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": False,
            "offload": False,
        },
        "run_fn": t2v_cogvideox.run_cogvideox,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 480,
            "width": 720,
            "num_frames": 49,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 8,
    }




# https://huggingface.co/zai-org/CogVideoX1.5-5B
def get_cogvideox1_5_768x1360x81_baseline_config():
    return {
        "model_name": "cogvideox_1_5_768x1360x81_baseline",
        "generated_vids_dir": "generated_vids",
        "num_samples": 5,
        "init_fn": t2v_cogvideox.get_cogvideox_pipeline,
        "init_fn_kwargs": {
            "model_path": "THUDM/CogVideoX1.5-5B",
        },
        "set_attnprocessor_fn": t2v_cogvideox.set_cogvideox_attention,
        "attnprocessor_kwargs": {
            "processor": "baseline",
            "num_layers": 42,
        },
        "run_fn": t2v_cogvideox.run_cogvideox,
        "run_fn_kwargs": {
            "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
            "height": 768,
            "width": 1360,
            "num_frames": 81,
            "guidance_scale": 6.0,
            "num_inference_steps": 50,
        },
        "fps": 16,
    }




