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
            "offload": True,
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
    import torch
    config = get_wan21_1_3b_480x832x81_baseline_config()
    config["model_name"] = "wan21_1.3b_480x832x81_topcdf"

    # Layer 0 thresholds (12 heads)
    layer0_is_sparse = [True, True, False, False, True, True, True, True, False, False, True, True]
    layer0_cdf = [0.8633, 0.9416, 1.0000, 1.0000, 0.9609, 0.9292, 0.8282, 0.8985, 1.0000, 1.0000, 0.9530, 0.9846]
    layer0_sim1 = [-0.9375, -0.9375, 1.0000, 1.0000, -0.9375, -0.9375, -0.9375, -0.9375, 1.0000, 1.0000, -0.9375, -0.9375]
    layer0_sim2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Layer 1 thresholds (is_sparse only provided)
    layer1_is_sparse = [True, True, True, True, True, True, True, True, True, True, False, True]

    # Build per-layer tables: fill missing layers by repeating layer 0 vectors
    L = 30  # number of layers for WAN 1.3B
    is_sparse_table = [layer0_is_sparse, layer1_is_sparse] + [layer0_is_sparse] * (L - 2)
    cdf_table = [layer0_cdf] + [layer0_cdf] * (L - 1)
    sim1_table = [layer0_sim1] + [layer0_sim1] * (L - 1)
    sim2_table = [layer0_sim2] + [layer0_sim2] * (L - 1)

    config["attnprocessor_kwargs"] = {
            "processor": "topcdf",
            "num_layers": L,
            "blocksz": 128,
            # Per-layer, per-head thresholds (shape: [L, 12])
            "is_sparse": torch.tensor(is_sparse_table, dtype=torch.bool),
            "cdfthreshd": torch.tensor(cdf_table, dtype=torch.float32),
            "simthreshd1": torch.tensor(sim1_table, dtype=torch.float32),
            "simthreshd2": torch.tensor(sim2_table, dtype=torch.float32),
        }
    return config


def get_wan21_1_3b_480x832x81_sdpa_topcdf16_config():
    import torch
    config = get_wan21_1_3b_480x832x81_baseline_config()
    config["model_name"] = "wan21_1.3b_480x832x81_sdpa_topcdf16"

    # Layer 0 thresholds (12 heads) - same as CUDA sparse version
    layer0_is_sparse = [True, True, False, False, True, True, True, True, False, False, True, True]
    layer0_cdf = [0.8633, 0.9416, 1.0000, 1.0000, 0.9609, 0.9292, 0.8282, 0.8985, 1.0000, 1.0000, 0.9530, 0.9846]
    layer0_sim1 = [-0.9375, -0.9375, 1.0000, 1.0000, -0.9375, -0.9375, -0.9375, -0.9375, 1.0000, 1.0000, -0.9375, -0.9375]
    layer0_sim2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Layer 1 thresholds (is_sparse only provided)
    layer1_is_sparse = [True, True, True, True, True, True, True, True, True, True, False, True]

    # Build per-layer tables: fill missing layers by repeating layer 0 vectors
    L = 30  # number of layers for WAN 1.3B
    is_sparse_table = [layer0_is_sparse, layer1_is_sparse] + [layer0_is_sparse] * (L - 2)
    cdf_table = [layer0_cdf] + [layer0_cdf] * (L - 1)
    sim1_table = [layer0_sim1] + [layer0_sim1] * (L - 1)
    sim2_table = [layer0_sim2] + [layer0_sim2] * (L - 1)

    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_topcdf16",
            "num_layers": L,
            "blocksz": 128,  # Use 16-token blocks for SDPA validation
            # Per-layer, per-head thresholds (shape: [L, 12])
            "is_sparse": torch.tensor(is_sparse_table, dtype=torch.bool),
            "cdfthreshd": torch.tensor(cdf_table, dtype=torch.float32),
            "simthreshd1": torch.tensor(sim1_table, dtype=torch.float32),
            "simthreshd2": torch.tensor(sim2_table, dtype=torch.float32),
            "model_name": "wan_1_3b_480",
        }
    return config


def get_wan21_1_3b_480x832x81_sdpa_topcdf16_global_config():
    """Use SDPA Top-CDF with a single global threshold shared across heads.

    This avoids per-head/per-layer threshold tables and instead passes
    scalar tau/gamma values that broadcast equally to all heads.
    """
    config = get_wan21_1_3b_480x832x81_baseline_config()
    config["model_name"] = "wan21_1.3b_480x832x81_sdpa_topcdf16_global"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_topcdf16",
            "num_layers": 30,
            "blocksz": 128,
            # Global thresholds (same for all heads and layers)
            "tau": 0.95,
            "gamma_q": 0.5,
            "gamma_k": 0.5,
            "model_name": "wan_1_3b_480",
        }
    return config


def get_wan21_1_3b_480x832x81_sdpa_cached_config():

    config = get_wan21_1_3b_480x832x81_baseline_config()
    blocksz = 128
    config["model_name"] = f"wan21_1.3b_480x832x81_sdpa_cached_block{blocksz}"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_cached",
            "num_layers": 30,
            "blocksz": blocksz,
            "thresh": 0.5/32670,  # Same threshold as CUDA cached version
            "compute_cache_at": [0, 5, 12, 20, 30, 40],  # Recompute mask periodically
            "model_name": "wan_1_3b_480",
        }
    return config


def get_wan21_14b_480x832x81_sdpa_topcdf128_global_config():
    """Use SDPA Top-CDF with a single global threshold shared across heads for 14B model.

    This avoids per-head/per-layer threshold tables and instead passes
    scalar tau/gamma values that broadcast equally to all heads.
    """
    config = get_wan21_14b_480x832x81_baseline_config()
    config["model_name"] = "wan21_14b_480x832x81_sdpa_topcdf128_global"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_topcdf16",
            "num_layers": 40,
            "blocksz": 16,
            # Global thresholds (same for all heads and layers)
            "tau": 0.95,
            "gamma_q": 0.5,
            "gamma_k": 0.5,
            "model_name": "wan_14b_480",
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


def get_wan21_14b_480x832x81_sdpa_cached_config():
    """SDPA with naive cache mask for WAN 2.1 14B at 480x832x81."""
    config = get_wan21_14b_480x832x81_baseline_config()
    blocksz = 128
    config["model_name"] = f"wan21_14b_480x832x81_sdpa_cached_block{blocksz}"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_cached",
            "num_layers": 40,
            "blocksz": blocksz,
            "thresh": 0.6/32760,  # Match 14B cached threshold scale
            "compute_cache_at": [0, 5, 12, 20, 30, 40],  # Periodic recompute
            "model_name": "wan_14b_480",
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


# SDPA variants for 720x1280x81 (requested in t2v_wan21.py 85-92)
def get_wan21_1_3b_720x1280x81_sdpa_topcdf_config():
    config = get_wan21_1_3b_720x1280x81_baseline_config()
    config["model_name"] = "wan21_1.3b_720x1280x81_sdpa_topcdf"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_topcdf16",
            "num_layers": 30,
            "tau": 0.90,
            "gamma_q": 0.6,
            "gamma_k": 0.6,
            "blocksz": 32,
            "model_name": "wan_1_3b_720",
        }
    return config


def get_wan21_1_3b_720x1280x81_sdpa_cached_config():
    config = get_wan21_1_3b_720x1280x81_baseline_config()
    blocksz = 64
    config["model_name"] = f"wan21_1.3b_720x1280x81_sdpa_cached_block{blocksz}"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_cached",
            "num_layers": 30,
            "blocksz": blocksz,
            "thresh": 0.5/75600,
            "compute_cache_at": [0, 5, 12, 20, 30, 40],
            "model_name": "wan_1_3b_720",
        }
    return config


def get_wan21_14b_720x1280x81_sdpa_topcdf_config():
    config = get_wan21_14b_720x1280x81_baseline_config()
    config["model_name"] = "wan21_14b_720x1280x81_sdpa_topcdf"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_topcdf16",
            "num_layers": 40,
            "tau": 0.90,
            "gamma_q": 0.6,
            "gamma_k": 0.6,
            "blocksz": 128,
            "model_name": "wan_14b_720",
        }
    return config


def get_wan21_14b_720x1280x81_sdpa_cached_config():
    config = get_wan21_14b_720x1280x81_baseline_config()
    blocksz = 16
    config["model_name"] = f"wan21_14b_720x1280x81_sdpa_cached_block{blocksz}"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_cached",
            "num_layers": 40,
            "blocksz": blocksz,
            "thresh": 0.5/75600,
            "compute_cache_at": [0, 5, 12, 20, 30, 40],
            "model_name": "wan_14b_720",
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


def get_hunyuan_720x1280x81_sdpa_cached_config():
    """Use SDPA with naive cache mask (threshold-based, pure PyTorch) for HunyuanVideo.
    
    This is a pure PyTorch implementation that doesn't rely on CUDA kernels,
    enabling flexible block sizes like 16. Uses column-wise max + threshold
    for conservative mask generation (OR logic across queries).
    """
    config = get_hunyuan_720x1280x81_baseline_config()
    blocksz = 16
    config["model_name"] = f"hunyuan_720x1280x81_sdpa_cached_block{blocksz}"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_cached",
            "num_layers": 60,
            "blocksz": blocksz,
            "thresh": 0.5/75856,  # Same threshold as CUDA cached version
            "compute_cache_at": [0, 5, 12, 20, 30, 40],  # Recompute mask periodically
        }
    return config


def get_hunyuan_720x1280x81_sdpa_cached_compressed_config():
    """Use SDPA with naive cache mask + bit-packed compression (pure PyTorch) for HunyuanVideo."""
    config = get_hunyuan_720x1280x81_baseline_config()
    blocksz = 16
    config["model_name"] = f"hunyuan_720x1280x81_sdpa_cached_compressed_block{blocksz}"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_cached_compressed",
            "num_layers": 60,
            "blocksz": blocksz,
            "thresh": 0.5/75856,  # Same threshold as CUDA cached version
            "compute_cache_at": [0, 5, 12, 20, 30, 40],  # Recompute mask periodically
        }
    return config

def get_hunyuan_720x1280x81_sdpa_topcdf16_global_config():
    """Use SDPA Top-CDF with a single global threshold shared across heads for HunyuanVideo.

    Pure PyTorch SDPA path that injects a Top-CDF mask computed at block level.
    """
    config = get_hunyuan_720x1280x81_baseline_config()
    config["model_name"] = "hunyuan_720x1280x81_sdpa_topcdf16_global"
    config["attnprocessor_kwargs"] = {
            "processor": "sdpa_topcdf16",
            "num_layers": 60,
            "blocksz": 16,
            # Global thresholds (same for all heads and layers)
            "tau": 0.95,
            "gamma_q": 0.5,
            "gamma_k": 0.5,
        }
    return config


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




