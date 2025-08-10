import t2v_wan21


def test_config():
    return {
        "model_name": "wan21_1.3b_480x832x81_test",
        "generated_vids_dir": "generated_vids",
        "num_samples": 5,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        },
        "set_attnprocessor_fn": t2v_wan21.set_wan21_attention,
        "attnprocessor_kwargs": {
            "processor": "cached",
            "num_layers": 30,
            "thresh": 0.5/32670,
            "blocksz": 128,
            "compute_cache_at": [0, 15, 30, 45, 60, 80],
            "compress": False,
            "offloat": False,
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
        "generated_vids_dir": "generated_vids",
        "num_samples": 5,
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
            "offloat": False,
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

