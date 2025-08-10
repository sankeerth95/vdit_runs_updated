import t2v_wan21



def get_wan21_1_3b_480x832x81_config():
    return {
        "model_name": "wan21_1.3b_480x832x81_0.5thresh",
        "generated_vids_dir": "generated_vids/wan21_1.3b_480x832x81",
        "num_samples": 5,
        "init_fn": t2v_wan21.get_wan21_pipeline,
        "init_fn_kwargs": {
            "model_path": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
            "attnprocessor": "baseline",
            "thresh": 0.5/32670,
            "blocksz": 128,
            "select_k": 0,
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
    



