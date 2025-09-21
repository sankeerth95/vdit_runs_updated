
# distributed_hello.py
import json
import torch
import torch.distributed
import torch.distributed.rpc
import os
import threading
import diffusers.utils
import pathlib
import numpy as np
import random
import subprocess
import sys



dist = torch.distributed
rpc = torch.distributed.rpc
export_to_video = diffusers.utils.export_to_video


class RoundRobinScheduler:
    def __init__(self, total_items, rank, world_size, dist=None):
        self.total_items = total_items
        self.world_size = world_size
        self.current_item = 0
        self.rank = rank

    def get_next_workitem(self):
        while self.current_item < self.total_items:
            item_to_process = self.current_item
            self.current_item += 1
            if item_to_process % self.world_size == self.rank:
                return item_to_process
        return None


class AtomicCounter:
    def __init__(self):
        self.counter_value = 0
        self.incr = 1
        self._lock = threading.Lock()

    def atomicincr_counter(self):
        with self._lock:
            current_value = self.counter_value + 0
            self.counter_value += self.incr
        return current_value


class AtomicCounterScheduler:
    counter_rref = None
    @staticmethod 
    def get_counter_rref():
        return AtomicCounterScheduler.counter_rref

    def __init__(self, total_items, rank, world_size, dist):
        self.total_items = total_items
        self.world_size = world_size
        if rank == 0:
            print("Worker 0 creating the counter and RRef...")
            AtomicCounterScheduler.counter_rref = rpc.RRef(AtomicCounter())

        dist.barrier()
        if rank != 0:
            AtomicCounterScheduler.counter_rref = \
                rpc.remote('worker0', AtomicCounterScheduler.get_counter_rref).to_here()
        dist.barrier()

    def get_next_workitem(self):
        current_val = AtomicCounterScheduler.counter_rref.rpc_sync().atomicincr_counter()
        if current_val >= self.total_items:
            return None
        return current_val


def run(config):
    if not ("RANK" in os.environ and "WORLD_SIZE" in os.environ):
        print("torchrun not properly setup")
        exit()
    host = os.getenv("SLURM_NODEID") 
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    rpc.init_rpc(
        name=f"worker{rank}", rank=rank, world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(_channels=[
            "cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"
        ]),
    )
    # torch.cuda.set_device(local_rank)
    print(f"Hello from Rank {rank}/{world_size} on GPU {local_rank} on node {os.getenv('SLURM_NODEID')}!")

    with open('VBench_full_info.json', 'r') as f:
        data = json.load(f)

    # scheduler = RoundRobinScheduler(len(data), rank, world_size)
    scheduler = AtomicCounterScheduler(len(data), rank, world_size, dist)

    # pipe.transformer.compile(mode='reduce-overhead', dynamic=True)

    model_name = config['model_name']
    base_output_dir =  pathlib.Path(config["generated_vids_dir"]) / config["model_name"]
    num_samples = config["num_samples"]

    child_env = os.environ.copy()

    # 2. Add or modify the environment variables
    child_env["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"
    print(f'Rank {rank}: starting to generate frames for {model_name}')
    dist.barrier()

    error = False
    while True:

        prompt_idx = scheduler.get_next_workitem()
        if prompt_idx is None:
            break

        prompt_data = data[prompt_idx]
        prompt = prompt_data['prompt_en']
        dimensions = prompt_data.get('dimension', [])
        
        print(f"Worker {rank} updates the counter value that is currently: {prompt_idx}")

        # https://github.com/Vchitect/VBench/issues/130
        print(dimensions)
        print(f"Rank {rank}: Generating {num_samples} samples for prompt: {prompt[:50]}...")


        for sample_idx in range(num_samples):
            num_file_exists = 0
            filepaths = []
            for dimension in dimensions:
                dim_dir = base_output_dir / dimension
                dim_dir.mkdir(parents=True, exist_ok=True)
                
                filename = f"{prompt}-{sample_idx}.mp4"
                filepath = dim_dir / filename               
                filepaths.append(filepath)
                if filepath.exists():
                    num_file_exists += 1

            if num_file_exists == len(dimensions):
                print(f"Video already exists in all paths: {prompt}-{sample_idx}.mp4. Skipping...")
                continue
            try:
                print('running on localrank: ', local_rank)
                actual_command = [
                    "/home/sdurvasula/miniconda3/envs/spartan/bin/python", 
                    "t2v_wan21.py",
                    "--prompt", f"{prompt}",
                    "--filepath", f"{filepaths[0]}",
                ]
                try:
                    result = subprocess.run(
                        actual_command, 
                        check=True,        # Raise an exception if the command returns a non-zero exit code
                        capture_output=True,
                        text=True,          # Decode stdout/stderr as text
                        env=child_env,
                    )
                    print("--- Subprocess STDOUT ---")
                    print(result.stdout)
                    print("--- Subprocess STDERR ---")
                    print(result.stderr)
                    print(f"Subprocess finished successfully for Array Task {prompt_idx}.")
                    for filepath in filepaths[1:]:
                        # copy
                        os.system(f"cp {filepaths[0]} {filepath}")

                except FileNotFoundError:
                    print(f"Error: The command '{actual_command[0]}' was not found.", file=sys.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"Error: Subprocess for Array Task {prompt_idx} failed with exit code {e.returncode}", file=sys.stderr)
                    print("--- Subprocess STDOUT ---")
                    print(e.stdout)
                    print("--- Subprocess STDERR ---")
                    print(e.stderr)

                print(f"Rank {rank}, host {host}: Saved sample {sample_idx} for prompt in {len(dimensions)} dimension(s)")

            except Exception as e:
                print(f"Rank {rank}, host {host}: Error generating sample {sample_idx} for prompt '{prompt[:50]}...': {e}")
                torch.cuda.empty_cache()




    print(f"Rank {rank}: Finished generating all videos")
    rpc.shutdown()
    print(f"Worker {rank} completed")
    if rank == 0:
        dist.destroy_process_group()


if __name__ == '__main__':
    import distributedrunconfig
    # config_ = distributedrunconfig.get_hunyuan_720x1280x81_baseline_config()
    config_ = distributedrunconfig.get_wan21_14b_720x1280x81_bitmaskcached_config()
    config = {
        "num_samples": config_["num_samples"],
        "model_name": config_["model_name"],
        "generated_vids_dir": config_["generated_vids_dir"],
    }
    run(config)
