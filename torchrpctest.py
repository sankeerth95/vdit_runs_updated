# rpc_counter_v2.py
import os
import threading
import torch.distributed.rpc as rpc
import torch.distributed as dist

# Shared state and lock, conceptually "owned" by worker0.
# The script is the same for all processes, so these are defined in each one,
# but only the instances on worker0 are ever modified.
counter_rref = None
def get_counter_rref():
    """
    A helper function that runs on rank 0 and returns its RRef.
    Other workers will call this remotely.
    """
    global counter_rref
    return counter_rref


class AtomicCounter:

    def __init__(self):
        # self.counter_value = torch.tensor(0, dtype=torch.int)
        # self.incr = torch.tensor(1, dtype=torch.int)
        self.counter_value = 0 #torch.tensor(0, dtype=torch.int)
        self.incr = 1 # torch.tensor(1, dtype=torch.int)

        self._lock = threading.Lock()

    def atomicincr_counter(self):
        """
        Atomically adds a value to the counter. This function is defined
        on all workers but only the one on worker0 is ever used.
        """
        with self._lock:
            current_value = self.counter_value + 0
            self.counter_value += self.incr
        return current_value


def run_worker(rank, world_size):
    """Main function for each worker process."""
    print(f"Worker {rank} starting...")


    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # 1. Initialize RPC for all workers
    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(_channels=[
            "cma", "mpt_uv", "basic", "cuda_xth", "cuda_ipc", "cuda_basic"
        ]),
    )
    global counter_rref
    if rank == 0:
        print("Worker 0 creating the counter and RRef...")
        counter_rref = rpc.RRef(AtomicCounter())

    dist.barrier()
    print(f"Worker {rank} fetching RRef from worker 0...")
    if rank != 0:
        counter_rref = rpc.remote('worker0', get_counter_rref).to_here()
    dist.barrier()



    print(f"Worker {rank} (remote) is calling worker0...")
    current_val = counter_rref.rpc_sync().atomicincr_counter()


    # The returned value depends on the order of execution.
    print(f"Worker {rank} updates the counter value that is currently: {current_val}")

    # 3. Shutdown RPC
    # This also acts as a synchronization barrier before the script exits.
    print(f"Worker {rank} shutting down.")
    rpc.shutdown()
    dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun sets these automatically.
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    run_worker(rank, world_size)