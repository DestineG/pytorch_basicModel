import torch


def print_gpu_info():
    if not torch.cuda.is_available():
        print("No GPU available.")
        return
    
    for i in range(torch.cuda.device_count()):
        prop = torch.cuda.get_device_properties(i)
        total_mem = prop.total_memory / 1024**2  # 转 MB
        allocated = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        free_mem = total_mem - reserved

        print(f"GPU {i}: {prop.name}")
        print(f"  Total Memory     : {total_mem:.2f} MB")
        print(f"  Memory Allocated : {allocated:.2f} MB")
        print(f"  Memory Reserved  : {reserved:.2f} MB")
        print(f"  Memory Free      : {free_mem:.2f} MB")
        print("-" * 40)

def main():
    # Create a tensor
    x = torch.tensor(129 * [1.0, 2.0, 3.0, 4.0])

    # Perform a simple operation
    y = x * 2

    # Check if CUDA is available and move tensor to GPU if possible
    if torch.cuda.is_available():
        x_gpu = x.to('cuda')
    else:
        print("CUDA is not available. Running on CPU.")
    device = torch.cuda.current_device()
    print("Allocated (bytes):", torch.cuda.memory_allocated(device))
    print("Reserved  (bytes):", torch.cuda.memory_reserved(device))
    print_gpu_info()

if __name__ == "__main__":
    # main()
    t = torch.randn(1, 3, requires_grad=True)
    print(t)
    y1 = t.pow(3).sum()
    y1.backward()
    print(t.grad)
    t.grad.zero_()
    y1= (-3 - t.pow(3).sum()).pow(2)
    y1.backward()
    print(t.grad)