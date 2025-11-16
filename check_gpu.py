
import torch
print("PyTorch version:", torch.__version__)

if torch.cuda.is_available():
    print(f"CUDA is available!")
    print(f"Found {torch.cuda.device_count()} GPU(s):")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")

    # Test tensor on GPU
    test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
    print(f"Test tensor on GPU: {test_tensor}")
else:
    print("No GPU was detected by PyTorch.")
    print("Using CPU for training.")
