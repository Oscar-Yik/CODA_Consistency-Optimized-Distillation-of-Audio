import torch

def check_cuda():
    cuda_available = torch.cuda.is_available()

    print(f"Is CUDA available: {cuda_available}")
    if cuda_available:
        devices = [d for d in range(torch.cuda.device_count())]
        device_names = [torch.cuda.get_device_name(d) for d in devices]

        print("Available devices:")
        for device_num, device_name in zip(devices, device_names):
            print(f"{device_num}: {device_name}")

def main():
    check_cuda()

if __name__ == "__main__":
    main()
