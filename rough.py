import torch
import threading

def infinite_loop(device):
    while True:
        # Do some work on the GPU
        torch.cuda.synchronize(device)

def main():
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # If there is only one GPU, use it for all threads
    if num_gpus == 1:
        device = 0

    # Create a thread for each GPU
    threads = []
    for i in range(50):
        thread = threading.Thread(target=infinite_loop, args=(device,))
        thread.daemon = True
        thread.start()

    # Wait for all threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()
