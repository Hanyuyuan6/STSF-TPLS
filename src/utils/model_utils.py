import torch  # PyTorch
import time  # time module

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # total number of trainable parameters in the model

def measure_inference_time(model, input_len, device='cuda', num_runs=50):
    model.eval()  # put the model in eval mode, disabling dropout and other training-only behaviour
    model = model.to(device)  # move the model to the given device (GPU or CPU)
    dummy = torch.randn(1, input_len).to(device)  # random input tensor of shape (1, input_len), moved to the device

    for _ in range(10):  # warm the model up with 10 forward passes so first-run cache latency does not skew the measurement
        with torch.no_grad():
            _ = model(dummy)  # forward pass, no gradients

    if torch.cuda.is_available() and 'cuda' in str(device):  # when running on CUDA
        torch.cuda.synchronize()  # wait for all GPU work to finish so the timing is accurate

    t0 = time.time()  # record the start time

    for _ in range(num_runs):  # repeat the inference num_runs times and average
        with torch.no_grad():
            _ = model(dummy)  # forward pass, no gradients

    if torch.cuda.is_available() and 'cuda' in str(device):  # synchronize the GPU again so all inferences have completed
        torch.cuda.synchronize()

    return (time.time() - t0) / num_runs * 1000.0  # average inference time in milliseconds