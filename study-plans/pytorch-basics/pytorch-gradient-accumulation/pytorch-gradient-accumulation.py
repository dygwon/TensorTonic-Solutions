import torch

def gradient_accumulation(w_init: torch.Tensor, micro_batches: list, lr: float, accum_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (final_weights, last_mean_gradient) as float32 tensors.
    """
    w = w_init.clone()
    w.requires_grad_()

    for i in range(0, len(micro_batches), accum_steps):
        micro_batch = micro_batches[i : i + accum_steps]
        mean_grads = 0.0
        for X, t in micro_batch:
            loss = (w.T @ X - t)**2
            loss.backward()

        mean_grads += w.grad / accum_steps
        
        with torch.no_grad():
            w -= lr * mean_grads
            w.grad = None

    return w, mean_grads