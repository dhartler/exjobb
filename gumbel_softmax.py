import torch
import torch.nn.functional as F

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    gumbel_noise = sample_gumbel(logits.shape)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize"""
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        # Straight-through trick
        y_hard = torch.zeros_like(y, dtype=torch.bfloat16).scatter_(0, y.argmax(dim=-1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y
    return y

def interpolate_vectors(vectors, temperature=1.0, device="cpu"):
    """
    Interpolate between a set of high-dimensional vectors using Gumbel-Softmax.
    
    Parameters:
    vectors (torch.Tensor): A tensor of shape (num_vectors, vector_dim).
    temperature (float): The temperature parameter for Gumbel-Softmax.
    
    Returns:
    torch.Tensor: The interpolated vector.
    """
    vectors.to("cpu")
    logits = torch.log(torch.ones(vectors.shape[0], dtype=torch.bfloat16))  # Uniform logits
    weights = gumbel_softmax(logits, temperature, hard=False)
    interpolated_vector = torch.matmul(weights.to("cpu").to(torch.bfloat16), vectors.to("cpu").to(torch.bfloat16))
    return interpolated_vector.to(device)