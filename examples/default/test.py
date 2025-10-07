import torch
from torch.nn import Linear, Sequential, ReLU

# Assuming DiscretePolicy and SplitLayer are in these locations
from rocket_learn.agent.discrete_policy import DiscreteFF
from rocket_learn.utils.util import SplitLayer

# TOTAL SIZE OF THE INPUT DATA
state_dim = 172

actor = DiscreteFF(state_dim,90,[256, 256, 256], "cpu")

# ------------------------------------------------------------------
# ADDED LINES FOR INFERENCE TEST
# ------------------------------------------------------------------

# 1. Create a dummy input tensor with the desired shape (batch_size=1, features=172)
dummy_input = torch.randn(2, state_dim)

# 2. Set the actor to evaluation mode (good practice for inference)
actor.eval()

# 3. Get an action without calculating gradients to save memory/computation
with torch.no_grad():
    output = actor.forward(dummy_input)
    action_distribution = actor.get_action_distribution(dummy_input)
    sampled_action = actor.sample_action(action_distribution)
    env_action = actor.env_compatible(sampled_action)

# 4. Print the input and output shapes to verify
print(f"Input shape: {dummy_input.shape}")
# print(f"Sampled Output (Tensor): {output}")
print(f"Sampled Output Shape: {output.shape}")
print(f"Sampled Action (Tensor): {sampled_action}")
print(f"Sampled Action shape (Tensor): {sampled_action.shape}")
print(f"Environment-compatible Action (NumPy): {env_action}")
print(f"Environment-compatible Action shape (NumPy): {env_action.shape}")

# EXPECTED OUTPUT:
# Input shape: torch.Size([1, 172])
# Sampled Action (Tensor): tensor([[0, 2, 1, 0, 2, 1, 0, 1]])  <- (Example values)
# Sampled Action shape (Tensor): torch.Size([1, 8])
# Environment-compatible Action (NumPy): [[0 2 1 0 2 1 0 1]]  <- (Example values)
# Environment-compatible Action shape (NumPy): (1, 8)