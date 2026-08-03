"""
SiTU-GLU Activation Module for Verace V1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SiTUGLU(nn.Module):
    """
    Sigmoid Tanh Unit Gated Linear Unit (SiTU-GLU).
    SiTU-GLU(x) = [ beta1 * tanh(W_g x / beta1) * Sigmoid(W_g x) ] * [ beta2 * tanh(W_u x / beta2) ]
    """
    def __init__(self, in_features: int, hidden_features: int, beta1: float = 4.0, beta2: float = 25.0):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_linear = self.w_gate(x)
        gate_capped = self.beta1 * torch.tanh(gate_linear / self.beta1)
        gate_act = gate_capped * torch.sigmoid(gate_linear)
        
        up_linear = self.w_up(x)
        up_capped = self.beta2 * torch.tanh(up_linear / self.beta2)
        
        return gate_act * up_capped
