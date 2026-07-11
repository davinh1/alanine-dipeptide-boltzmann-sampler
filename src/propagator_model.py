"""Conditional flow-matching propagator p(x_{t+tau} | x_t) for the dynamics emulator.
The vector field is conditioned on x_t by concatenation. Trained with conditional OT-CFM;
sampled by a conditional Heun ODE from Gaussian noise. See models.py for shared helpers."""
import torch, torch.nn as nn
from models import sinusoidal_embedding, EMA


class CondResMLP(nn.Module):
    def __init__(self, dim, hidden=256, n_blocks=4, temb_dim=128):
        super().__init__()
        self.dim = dim; self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                      nn.Linear(temb_dim, temb_dim))
        self.inp = nn.Linear(dim * 2 + temb_dim, hidden)   # [x_interp, x_cond, temb]
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                     nn.Linear(hidden, hidden)) for _ in range(n_blocks)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_blocks)])
        self.out = nn.Linear(hidden, dim)

    def forward(self, x, t, cond):
        te = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        h = self.inp(torch.cat([x, cond, te], -1))
        for blk, nm in zip(self.blocks, self.norm):
            h = h + blk(nm(h))
        return self.out(h)


def cfm_cond_loss(model, x0_cond, x1_target, sigma_min=1e-4):
    """Conditional OT-CFM loss: regress the vector field from Gaussian base to x1_target,
    conditioned on x0_cond (the current state)."""
    B = x0_cond.shape[0]
    t = torch.rand(B, 1)
    x0 = torch.randn_like(x1_target)
    xt = (1 - (1 - sigma_min) * t) * x0 + t * x1_target
    u = x1_target - (1 - sigma_min) * x0
    return ((model(xt, t.squeeze(-1), x0_cond) - u) ** 2).mean()


@torch.no_grad()
def propagate(model, cond, dim, steps=30):
    """Sample x_{t+tau} ~ p(.|cond) via conditional Heun probability-flow ODE."""
    B = cond.shape[0]; x = torch.randn(B, dim); dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((B,), i * dt)
        v = model(x, t, cond)
        xm = x + dt * v
        tn = torch.full((B,), (i + 1) * dt)
        vn = model(xm, tn, cond)
        x = x + dt * 0.5 * (v + vn)
    return x
