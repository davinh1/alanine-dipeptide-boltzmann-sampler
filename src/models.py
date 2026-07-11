"""Flow-matching (OT-CFM) and DDPM models over the ala2 internal-coordinate feature vector.
Shared residual-MLP backbone with sinusoidal time embedding. EMA included.
"""
import math, copy
import numpy as np
import torch
import torch.nn as nn


def sinusoidal_embedding(t, dim):
    # t in [0,1] (FM) or step index scaled (DDPM), shape (B,)
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / (half - 1))
    args = t[:, None].float() * freqs[None] * 1000.0
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], -1)
    return emb


class ResBlock(nn.Module):
    def __init__(self, dim, temb_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lin1 = nn.Linear(dim, dim)
        self.temb = nn.Linear(temb_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.lin2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x, temb):
        h = self.lin1(self.act(self.norm1(x)))
        h = h + self.temb(temb)
        h = self.lin2(self.act(self.norm2(h)))
        return x + h


class ResMLP(nn.Module):
    """Predicts a vector field (FM) or noise (DDPM) of same dim as input."""
    def __init__(self, data_dim, hidden=512, n_blocks=6, temb_dim=128):
        super().__init__()
        self.temb_dim = temb_dim
        self.time_mlp = nn.Sequential(nn.Linear(temb_dim, temb_dim), nn.SiLU(),
                                      nn.Linear(temb_dim, temb_dim))
        self.inp = nn.Linear(data_dim, hidden)
        self.blocks = nn.ModuleList([ResBlock(hidden, temb_dim) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, data_dim))

    def forward(self, x, t):
        temb = self.time_mlp(sinusoidal_embedding(t, self.temb_dim))
        h = self.inp(x)
        for b in self.blocks:
            h = b(h, temb)
        return self.out(h)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for s, p in zip(self.shadow.parameters(), model.parameters()):
            s.mul_(self.decay).add_(p, alpha=1 - self.decay)
        for s, p in zip(self.shadow.buffers(), model.buffers()):
            s.copy_(p)


# ---------------- Flow Matching (independent-coupling OT-CFM) ----------------
def fm_loss(model, x1, sigma_min=1e-4):
    B = x1.shape[0]
    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    tt = t[:, None]
    # linear interpolant with small sigma; target vector field u = x1 - (1-sigma_min) x0
    xt = (1 - (1 - sigma_min) * tt) * x0 + tt * x1
    target = x1 - (1 - sigma_min) * x0
    pred = model(xt, t)
    return ((pred - target) ** 2).mean()


@torch.no_grad()
def fm_sample(model, n, dim, device, steps=100, method="heun"):
    x = torch.randn(n, dim, device=device)
    ts = torch.linspace(0, 1, steps + 1, device=device)
    for i in range(steps):
        t0, t1 = ts[i], ts[i + 1]
        h = t1 - t0
        tb = torch.full((n,), t0, device=device)
        k1 = model(x, tb)
        if method == "euler":
            x = x + h * k1
        else:  # heun
            x2 = x + h * k1
            tb1 = torch.full((n,), t1, device=device)
            k2 = model(x2, tb1)
            x = x + h * 0.5 * (k1 + k2)
    return x


# ---------------- DDPM (cosine schedule, epsilon prediction) ----------------
def cosine_betas(T, s=0.008):
    steps = T + 1
    x = torch.linspace(0, T, steps)
    ac = torch.cos(((x / T + s) / (1 + s)) * math.pi / 2) ** 2
    ac = ac / ac[0]
    betas = 1 - (ac[1:] / ac[:-1])
    return torch.clip(betas, 1e-4, 0.999)


class DDPMSchedule:
    def __init__(self, T=1000, device="cpu"):
        self.T = T
        betas = cosine_betas(T).to(device)
        self.betas = betas
        self.alphas = 1 - betas
        self.acp = torch.cumprod(self.alphas, 0)          # alpha-bar
        self.sqrt_acp = torch.sqrt(self.acp)
        self.sqrt_1macp = torch.sqrt(1 - self.acp)


def ddpm_loss(model, x0, sched):
    B = x0.shape[0]
    t = torch.randint(0, sched.T, (B,), device=x0.device)
    noise = torch.randn_like(x0)
    xt = sched.sqrt_acp[t][:, None] * x0 + sched.sqrt_1macp[t][:, None] * noise
    pred = model(xt, t.float() / sched.T)
    return ((pred - noise) ** 2).mean()


@torch.no_grad()
def ddpm_sample(model, n, dim, sched, device, mode="ancestral", ddim_steps=100, x0_clip=12.0):
    """Predict x0 from eps, clamp to data range (data is standardized), then use the
    proper DDPM posterior (ancestral) or DDIM update. x0-clamping prevents the divergence
    that occurs when alpha_bar -> 0 at the final timesteps of a cosine schedule."""
    acp = sched.acp
    def eps_to_x0(x, eps, t):
        return (x - sched.sqrt_1macp[t] * eps) / sched.sqrt_acp[t]
    if mode == "ancestral":
        x = torch.randn(n, dim, device=device)
        for t in reversed(range(sched.T)):
            tb = torch.full((n,), t / sched.T, device=device)
            eps = model(x, tb)
            x0 = torch.clamp(eps_to_x0(x, eps, t), -x0_clip, x0_clip)
            acp_prev = acp[t - 1] if t > 0 else torch.ones((), device=device)
            beta_t = sched.betas[t]
            # posterior mean coefficients
            c_x0 = torch.sqrt(acp_prev) * beta_t / (1 - acp[t])
            c_xt = torch.sqrt(sched.alphas[t]) * (1 - acp_prev) / (1 - acp[t])
            mean = c_x0 * x0 + c_xt * x
            if t > 0:
                var = beta_t * (1 - acp_prev) / (1 - acp[t])
                x = mean + torch.sqrt(var) * torch.randn_like(x)
            else:
                x = mean
        return x
    else:  # DDIM (deterministic)
        ts = torch.linspace(sched.T - 1, 0, ddim_steps, device=device).long()
        x = torch.randn(n, dim, device=device)
        for i in range(len(ts)):
            t = ts[i]
            tb = torch.full((n,), t.item() / sched.T, device=device)
            eps = model(x, tb)
            x0 = torch.clamp(eps_to_x0(x, eps, t), -x0_clip, x0_clip)
            if i < len(ts) - 1:
                acp_next = acp[ts[i + 1]]
                x = torch.sqrt(acp_next) * x0 + torch.sqrt(1 - acp_next) * eps
            else:
                x = x0
        return x


def train_model(model, data, n_steps, batch_size, lr, weight_decay, warmup_steps,
                grad_clip, ema_decay, loss_fn, device, log_every=1000):
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (n_steps - warmup_steps)))
    sched_lr = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    ema = EMA(model, ema_decay)
    N = data.shape[0]
    losses = []
    for step in range(n_steps):
        idx = torch.randint(0, N, (batch_size,), device=device)
        x = data[idx]
        loss = loss_fn(model, x)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step(); sched_lr.step(); ema.update(model)
        losses.append(loss.item())
        if (step + 1) % log_every == 0:
            print(f"  step {step+1}/{n_steps} loss {np.mean(losses[-log_every:]):.5f}", flush=True)
    return model, ema, losses
