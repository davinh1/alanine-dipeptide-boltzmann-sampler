"""Train FM (OT-CFM) and DDPM on encoded ala2 internal coords. Saves checkpoints + EMA + loss curves."""
import sys, os, json, time
import numpy as np, torch
sys.path.insert(0, os.getcwd())
from models import ResMLP, EMA, fm_loss, ddpm_loss, DDPMSchedule, train_model

def main():
    which = sys.argv[1]  # 'fm' or 'ddpm'
    torch.manual_seed(20260710); np.random.seed(20260710)
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "8")))
    device = "cpu"
    d = np.load("run/data/encoded.npz", allow_pickle=True)
    X = torch.from_numpy(d["X_train"].astype(np.float32)).to(device)
    DIM = X.shape[1]
    H, NB, TE = 256, 4, 128
    BS, NSTEPS, LR, WD, WARM, CLIP, EMAD = 1024, 20000, 2e-4, 1e-5, 500, 1.0, 0.999

    net = ResMLP(DIM, hidden=H, n_blocks=NB, temb_dim=TE).to(device)
    if which == "fm":
        loss_fn = lambda m, x: fm_loss(m, x, sigma_min=1e-4)
        ckpt = "run/checkpoints/fm.pt"
    else:
        sched = DDPMSchedule(T=1000, device=device)
        loss_fn = lambda m, x: ddpm_loss(m, x, sched)
        ckpt = "run/checkpoints/ddpm.pt"

    t0 = time.time()
    net, ema, losses = train_model(net, X, NSTEPS, BS, LR, WD, WARM, CLIP, EMAD,
                                   loss_fn, device, log_every=2000)
    wall = (time.time()-t0)/60
    torch.save({"model": net.state_dict(), "ema": ema.shadow.state_dict(),
                "losses": losses, "config": {"dim": DIM, "hidden": H, "n_blocks": NB,
                "temb_dim": TE, "n_steps": NSTEPS, "batch": BS, "lr": LR, "which": which,
                "ddpm_T": 1000}}, ckpt)
    np.save(f"run/checkpoints/{which}_losses.npy", np.array(losses))
    print(f"[{which}] done in {wall:.1f} min, final loss {np.mean(losses[-500:]):.5f} -> {ckpt}", flush=True)

if __name__ == "__main__":
    main()
