"""
Fine-tune the Re-ID backbone on data mined from the site (scripts/mine_reid_data.py) and MEASURE the gain.

    python scripts/train_reid.py --data data/reid_mining
    python scripts/train_reid.py --data data/reid_mining --eval-only --weights weights/osnet_x1_0_site.pt

Whole videos are held out for validation. Before and after training it prints, on the held-out videos:
    threshold       the similarity that rejects 99.5 % of DIFFERENT workers seen at the same time (+0.03),
                    i.e. the re-entry threshold the identity manager would learn on this site
    genuine_accept  the share of SAME-worker pairs that still pass that threshold   <- the number to improve
    eer             equal error rate
The fine-tuned weights are only worth using when genuine_accept clearly increases. Then, in configs/settings.yaml:
    reid: {weights_path: weights/osnet_x1_0_site.pt}      (same model_name)
GPU strongly recommended (an RTX-class card trains 10 epochs on a few hours of video in well under an hour).
"""
from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.identity.finetune import (  # noqa: E402
    MinedData,
    batch_hard_triplet,
    l2sp_penalty,
    pair_similarities,
    sample_group_batch,
    separation_report,
)

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_image(path: str, augment: bool, rng: random.Random) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((256, 128, 3), np.uint8)
    img = cv2.cvtColor(cv2.resize(img, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if augment:
        if rng.random() < 0.5:
            img = img[:, ::-1]
        # light differs between cameras / day and night: brightness & contrast jitter
        img = np.clip((img - 0.5) * rng.uniform(0.75, 1.25) + 0.5 + rng.uniform(-0.12, 0.12), 0.0, 1.0)
        # small shift
        pad = np.pad(img, ((10, 10), (5, 5), (0, 0)), mode="edge")
        y, x = rng.randint(0, 20), rng.randint(0, 10)
        img = pad[y:y + 256, x:x + 128]
        # random erasing: partial occlusion (pallets, machines, other people)
        if rng.random() < 0.5:
            h, w = rng.randint(30, 110), rng.randint(20, 70)
            y, x = rng.randint(0, 256 - h), rng.randint(0, 128 - w)
            img = img.copy()
            img[y:y + h, x:x + w] = MEAN
    img = (img - MEAN) / STD
    return np.ascontiguousarray(img.transpose(2, 0, 1))


def build_model(model_name: str, weights: str | None, device):
    import torch
    import torchreid

    model = torchreid.models.build_model(name=model_name, num_classes=1, loss="triplet", pretrained=False)
    if weights:
        torchreid.utils.load_pretrained_weights(model, weights)
    return model.to(device)


def make_embedder(model, device, body_w: float, head_w: float, batch: int = 64):
    """Production-like descriptor: weighted body + upper-body features (the colour branch is not learned)."""
    import torch
    import torch.nn.functional as F

    def embed_paths(paths):
        out = []
        for i in range(0, len(paths), batch):
            x = torch.from_numpy(np.stack([load_image(p, False, random.Random(0)) for p in paths[i:i + batch]]))
            with torch.no_grad():
                out.append(F.normalize(model(x.to(device)), dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 512), np.float32)

    def embed(pairs):
        model.eval()
        b = embed_paths([p[0] for p in pairs])
        u = embed_paths([p[1] for p in pairs])
        e = np.concatenate([np.sqrt(body_w) * b, np.sqrt(head_w) * u], axis=1)
        return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)

    return embed


def evaluate(model, data: MinedData, device, body_w: float, head_w: float) -> dict:
    g, i = pair_similarities(data, make_embedder(model, device, body_w, head_w))
    return separation_report(g, i)


def fmt(rep: dict) -> str:
    if "genuine_accept" not in rep:
        return f"not enough validation pairs ({rep})"
    return (f"threshold {rep['threshold']:.3f}  genuine_accept {100 * rep['genuine_accept']:5.1f} %  "
            f"eer {100 * rep['eer']:4.1f} %  (same worker median {rep['genuine_median']:.3f}, different workers "
            f"median {rep['impostor_median']:.3f}; {int(rep['n_genuine'])} / {int(rep['n_impostor'])} pairs)")


def train(args) -> int:
    import torch

    from app.core.config import load_settings
    from app.identity.reid_model import resolve_reid_weights

    cfg = load_settings(args.config)
    r = cfg["reid"]
    model_name = args.model_name or r["model_name"]
    weights = None if args.weights == "none" else (args.weights or resolve_reid_weights(model_name, r["weights_path"]))
    head_w, color_w = float(r["head_branch_weight"]), float(r["color_branch_weight"])
    body_w = max(0.30, 1.0 - head_w - color_w)
    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    data = MinedData.load(args.data)
    train_d, val_d = data.split_by_video(args.val_fraction, args.seed)
    print(f"data: {len(data.segments)} segments, {len(data.groups)} co-occurrence groups | "
          f"train {len(train_d.segments)} / {len(train_d.groups)}  val {len(val_d.segments)} / {len(val_d.groups)}")
    if not train_d.groups and not args.eval_only:
        print("No co-occurrence groups to train on: the videos must show several workers at the same time.")
        return 2

    model = build_model(model_name, weights, device)
    base = evaluate(model, val_d, device, body_w, head_w)
    print(f"[before] {fmt(base)}")
    if args.eval_only:
        return 0

    reference = {n: p.detach().clone() for n, p in model.named_parameters()}
    params = [p for n, p in model.named_parameters() if not n.startswith("classifier")]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=5e-4)
    best_rep, best_state = base, copy.deepcopy(model.state_dict())
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for _ in range(args.iters):
            batch = sample_group_batch(train_d, rng, args.groups, args.p, args.k)
            loss = 0.0
            for kind in (0, 1):                      # body images, then upper-body images
                imgs, labels, group_of = [], [], []
                for gi, group in enumerate(batch):
                    for si, (seg, idx) in enumerate(group):
                        smp = train_d.segments[seg]
                        for j in idx:
                            imgs.append(load_image(str(train_d.root / smp[j][kind]), True, rng))
                            labels.append(si)
                            group_of.append(gi)
                x = torch.from_numpy(np.stack(imgs)).to(device)
                out = model(x)
                feats = out[1] if isinstance(out, tuple) else out
                lab, grp = torch.tensor(labels, device=device), torch.tensor(group_of, device=device)
                # triplets only INSIDE a co-occurrence group (segments of different moments may be one person)
                parts = [batch_hard_triplet(feats[grp == gi], lab[grp == gi], args.margin) for gi in range(len(batch))]
                loss = loss + torch.stack(parts).mean()
            loss = loss + args.l2sp * l2sp_penalty(model, reference)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        rep = evaluate(model, val_d, device, body_w, head_w)
        print(f"[epoch {epoch:2d}] loss {np.mean(losses):.4f}  {fmt(rep)}  ({time.time() - t0:.0f} s)")
        if rep.get("genuine_accept", -1) > best_rep.get("genuine_accept", -1):
            best_rep, best_state = rep, copy.deepcopy(model.state_dict())

    out = Path(args.out or f"weights/{model_name}_site.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    if best_state is None or best_rep is base:
        print("No epoch improved the held-out separation: the original weights stay the better choice. "
              "(More / more varied video usually helps.)")
        return 1
    torch.save({"state_dict": best_state}, out)
    print(f"\nbefore: {fmt(base)}\nafter:  {fmt(best_rep)}\nsaved {out}\n"
          f"use it:  configs/settings.yaml -> reid: {{model_name: {model_name}, weights_path: {out.as_posix()}}}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fine-tune the Re-ID model on mined site data and measure the gain")
    ap.add_argument("--data", default="data/reid_mining")
    ap.add_argument("--config", default="configs/settings.yaml")
    ap.add_argument("--model-name", default=None)
    ap.add_argument("--weights", default=None, help="start weights (default: the configured ones; 'none' = random)")
    ap.add_argument("--out", default=None, help="default weights/<model>_site.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--iters", type=int, default=200, help="steps per epoch")
    ap.add_argument("--groups", type=int, default=8, help="co-occurrence groups per step")
    ap.add_argument("--p", type=int, default=4, help="people per group")
    ap.add_argument("--k", type=int, default=4, help="images per person")
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--margin", type=float, default=0.3)
    ap.add_argument("--l2sp", type=float, default=1e-3, help="pull towards the pre-trained weights")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true", help="only measure the given weights on the data")
    return train(ap.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
