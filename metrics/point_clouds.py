# pytorch re-implementation of
# https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
# https://github.com/stevenygd/PointFlow/blob/master/metrics/evaluation_metrics.py

import torch
from tqdm import tqdm

from metrics.distance.cd.chamfer_distance import chamfer_distance
from metrics.distance.emd.earth_mover_distance import earth_mover_distance


def _compute_emd(pcs_1, pcs_2):
    B, N_1, N_2 = pcs_1.size(0), pcs_1.size(1), pcs_2.size(1)
    assert N_1 == N_2
    emd = earth_mover_distance(pcs_1, pcs_2)  # (B,)
    emd_norm = emd / float(N_1)  # (B,)
    return emd_norm


def _compute_cd(pcs_1, pcs_2):
    dl, dr = chamfer_distance(pcs_1, pcs_2)
    return dl.mean(dim=1) + dr.mean(dim=1)


def _pairwise_distance(pcs_1, pcs_2, batch_size, metrics=("cd", "emd"), verbose=True):
    B_1 = pcs_1.size(0)
    B_2 = pcs_2.size(0)
    device = pcs_1.device

    distance = {}
    for key in metrics:
        distance[key] = torch.zeros(B_1, B_2, device=device)

    for i in tqdm(
        range(B_1),
        desc="distance matrix {}".format(str(metrics)),
        leave=False,
        disable=not verbose,
    ):
        for j in range(0, B_2, batch_size):
            # The size of 'batch_2' may be not 'batch_size'
            batch_2 = pcs_2[j : j + batch_size]
            batch_1 = pcs_1[[i]].expand(batch_2.size(0), -1, -1)

            if "cd" in metrics:
                dist_cd = _compute_cd(batch_1, batch_2)
                distance["cd"][i, j : j + batch_size] = dist_cd
            if "emd" in metrics:
                dist_emd = _compute_emd(batch_1, batch_2)
                distance["emd"][i, j : j + batch_size] = dist_emd

    return distance


def _compute_cov_mmd(M_rg):
    N_ref, N_gen = M_rg.shape
    mmd_gen, min_idx_gen = M_rg.min(dim=0)
    mmd_ref, _ = M_rg.min(dim=1)
    mmd = mmd_ref.mean().item()
    mmd_gen = mmd_gen.mean().item()
    cov = float(len(torch.unique(min_idx_gen))) / float(N_ref)
    return {
        "mmd": mmd,
        "mmd(sample)": mmd_gen,
        "cov": cov,
    }


def _compute_nna(M_rr, M_rg, M_gg, k, sqrt=False):
    N_ref, N_gen = M_rg.shape
    device = M_rg.device

    label_ref = torch.ones(N_ref, device=device)
    label_gen = torch.zeros(N_gen, device=device)
    label = torch.cat([label_ref, label_gen], dim=0)

    # matrix for leave-one-out
    M_ref = torch.cat((M_rr, M_rg), dim=1)
    M_gen = torch.cat((M_rg.t(), M_gg), dim=1)
    M = torch.cat([M_ref, M_gen], dim=0)  # (N_r+N_g, N_r+N_g)
    M = M.abs().sqrt() if sqrt else M
    M = M + torch.diag(float("inf") * torch.ones_like(label))
    _, idx = M.topk(k=k, dim=0, largest=False)  # idx.shape is (k, N_r+N_g)

    # vote & classify
    count = torch.zeros_like(label)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = (count / k >= 0.5).float()

    s = {
        "tp": (pred * label).sum().item(),
        "fp": (pred * (1 - label)).sum().item(),
        "fn": ((1 - pred) * label).sum().item(),
        "tn": ((1 - pred) * (1 - label)).sum().item(),
    }

    s.update(
        {
            "precision": s["tp"] / (s["tp"] + s["fp"] + 1e-10),
            "recall": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "accuracy_t": s["tp"] / (s["tp"] + s["fn"] + 1e-10),
            "accuracy_f": s["tn"] / (s["tn"] + s["fp"] + 1e-10),
            "accuracy": torch.eq(label, pred).float().mean().item(),
        }
    )
    return s


@torch.no_grad()
def compute_cov_mmd_1nna(
    pcs_gen, pcs_ref, batch_size, metrics=("cd", "emd"), verbose=True
):
    assert isinstance(metrics, tuple)
    results = {}

    M_rr = _pairwise_distance(pcs_ref, pcs_ref, batch_size, metrics, verbose)
    M_rg = _pairwise_distance(pcs_ref, pcs_gen, batch_size, metrics, verbose)
    M_gg = _pairwise_distance(pcs_gen, pcs_gen, batch_size, metrics, verbose)

    for metric in metrics:
        # COV and MMD
        scores_mmd_cov = _compute_cov_mmd(M_rg[metric])

        for k, v in scores_mmd_cov.items():
            results.update({"{}-{}".format(k, metric): v})

        # 1-NNA
        scores_1nna = _compute_nna(
            M_rr[metric],
            M_rg[metric],
            M_gg[metric],
            k=1,
            sqrt=False,
        )

        for k, v in scores_1nna.items():
            results.update({"1-nn-{}-{}".format(k, metric): v})

    return results


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    a = torch.rand(100, 2048, 3).cuda()
    b = torch.rand_like(a)
    r = compute_cov_mmd_1nna(a, b, 512, ("cd",))
    for k, v in r.items():
        print(k.rjust(20), v)
