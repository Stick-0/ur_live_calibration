import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

@dataclass
class PVNetVoteConfig:
    seg_thr: float = 0.5
    theta: float = 0.99
    max_pixels: int = 4000
    num_hypotheses: int = 32
    weight_by_segprob: bool = True
    min_valid_pixels: int = 50
    use_angle_binning: bool = True     
    angle_bin_deg: int = 3              
    min_hypotheses: int = 8 
    min_effective_hyps: int = 4  
    use_neighbor_fallback: bool = True
    neighbor_max_hops: Optional[int] = None  
    on_fail: str = "none"        
    enforce_in_image: bool = True  
    use_neighbor_pixel_trigger: bool = True  
    neighbor_min_pixels: int = 30

def _to_unit(v: torch.Tensor, eps: float = 1e-8):
    # v: (..., 2)
    n = torch.clamp(torch.linalg.norm(v, dim=-1, keepdim=True), min=eps)
    return v / n
def _neighbor_indices(k: int, K: int, max_hops: Optional[int] = None):
    # k-1, k+1, k-2, k+2 ... (자기 자신 제외)
    order = []
    for d in range(1, K):
        if max_hops is not None and d > max_hops: break
        if k - d >= 0: order.append(k - d)
        if k + d < K:  order.append(k + d)
    return order

def _reduce_pixels_by_angle(vec: torch.Tensor,      # (2,H,W)
                            prob: torch.Tensor,     # (H,W) in [0,1]
                            mask: torch.Tensor,     # (H,W) bool
                            angle_bin_deg: Optional[int],
                            weight_by_prob: bool = True):
    """
    반환:
      Pxy:    (M,2)  [x,y]
      Vunit:  (M,2)
      W:      (M,)   (선택적으로 prob)
    M <= ceil(360 / angle_bin_deg)
    """
    device = vec.device
    H, W = prob.shape
    vy = vec[1]
    vx = vec[0]

    # 유효 벡터 필터
    n2 = vx * vx + vy * vy
    valid = mask & torch.isfinite(vx) & torch.isfinite(vy) & (n2 > 1e-10)

    idx = valid.nonzero(as_tuple=False)  # (M0,2) [y,x]
    if idx.numel() == 0:
        return (torch.empty(0, 2, device=device),
                torch.empty(0, 2, device=device),
                torch.empty(0, device=device))

    vx_m = vx[valid]               # (M0,)
    vy_m = vy[valid]
    n_m = torch.sqrt(vx_m * vx_m + vy_m * vy_m).clamp_min(1e-8)
    vx_m = vx_m / n_m
    vy_m = vy_m / n_m

    # 각도(deg) 0~360
    ang = torch.atan2(vy_m, vx_m) * (180.0 / math.pi)   # (-180,180]
    ang = (ang + 360.0) % 360.0

    deg = max(1, int(angle_bin_deg or 1))
    nbins = int(math.ceil(360.0 / deg))
    bin_idx = torch.clamp((ang / deg).floor().long(), 0, nbins - 1)  # (M0,)

    weights = prob[valid] if weight_by_prob else torch.ones_like(ang, device=device)

    # 각 bin마다 prob 최대인 인덱스 뽑기
    chosen = []
    all_idx = torch.arange(bin_idx.numel(), device=device)
    for b in range(nbins):
        sel = (bin_idx == b)
        if not torch.any(sel):
            continue
        cand_idx = all_idx[sel]
        w_cand = weights[sel]
        j = torch.argmax(w_cand)
        i_sel = cand_idx[j]
        chosen.append(i_sel)

    if len(chosen) == 0:
        return (torch.empty(0, 2, device=device),
                torch.empty(0, 2, device=device),
                torch.empty(0, device=device))

    chosen = torch.stack(chosen, dim=0)       # (M,)
    yx = idx[chosen]                          # (M,2) [y,x]
    Pxy = torch.stack([yx[:, 1].float(), yx[:, 0].float()], dim=1)  # (M,2) [x,y]
    Vunit = torch.stack([vx_m[chosen], vy_m[chosen]], dim=1)        # (M,2)
    W = weights[chosen]                                              # (M,)

    return Pxy, Vunit, W

def _fallback_center(H: int, W: int, device):
    center = torch.tensor([(W - 1) / 2.0, (H - 1) / 2.0], device=device)
    mu = center.clone()
    cov = torch.eye(2, device=device) * 1e3
    hyps = center.unsqueeze(0)
    scores = torch.ones(1, device=device)
    return mu, cov, hyps, scores


def _collect_candidates_nobinning(prob, mask, vec_k, cfg: PVNetVoteConfig):
    """기존 방식: 마스크 내 top-k 픽셀 + 단위벡터"""
    device = prob.device
    yx = mask.nonzero(as_tuple=False)  # (M,2)
    M = yx.size(0)
    if M == 0:
        return (torch.empty(0, 2, device=device),
                torch.empty(0, 2, device=device),
                torch.empty(0, device=device))
    # 상위 픽셀 제한
    if M > cfg.max_pixels:
        pvals = prob[mask].flatten()
        topk = min(cfg.max_pixels, pvals.numel())
        idx_sorted = torch.argsort(pvals, descending=True)[:topk]
        yx = yx[idx_sorted]
        M = yx.size(0)

    vx = vec_k[0][yx[:, 0], yx[:, 1]]
    vy = vec_k[1][yx[:, 0], yx[:, 1]]
    v = torch.stack([vx, vy], dim=-1)
    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v = _to_unit(v)  # (M,2)

    Pxy = torch.stack([yx[:, 1].float(), yx[:, 0].float()], dim=1)  # (M,2)
    Wp = prob[yx[:, 0], yx[:, 1]] if cfg.weight_by_segprob else torch.ones(M, device=device)
    return Pxy, v, Wp


def _intersections_from_pairs(Pxy, Vunit, pair_ij, eps=1e-8):
    # Pxy:   (M,2), Vunit: (M,2), pair_ij: (Nh,2) int64
    pi = Pxy[pair_ij[:,0]]         # (Nh,2)
    vi = Vunit[pair_ij[:,0]]       # (Nh,2)
    pj = Pxy[pair_ij[:,1]]         # (Nh,2)
    vj = Vunit[pair_ij[:,1]]       # (Nh,2)
    # 2D cross(a,b) = a_x b_y - a_y b_x
    def cross(a,b): return a[:,0]*b[:,1] - a[:,1]*b[:,0]
    det = cross(vi, vj)                                        # (Nh,)
    ok  = torch.isfinite(det) & (det.abs() >= eps)
    # t = cross((pj - pi), vj) / cross(vi, vj)
    t = cross(pj - pi, vj)
    t = torch.where(ok, t / det, torch.zeros_like(t))
    h = pi + t.unsqueeze(-1) * vi                              # (Nh,2)
    ok = ok & torch.isfinite(h).all(dim=1)
    return h, ok
def pvnet_vote_single_k(seg_k: torch.Tensor, vec_k: torch.Tensor, cfg: PVNetVoteConfig):
    device = seg_k.device
    H, W = seg_k.shape

    seg_k = torch.nan_to_num(seg_k, nan=0.0, posinf=50.0, neginf=-50.0)
    vec_k = torch.nan_to_num(vec_k, nan=0.0, posinf=0.0, neginf=0.0)

    prob = torch.sigmoid(seg_k).clamp_(0, 1)
    mask = (prob >= cfg.seg_thr)

    if mask.sum() < cfg.min_valid_pixels:
        flat = prob.view(-1)
        topk = min(cfg.min_valid_pixels, flat.numel())
        top_idx = torch.topk(flat, topk).indices
        mask = torch.zeros_like(prob, dtype=torch.bool)
        mask.view(-1)[top_idx] = True

    # 후보 축소
    if cfg.use_angle_binning:
        Pxy, Vunit, Wp = _reduce_pixels_by_angle(
            vec_k, prob, mask,
            angle_bin_deg=cfg.angle_bin_deg,
            weight_by_prob=cfg.weight_by_segprob
        )
        if Pxy.size(0) < 2:
            Pxy, Vunit, Wp = _collect_candidates_nobinning(prob, mask, vec_k, cfg)
    else:
        Pxy, Vunit, Wp = _collect_candidates_nobinning(prob, mask, vec_k, cfg)

    # 실패 케이스 처리 (가설 생성 불가)
    if Pxy.size(0) < 2:
        # center로 채우되 valid=False로 표시
        mu, cov, hyps, scores = _fallback_center(H, W, device)
        return mu, cov, hyps, scores, torch.tensor(False, device=device)

    M = Pxy.size(0)
    Nh = cfg.num_hypotheses
    hyps = None
    scores = None

    max_pairs = M * (M - 1) // 2
    if max_pairs <= Nh:
        ii, jj = torch.triu_indices(M, M, offset=1, device=Pxy.device)
        pairs = torch.stack([ii, jj], dim=1)
        h_all, ok = _intersections_from_pairs(Pxy, Vunit, pairs)
        pairs = pairs[ok]; h_all = h_all[ok]
        if h_all.numel() == 0:
            mu, cov, hyps, scores = _fallback_center(H, W, device)
            return mu, cov, hyps, scores, torch.tensor(False, device=device)

        U = F.normalize(h_all[:, None, :] - Pxy[None, :, :], dim=-1)
        cos = (U * Vunit[None, :, :]).sum(dim=-1)


        w_inl = (cos >= cfg.theta).float()
        s_all = (w_inl * Wp.unsqueeze(0)).sum(dim=1) if cfg.weight_by_segprob else w_inl.sum(dim=1)

        if s_all.numel() > Nh:
            top = torch.topk(s_all, Nh)
            h_all = h_all[top.indices]
            s_all = top.values

        hyps, scores = h_all, s_all
    else:
        over = int(Nh * 2.5)
        i = torch.randint(0, M, (over,), device=Pxy.device)
        j = torch.randint(0, M-1, (over,), device=Pxy.device)
        j = j + (j >= i)
        pairs = torch.stack([i, j], dim=1)

        cos_min = math.cos(math.radians(8.0))
        sim = (Vunit[i] * Vunit[j]).sum(dim=1)
        keep = sim <= cos_min
        pairs = pairs[keep]
        if pairs.size(0) == 0:
            mu, cov, hyps, scores = _fallback_center(H, W, device)
            return mu, cov, hyps, scores, torch.tensor(False, device=device)

        h_all, ok = _intersections_from_pairs(Pxy, Vunit, pairs)
        h_all = h_all[ok]
        if h_all.size(0) == 0:
            mu, cov, hyps, scores = _fallback_center(H, W, device)
            return mu, cov, hyps, scores, torch.tensor(False, device=device)

        U = F.normalize(h_all[:, None, :] - Pxy[None, :, :], dim=-1)
        cos = (U * Vunit[None, :, :]).sum(dim=-1)

        w_inl = (cos >= cfg.theta).float()
        if cfg.weight_by_segprob:
            s_all = (w_inl * Wp.unsqueeze(0)).sum(dim=1)
        else:
            s_all = w_inl.sum(dim=1)

        if s_all.numel() > Nh:
            top = torch.topk(s_all, Nh)
            hyps = h_all[top.indices]
            scores = top.values
        else:
            hyps = h_all
            scores = s_all

    if hyps is None or hyps.size(0) == 0:
        mu, cov, hyps, scores = _fallback_center(H, W, device)
        return mu, cov, hyps, scores, torch.tensor(False, device=device)

    # 최소 가설/유효 가설 체크
    if hyps.size(0) < cfg.min_hypotheses:
        mu, cov, hyps2, scores2 = _fallback_center(H, W, device)
        return mu, cov, hyps2, scores2, torch.tensor(False, device=device)
    eff = int((scores > 0).sum().item())
    if eff < cfg.min_effective_hyps:
        mu, cov, hyps2, scores2 = _fallback_center(H, W, device)
        return mu, cov, hyps2, scores2, torch.tensor(False, device=device)

    # 가중 평균/공분산
    wsum = scores.sum().clamp_min(1e-8)
    mu = (hyps * scores.unsqueeze(-1)).sum(dim=0) / wsum
    diff = hyps - mu.unsqueeze(0)
    cov = (scores.view(-1,1,1) * (diff.unsqueeze(-1) @ diff.unsqueeze(-2))).sum(dim=0) / wsum

    # in-image 체크 → 이미지 밖이면 실패로 간주
    in_img = (
        torch.isfinite(mu).all()
        and (mu[0] >= 0) and (mu[0] < W) and (mu[1] >= 0) and (mu[1] < H)
    )
    mu = torch.nan_to_num(mu, nan=(W-1)/2, posinf=(W-1)/2, neginf=(W-1)/2)
    cov = torch.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    valid = torch.tensor(bool(in_img), device=device)

    return mu, cov, hyps, scores, valid

@torch.no_grad()
def pvnet_vote(seg: torch.Tensor, vec: torch.Tensor, cfg: PVNetVoteConfig, return_debug: bool = False):
    B, K, H, W = seg.shape
    device = seg.device

    mus  = torch.zeros(B, K, 2,    device=device)
    covs = torch.zeros(B, K, 2, 2, device=device)
    valid_mask = torch.zeros(B, K, dtype=torch.bool, device=device)

    ragged_hyps   = [] if return_debug else None
    ragged_scores = [] if return_debug else None

    for b in range(B):
        hyps_b = [] if return_debug else None
        scores_b = [] if return_debug else None

        for k in range(K):
            seg_k = seg[b, k]
            vec_k = vec[b, k]

            # ---------- [픽셀수 트리거] ----------
            use_neighbor = False
            neigh_choice = None
            if getattr(cfg, "use_neighbor_pixel_trigger", False):
                prob_k  = torch.sigmoid(torch.nan_to_num(seg_k, nan=0.0, posinf=50.0, neginf=-50.0)).clamp_(0,1)
                mask_k  = (prob_k >= cfg.seg_thr)
                num_k   = int(mask_k.sum().item())

                if num_k < getattr(cfg, "neighbor_min_pixels", cfg.min_valid_pixels):
                    neigh_idxs = _neighbor_indices(k, K, getattr(cfg, "neighbor_max_hops", None))
                    for kn in neigh_idxs:
                        prob_n = torch.sigmoid(torch.nan_to_num(seg[b, kn], nan=0.0, posinf=50.0, neginf=-50.0)).clamp_(0,1)
                        mask_n = (prob_n >= cfg.seg_thr)
                        num_n  = int(mask_n.sum().item())
                        
                        if num_n >= getattr(cfg, "neighbor_min_pixels", cfg.min_valid_pixels):
                            use_neighbor = True
                            neigh_choice = kn
                            # print(f"[vote] use neighbor {kn} for batch {b} keypoint {k} ({num_k} -> {num_n} pixels)")
                            break

            if use_neighbor:
                # 이웃 세그로 교체, 벡터장은 k 그대로
                mu, cov, hyps_t, scores_t, valid = pvnet_vote_single_k(seg[b, neigh_choice], vec_k, cfg)
            else:
                # 원래대로
                mu, cov, hyps_t, scores_t, valid = pvnet_vote_single_k(seg_k, vec_k, cfg)



            mus[b, k]  = mu
            covs[b, k] = cov
            valid_mask[b, k] = valid
            if return_debug:
                hyps_b.append(hyps_t)
                scores_b.append(scores_t)

        if return_debug:
            ragged_hyps.append(hyps_b)
            ragged_scores.append(scores_b)

    extras = {"valid": valid_mask}
    if return_debug:
        extras["hyps"] = ragged_hyps
        extras["scores"] = ragged_scores

    return mus, covs, extras


