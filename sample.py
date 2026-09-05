import torch
import torch.nn.functional as F
import math, os

def adjust_by_ceil_distance(k_sequence, raw_values, current_total, target_total):
    if current_total >= target_total:
        return k_sequence
    need_increase = target_total - current_total
    adjusted_k = k_sequence.copy()
    fractional_parts = []
    for _, raw in enumerate(raw_values):
        frac = raw - torch.floor(torch.tensor(raw)).item()
        fractional_parts.append(frac)
    indices_sorted = sorted(range(len(adjusted_k)), key=lambda i: fractional_parts[i], reverse=True)
    increased_count = 0
    for idx in indices_sorted:
        if need_increase <= 0:
            break
        adjusted_k[idx] += 1
        need_increase -= 1
        increased_count += 1
    return adjusted_k

def find_k_sequence(w_k, NFE, get_alpha_sigma_func, device='cpu', configuration=3):
    if not isinstance(NFE, int):
        NFE = int(NFE)
    if not isinstance(w_k, torch.Tensor):
        w_k = torch.tensor(float(w_k), dtype=torch.float32, device=device)
    else:
        w_k = w_k.float().to(device)
    
    low, high = 1, NFE
    
    best_N = None
    best_k_seq = None
    best_total = 0
    
    while low <= high:
        mid = (low + high) // 2
        
        t = (torch.linspace(1, 2 * mid - 1, int(mid), device=device)/(2 * mid)).flip(0)
        alpha_seq, sigma_seq = get_alpha_sigma_func(t)
        
        alpha_seq = alpha_seq.float().to(device)
        sigma_seq = sigma_seq.float().to(device)
        if configuration == 2:
            denominator = 1 + torch.sqrt(sigma_seq/alpha_seq)
        elif configuration == 3:
            denominator = 1 + sigma_seq/alpha_seq
        raw_values = w_k / denominator
        
        k_float = torch.ceil(raw_values)
        k_seq = k_float.to(torch.int64)
        
        total_float = torch.sum(k_float + 1)
        total = int(total_float.item())
        
        if total == NFE:
            return int(mid), k_seq.cpu().tolist()
        elif total < NFE:
            best_N = int(mid)
            best_k_seq = k_seq.cpu().tolist()
            best_raw_values = raw_values.cpu().tolist()
            best_total = total
            low = mid + 1
        else:
            high = mid - 1
    
    while best_total < NFE:
        best_k_seq = adjust_by_ceil_distance(best_k_seq, best_raw_values, best_total, NFE)
        best_total = sum(k + 1 for k in best_k_seq)
    return int(best_N), best_k_seq

def get_alpha_sigma(t: torch.Tensor) -> torch.Tensor:
    gamma = torch.distributions.Gumbel(4.9563, 0.9637).icdf(t)
    alphas = torch.sqrt(torch.sigmoid(-gamma)).double()
    sigmas = torch.sqrt(torch.sigmoid(gamma)).double()
    return alphas, sigmas

def find_model(ckpt_path):
    assert os.path.isfile(ckpt_path), f'Could not find checkpoint at {ckpt_path}'
    checkpoint = torch.load(
        ckpt_path,
        map_location=lambda storage, loc: storage,
        weights_only=False,
    )
    if "ema" in checkpoint:  # supports checkpoints from train.py
        ema_checkpoint = checkpoint["ema"]
        checkpoint = checkpoint["model"]
    return ema_checkpoint, checkpoint

def proposal_compute(t: torch.Tensor) -> torch.Tensor:
    gamma = torch.distributions.Gumbel(4.9563, 0.9637).icdf(t)
    gamma_min = float(4.9563 - math.log(-math.log(1e-5)) * 0.9637)
    gamma_max = float(4.9563 - math.log(1e-5) * 0.9637)
    alphas = torch.sqrt(torch.sigmoid(-gamma)).double()
    sigmas = torch.sqrt(torch.sigmoid(gamma)).double()
    return gamma.clamp(min=gamma_min, max=gamma_max), alphas, sigmas

def find_closest_token(model, z, alpha=1, sigma=1):
    device = z.device
    D = z.size(-1)
    original_shape = z.shape[:-1]
    all_ids = torch.arange(model.config.vocab_size, device=device)
    z_flat = z.double().reshape(-1, D)
    X0 = model._embed_tokens(all_ids.unsqueeze(0)).squeeze(0).double()

    z_norm2 = (z_flat ** 2).sum(dim=1)
    x0_norm2 = (X0 ** 2).sum(dim=1)
    dist_compute = z_flat @ X0.T
    dist_compute.mul_(-2 * alpha)
    dist_compute += z_norm2[:, None]
    dist_compute += (alpha ** 2) * x0_norm2[None, :]
    dist_compute.clamp_min_(0)
    dist_compute = dist_compute / (D * sigma ** 2)

    closest_token_indices = dist_compute.argmin(dim=1)
    closest_tokens = all_ids[closest_token_indices]
    closest_tokens = closest_tokens.reshape(original_shape)
    return closest_tokens

def compute_mu(model, z, gamma_expanded, output_form, x_self_cond=None):
    if output_form == "weight":
        logits = model(
            noisy_embeds=z.float(),
            timesteps=gamma_expanded,
            x_self_cond=x_self_cond,
            return_dict=False).double()
        probs = F.softmax(logits.float(), dim=-1)
        mu = model._embed_tokens(probs)
        return mu
    elif output_form == "distance":
        mu = model(
            noisy_embeds=z.float(),
            timesteps=gamma_expanded,
            x_self_cond=x_self_cond,
            return_dict=False).double()
        return mu
    else:
        raise ValueError(f"The input output form {output_form} is not supported!")

def sample(model, args, device):
    torch.set_default_dtype(torch.float64)
    seq_length = args.seq_length
    embed_dim = model.config.hidden_size
    wscg = args.wscg
    batch_size = args.batch_size
    output_form = args.output_form
    config_type = args.config_type
    if config_type == 1:
        N = torch.ceil(torch.tensor(args.NFE/(args.wk + 1))).int().item()
        remain_k = N * (args.wk + 1) - args.NFE
        k_seq = torch.tensor([args.wk - 1 if i < remain_k else args.wk for i in range(N)])
    else:
        N, k_seq = find_k_sequence(args.wk, args.NFE, get_alpha_sigma, device, args.config_type)
    
    z = torch.randn(batch_size, seq_length, embed_dim, device=device)
    if args.time_schedule == "LangFlow":
        eps = 1e-5
        t = torch.linspace(1.0 - eps, eps, N, device=device)
    elif args.time_schedule == "ConvergeFlow":
        t = (torch.linspace(1, 2 * N - 1, N, device=device)/(2 * N)).flip(0)
    else:
        raise ValueError(f"The input time_schedule {args.time_schedule} is not supported!")
    
    gamma = model.proposal(t)
    for i in range(len(gamma) - 1):
        gamma_t = gamma[i]
        gamma_s = gamma[i+1]
        gamma_expanded = gamma_t.unsqueeze(0).expand(batch_size)
        
        s_view = gamma_s.view(1, 1, 1).double()
        alpha_s = torch.sqrt(torch.sigmoid(-s_view))
        sigma_s = torch.sqrt(torch.sigmoid(s_view))
        t_view = gamma_t.view(1, 1, 1).double()
        alpha_t = torch.sqrt(torch.sigmoid(-t_view))
        sigma_t = torch.sqrt(torch.sigmoid(t_view))
        
        mu_uncond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=None)
        if wscg > 0:
            x_self_cond = mu_uncond
            k = k_seq[i]
                
            if args.config_type == 1:
                for _ in range(k):
                    mu_cond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=x_self_cond)
                    x_pred = (1 - wscg) * mu_uncond + wscg * mu_cond
                    x_self_cond = x_pred
            elif args.config_type == 2:
                for _ in range(k):
                    mu_cond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=x_self_cond)
                    x_pred = (1 - wscg/(1 + torch.sqrt(sigma_t/alpha_t))) * mu_uncond.double() + wscg * mu_cond.double()/(1 + torch.sqrt(sigma_t/alpha_t))
                    x_self_cond = x_pred.float()
            elif args.config_type == 3:
                for _ in range(k):
                    mu_cond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=x_self_cond)
                    x_pred = (1 - wscg/(1 + sigma_t/alpha_t)) * mu_uncond.double() + wscg * mu_cond.double()/(1 + sigma_t/alpha_t)
                    x_self_cond = x_pred.float()
            else:
                raise ValueError(f"The input wscg type {args.config_type} is not supported!")
                
        else:
            x_pred = mu_uncond

        eps_pred = (z - alpha_t * x_pred) / sigma_t
        
        if args.config_type == 1:
            z = alpha_s / alpha_t * z + alpha_s * (1 + args.wug) * (sigma_s/alpha_s - sigma_t/alpha_t) * eps_pred
        elif args.config_type == 2:
            z = alpha_s / alpha_t * z + alpha_s * (1 + args.wug/(1 + torch.sqrt(sigma_t/alpha_t))) * (sigma_s/alpha_s - sigma_t/alpha_t) * eps_pred
        elif args.config_type == 3:
            z = alpha_s / alpha_t * z + alpha_s * (1 + args.wug/(1 + sigma_t/alpha_t)) * (sigma_s/alpha_s - sigma_t/alpha_t) * eps_pred
        else:
            raise ValueError(f"The input wug type {args.config_type} is not supported!")

    # Final step: get logits and take argmax
    gamma_final = gamma[-1]
    gamma_expanded = gamma_final.unsqueeze(0).expand(batch_size)
    mu_uncond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=None)
    x_self_cond = mu_uncond
    k = k_seq[-1]
        
    if args.config_type == 1:
        for _ in range(k - 1):
            mu_cond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=x_self_cond)
            x_pred = (1 - wscg) * mu_uncond + wscg * mu_cond
            x_self_cond = x_pred
    elif args.config_type == 2:
        for _ in range(k - 1):
            mu_cond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=x_self_cond)
            x_pred = (1 - wscg/(1 + torch.sqrt(sigma_s/alpha_s))) * mu_uncond.double() + wscg * mu_cond.double()/(1 + torch.sqrt(sigma_s/alpha_s))
            x_self_cond = x_pred.float()
    elif args.config_type == 3:
        for _ in range(k - 1):
            mu_cond = compute_mu(model, z, gamma_expanded, output_form, x_self_cond=x_self_cond)
            x_pred = (1 - wscg/(1 + sigma_s/alpha_s)) * mu_uncond.double() + wscg * mu_cond.double()/(1 + sigma_s/alpha_s)
            x_self_cond = x_pred.float()
    else:
        raise ValueError(f"The input wscg type {args.config_type} is not supported!")
            
    logits = model(
        noisy_embeds=z.float(),
        timesteps=gamma_expanded,
        x_self_cond=x_self_cond.float(),
        return_dict=False).double()
            
    if args.pred_form == "distance":
        probs = F.softmax(logits.float(), dim=-1)
        mu = model._embed_tokens(probs)
        samples = find_closest_token(model, mu)
    elif args.pred_form == "weight":
        samples = logits.argmax(dim=-1)
    return samples