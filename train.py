import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import os
import datetime
import numpy as np
from glob import glob
from time import time
from copy import deepcopy
from tqdm import tqdm
import argparse
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from collections import OrderedDict
from model import LangFlow, LangFlowConfig
import torch.nn as nn
import json

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def cleanup():
    dist.destroy_process_group()

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

class OpenWebTextDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, block_size, seed=42, split="train"):
        if split == 'train':
            self.data_path = os.path.join(data_dir, 'train.bin')
        else:
            self.data_path = os.path.join(data_dir, 'val.bin')
        self.data_path = self.data_path
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        data = np.memmap(self.data_path, dtype=np.uint16, mode="r")
        rng = np.random.default_rng(self.seed + dist.get_rank())

        while True:
            idx = rng.integers(0, len(data) - self.block_size - 1)
            chunk = np.array(data[idx:idx + self.block_size + 1], dtype=np.int64)
            x = torch.from_numpy(chunk[:-1])
            y = torch.from_numpy(chunk[1:])
            yield x, y

def sample_t_across_gpus(B, grad_accu, world_size, device, eps=1e-5):
    rank = dist.get_rank()
    B_per_gpu_per_step = B//world_size//grad_accu
    if rank == 0:
        t_base = torch.empty(B, device=device).uniform_(0.0, 1.0)
        t_indices = torch.arange(t_base.size(0), device=t_base.device).float()
        t_whole = eps + (1 - 2 * eps) * (t_indices + t_base)/B
        shuffle_indices = torch.randperm(B, device=device)
        t_all = t_whole[shuffle_indices]
    else:
        t_all = torch.empty(B, device=device)

    dist.broadcast(t_all, src=0)
    samples_per_gpu = B_per_gpu_per_step * grad_accu
    start_idx = rank * samples_per_gpu
    end_idx = start_idx + samples_per_gpu
    t_gpu = t_all[start_idx:end_idx]
    return t_gpu.view(grad_accu, B_per_gpu_per_step)

class LangFlowDDPWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, t, grad_accu):
        loss = self.model.compute_loss_condition(x, y, t, training_prob=0.25) / grad_accu
        return loss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Continuous-Rivals-Discrete/langflow-owt")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/code/LangFlow/openwebtext")
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=25_000)
    parser.add_argument("--ckpt_path", type=str, default=None)
    
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--batch_size", type=int, default=480, help="Generation batch size.")
    parser.add_argument("--grad_accu", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--embed_trainable", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    assert args.batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.set_device(device)
    torch.backends.cudnn.deterministic = True
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        time_name = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{time_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info("Arguments:\n%s", json.dumps(vars(args), indent=4, ensure_ascii=False, default=str))
    else:
        logger = create_logger(None)

    # Create model:
    config = LangFlowConfig.from_pretrained(args.config_path)
    model = LangFlow(config)
    
    hf_model = LangFlow.from_pretrained(args.model)
    model.load_state_dict(hf_model.state_dict())
    del hf_model
    train_model = LangFlowDDPWrapper(model)
    train_model = DDP(train_model.to(device), device_ids=[device], find_unused_parameters=True)

    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    requires_grad(model.proposal)
    if args.embed_trainable == False:
        model.backbone.vocab_embed.embedding.requires_grad = False
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f"model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Setup optimizer:
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, train_model.parameters()),
        lr=args.lr,
        weight_decay=0,
    )

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=f"cuda:{rank}", weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        ema.load_state_dict(ckpt["ema"])
        
        # opt.load_state_dict(ckpt["opt"])
        old_optimizer_state = ckpt['opt']
        new_optimizer_state = opt.state_dict()
        old_group = old_optimizer_state['param_groups'][0]
        new_group = new_optimizer_state['param_groups'][0]
        new_group['lr'] = old_group['lr']
        new_group['weight_decay'] = old_group.get('weight_decay', 0)
        new_group['betas'] = old_group.get('betas', (0.9, 0.999))
        new_group['eps'] = old_group.get('eps', 1e-8)
        old_state = old_optimizer_state['state']
        new_state = new_optimizer_state['state']
        old_params = []
        for group in old_optimizer_state['param_groups']:
            for param_id in group['params']:
                old_params.append(param_id)
        new_params = []
        for group in new_optimizer_state['param_groups']:
            for param_id in group['params']:
                new_params.append(param_id)
        for old_id, new_id in zip(old_params, new_params):
            if new_id in new_state:
                new_state[new_id] = old_state[old_id]
        opt.load_state_dict(new_optimizer_state)
        
        train_steps = int(os.path.basename(args.ckpt_path).replace(".pt", ""))
        logger.info(f"Resumed from checkpoint {args.ckpt_path}, step={train_steps}")
        
    else:
        train_steps = 0

    # Setup data:
    dataset = OpenWebTextDataset(args.data_path, args.block_size, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size // dist.get_world_size() // args.grad_accu),
        num_workers=0,
        pin_memory=False,
    )

    # Prepare models for training:
    if args.ckpt_path is None:
        update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    eps = args.eps
    accu_idx = 0
    start_time = time()
    
    t_all_gpus = sample_t_across_gpus(B=args.batch_size, grad_accu=args.grad_accu, world_size=dist.get_world_size(), device=device, eps=eps)
    dist.barrier()
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)
            t = t_all_gpus[accu_idx]
            loss = train_model(x, y, t, 0.25, args.grad_accu)
                    
            running_loss += loss.item()
            loss.backward()
            accu_idx += 1
            if accu_idx == args.grad_accu:
                opt.step()
                opt.zero_grad(set_to_none=True)
                update_ema(ema, model.module)
                accu_idx = 0
    
                t_all_gpus = sample_t_across_gpus(B=args.batch_size, grad_accu=args.grad_accu, world_size=dist.get_world_size(), device=device, eps=eps)
                # Log loss values:
                log_steps += 1
                train_steps += 1
                if train_steps % args.log_every == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                # Save checkpoint:
                if train_steps % args.ckpt_every == 0 and train_steps > 0:
                    if rank == 0:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                        torch.save(checkpoint, checkpoint_path)
                        logger.info(f"Saved checkpoint to {checkpoint_path}")
                    dist.barrier()

    logger.info("Done!")
    cleanup()
