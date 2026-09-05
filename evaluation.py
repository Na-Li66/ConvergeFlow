import torch
from tqdm import tqdm, trange
import argparse
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
import numpy as np
import os
from sample import sample
from model import LangFlow, LangFlowConfig

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Continuous-Rivals-Discrete/langflow-owt")
    parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/code/LangFlow/openwebtext")
    parser.add_argument("--config_path", type=str, default="config.json")
    parser.add_argument("--num_samples", type=int, default=1024, help="Total number of samples to generate.")
    parser.add_argument("--NFE", type=int, default=256, help="Number of LangFlow denoising steps.")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size.")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--seq_length", type=int, default=1024, help="Generated sample length in LangFlow token ids.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--time_schedule", type=str, default="ConvergeFlow")
    parser.add_argument("--output_form", type=str, default="weight")
    parser.add_argument("--pred_form", type=str, default="weight")
    parser.add_argument("--config_type", type=int, default=1)
    parser.add_argument("--wk", type=int, default=1)
    parser.add_argument("--wscg", type=float, default=1.0)
    parser.add_argument("--wug", type=float, default=0.0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased", help="Tokenizer used to decode generated LangFlow token ids.")
    parser.add_argument("--ppl_model", type=str, default="gpt2-large", help="Hugging Face causal LM used to score generated text.")
    parser.add_argument("--max_ppl_length", type=int, default=1024, help="Maximum tokenized length for the causal LM.")
    parser.add_argument("--ppl_batch_size", type=int, default=4, help="Batch size for the perplexity scorer.")
    return parser.parse_args()

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


def metric(args, sample_ids, texts, device, ckpt_path=None):
    batch_size = args.ppl_batch_size
    num_samples = args.num_samples
    scorer_tokenizer = AutoTokenizer.from_pretrained(args.ppl_model)
    scorer = AutoModelForCausalLM.from_pretrained(args.ppl_model)
    if scorer_tokenizer.pad_token is None:
        scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer.to(device)
    scorer.eval()

    total_nll = 0.0
    total_tokens = 0
    Entropy = 0
    finished_num = 0
    with torch.inference_mode():
        pbar = tqdm(total=num_samples, desc="Computing Metrics")
        while finished_num < num_samples:
            start = finished_num
            end = min(start + batch_size, num_samples)
            
            batch_texts = texts[start:end]
            encoded = scorer_tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_ppl_length,
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            valid_tokens = int((attention_mask.sum(dim=1) - 1).clamp(min=0).sum().item())
            labels = input_ids.masked_fill(attention_mask == 0, -100)
            outputs = scorer(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            nll = outputs.loss.item() * valid_tokens

            total_nll += nll
            total_tokens += valid_tokens
            
            for i in range(start, end):
                ids = sample_ids[i]
                _, counts = torch.unique(ids, return_counts=True)
                probs = counts.float() / counts.sum()
                Entropy += -(probs * torch.log(probs)).sum()
            finished_num += end - start
            pbar.update(end - start)
        pbar.close()

    NLL = total_nll / total_tokens
    PPL = torch.exp(torch.tensor(NLL)).item()
    Entropy = Entropy / num_samples
    return NLL, PPL, Entropy

def load_model(args, device):
    # Load model:
    config = LangFlowConfig.from_pretrained(args.config_path)
    model = LangFlow(config, args.output_form)

    if args.ckpt_path == "None":
        model = LangFlow.from_pretrained(args.model)
    else:
        _, checkpoint = find_model(args.ckpt_path)
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    args = parse_args()
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = args.seed
    seed_everything(seed)
    
    # load model
    model = load_model(args, device)
    
    # sample
    seq_length = args.seq_length
    embed_dim = model.config.hidden_size
    batch_size = args.batch_size
    sample_ids_all = []
        
    for i in trange(args.num_samples // args.batch_size):
        sample_ids = sample(model, args, device)
        sample_ids_all.append(sample_ids)
    sample_ids_all = torch.cat(sample_ids_all, dim=0)
    
    # compute PPL and Entropy
    generator_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if generator_tokenizer.pad_token is None:
        generator_tokenizer.pad_token = generator_tokenizer.eos_token
    texts = [generator_tokenizer.decode(ids.tolist(), skip_special_tokens=False) for ids in sample_ids_all.detach().cpu()]
    print(texts[0])
    del generator_tokenizer
    
    NLL, PPL, Entropy = metric(args, sample_ids_all, texts, device, ckpt_path=None)
    print(f"Perplexity: {PPL:.4f}")
    print(f"Negative Log-Likelihood: {NLL:.4f}")
    print(f"Sample Entropy: {Entropy:.4f}")
    print("Done.")