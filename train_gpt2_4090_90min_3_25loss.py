import os
import sys
import glob
import math
import time
import uuid
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import torch.distributed as dist

# -----------------------------------------------------------------------------
# 1. Environment & Performance Setup
# -----------------------------------------------------------------------------
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_LOGS"] = "+dynamo"
os.environ["TORCHDYNAMO_VERBOSE"] = "1"

try:
    from torchao.float8 import convert_to_float8_training
    print("Successfully imported torchao. FP8 training enabled.")
except ImportError:
    print("Install torchao nightly to enable FP8: pip install --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu124")
    sys.exit(1)

import torch._dynamo as dynamo
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

# Set recompile limit slightly higher for dynamic window sizing
dynamo.config.recompile_limit = 128
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

# -----------------------------------------------------------------------------
# 2. Optimizer (Muon)
# -----------------------------------------------------------------------------

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix (orthogonalization).
    """
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): 
        X = X.T
    
    # Constants for Newton-Schulz 5
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.size(0) > G.size(1): 
        X = X.T
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized optimizer. 
    Designed for massive throughput on matrix multiplications.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, backend_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, backend_steps=backend_steps))

    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p.grad)
                
                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(p.grad)
                
                g = p.grad.add(buf, alpha=group['momentum']) if group['nesterov'] else buf
                g = zeropower_via_newtonschulz5(g, steps=group['backend_steps'])
                g *= max(1, g.size(0) / g.size(1)) ** 0.5
                
                p.data.add_(g, alpha=-group['lr'])

# -----------------------------------------------------------------------------
# 3. Model Architecture (Modded-GPT)
# -----------------------------------------------------------------------------

def norm(x: Tensor): 
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features, out_features): 
        super().__init__(in_features, out_features, bias=False)
        
    def reset_parameters(self): 
        # Initialize to zero; essential for the "identity-init" strategy
        with torch.no_grad(): 
            self.weight.zero_()
            
    def forward(self, x): 
        return F.linear(x, self.weight, self.bias)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).float()
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=x.device).float() / self.dim))
            freqs = torch.outer(t, inv_freq)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
            
        cos, sin = self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]
        d = x.shape[3] // 2
        return torch.cat([x[..., :d] * cos + x[..., d:] * sin, x[..., :d] * (-sin) + x[..., d:] * cos], 3).type_as(x)

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = 1 # MQA
        self.head_dim = config.n_embd // config.n_head
        
        self.c_q = CastedLinear(config.n_embd, config.n_embd)
        self.c_k = CastedLinear(config.n_embd, self.n_kv_head * self.head_dim)
        self.c_v = CastedLinear(config.n_embd, self.n_kv_head * self.head_dim)
        self.val_proj = CastedLinear(config.n_embd, self.n_kv_head * self.head_dim)
        
        # Initialize projections
        with torch.no_grad():
            std = 0.5 * (config.n_embd ** -0.5)
            bound = (3 ** 0.5) * std
            self.c_q.weight.uniform_(-bound, bound)
            self.c_k.weight.uniform_(-bound, bound)
            self.c_v.weight.uniform_(-bound, bound)
            self.val_proj.weight.uniform_(-bound, bound)
            
        self.lamb = nn.Parameter(torch.tensor(0.5))
        self.rotary = Rotary(self.head_dim)
        self.c_proj = CastedLinear(config.n_embd, config.n_embd)
        self.attn_gate = CastedLinear(12, config.n_head)
        self.flex_kernel_options = config.flex_kernel_options

    def forward(self, x, v1, block_mask, attn_scale=0.1):
        B, T = x.size(0), x.size(1)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        
        # Value embedding injection
        if v1 is None: 
            v1 = v 
        else:
            if v1.size(-1) != self.head_dim:
                v1 = self.val_proj(v1).view(B, T, self.n_kv_head, self.head_dim)
        
        v = (1 - self.lamb) * v + self.lamb * v1
        
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        
        # Flex Attention with FP8 compute happens inside if enabled globally, 
        # but inputs here are BF16.
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), 
                           block_mask=block_mask, scale=attn_scale, 
                           kernel_options=self.flex_kernel_options, enable_gqa=True)
        
        y = y.transpose(1, 2).contiguous()
        
        # Gating mechanism
        gate = torch.sigmoid(self.attn_gate(x[..., :self.attn_gate.weight.size(-1)])).unsqueeze(-1)
        y = y * gate
        y = y.view(B, T, -1)
        return self.c_proj(y), v1

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.c_fc = CastedLinear(dim, 4 * dim)
        self.c_proj = CastedLinear(4 * dim, dim)
        with torch.no_grad(): 
            self.c_fc.weight.uniform_(-(3**0.5)*(0.5*dim**-0.5), (3**0.5)*(0.5*dim**-0.5))
            
    def forward(self, x): 
        return self.c_proj(F.relu(self.c_fc(x)).square())

class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # Sparsified attention/mlp blocks for speedrun config
        self.attn = CausalSelfAttention(config) if layer_idx not in [0, 7] else None
        self.mlp = MLP(config.n_embd) if layer_idx != 0 else None
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x, v1, x0, block_mask, attn_scale):
        # U-Net style residuals
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x1, v1 = self.attn(norm(x), v1, block_mask, attn_scale)
            x = x + x1
        if self.mlp is not None:
            x = x + self.mlp(norm(x))
        return x, v1

@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6
    n_embd: int = 768
    flex_kernel_options: dict = None

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_encoder_layers = config.n_layer // 2
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            h=nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        ))
        
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.smear_gate = CastedLinear(12, 1)
        self.value_embeds = nn.ModuleList([nn.Embedding(config.vocab_size, config.n_embd) for _ in range(3)])
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        self.backout_lambda = nn.Parameter(0.5 * torch.ones(1))

    def forward(self, idx, target, attn_blocksize):
        # Generate block mask on the fly - fast on GPU
        docs = (idx == 50256).cumsum(0)
        
        def document_causal_mask(b, h, q_idx, kv_idx):
            return (q_idx >= kv_idx) & (docs[q_idx] == docs[kv_idx]) & (q_idx - kv_idx < attn_blocksize)
        
        S = len(idx)
        block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

        x = self.transformer.wte(idx[None])
        smear_gate_out = self.smear_lambda * torch.sigmoid(self.smear_gate(x[:, 1:, :self.smear_gate.weight.size(-1)]))
        x = torch.cat([x[:, :1], x[:, 1:] + smear_gate_out * x[:, :-1]], dim=1)
        x = norm(x); x0 = x; v1 = None

        ve = [value_embed(idx) for value_embed in self.value_embeds]
        ve_list = [None, ve[1], ve[2]] + [None] * (len(self.transformer.h) - 6) + [ve[0], ve[1], ve[2]]
        
        skip_connections = []
        x_backout = None
        backout_layer = 8

        # Encoder pass
        for i in range(self.num_encoder_layers):
            if ve_list[i] is not None and v1 is None: v1 = ve_list[i][None] 
            x, v1 = self.transformer.h[i](x, v1, x0, block_mask, 0.1)
            skip_connections.append(x)
            if i == backout_layer: x_backout = x

        # Decoder pass (with U-Net skips)
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i
            x = x + self.skip_weights[i] * skip_connections.pop()
            if ve_list[layer_idx] is not None and v1 is None: v1 = ve_list[layer_idx][None]
            x, v1 = self.transformer.h[layer_idx](x, v1, x0, block_mask, 0.1)
            if layer_idx == backout_layer: x_backout = x

        if x_backout is not None: 
            x = x - self.backout_lambda * x_backout
        
        # Scaled tanh logits optimization (prevents instabilities)
        logits = 30 * torch.tanh(self.lm_head(norm(x)) / 30)
        return F.cross_entropy(logits.view(-1, logits.size(-1)).bfloat16(), target.view(-1))

# -----------------------------------------------------------------------------
# 4. Data Loading (Async)
# -----------------------------------------------------------------------------
@dataclass
class Hyperparameters:
    input_bin: str = 'data/finewebEDU/fineweb_edu_train_*.bin'
    input_val_bin: str = 'data/finewebEDU/fineweb_edu_val_*.bin'
    batch_size: int = 16
    sequence_length: int = 32 * 1024
    num_iterations: int = 1750
    warmup_iters: int = 0
    cooldown_iters: int = 640

def _load_data_shard(filename):
    with open(filename, "rb") as f: 
        # Skip header, read data
        return torch.frombuffer(f.read(), dtype=torch.uint16)[256*2:]

class DistributedDataLoader:
    def __init__(self, filename_pattern, T):
        self.T = T
        self.files = sorted(glob.glob(filename_pattern))
        self.reset()
        
    def reset(self): 
        self.current_shard = -1
        self.advance()
        
    def advance(self):
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        
    def next_batch(self):
        if self.current_position + self.T + 1 >= len(self.tokens): 
            self.advance()
        
        # Get slice
        buf = self.tokens[self.current_position:self.current_position+self.T+1]
        self.current_position += self.T
        
        # Convert to int32 (long) on CPU, pinned memory for fast transfer
        # Note: Pinning memory is crucial for Async transfer
        data = torch.tensor(buf.numpy().astype('int32'), dtype=torch.long)
        data = data.pin_memory() 
        return data[:-1], data[1:]

class AsyncDataLoader:
    """
    Prefetches batches to GPU asynchronously.
    """
    def __init__(self, loader, device='cuda'):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream()
        self.next_input = None
        self.next_target = None
        self.preload()

    def preload(self):
        try:
            input_cpu, target_cpu = self.loader.next_batch()
        except Exception as e:
            # Handle end of epoch or errors if necessary
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = input_cpu.to(self.device, non_blocking=True)
            self.next_target = target_cpu.to(self.device, non_blocking=True)

    def next_batch(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        
        if self.next_input is None:
            # If loader exhausted, reset (or handle as needed)
            self.loader.reset()
            self.preload()
            torch.cuda.current_stream().wait_stream(self.stream)
            
        x, y = self.next_input, self.next_target
        self.preload() # Start fetching next immediately
        return x, y

# -----------------------------------------------------------------------------
# 5. Execution Setup
# -----------------------------------------------------------------------------

args = Hyperparameters()
config = GPTConfig(flex_kernel_options={
    "BLOCK_M": 64, "BLOCK_N": 64, 
    "BLOCK_M1": 32, "BLOCK_N1": 64, 
    "BLOCK_M2": 64, "BLOCK_N2": 32
})
torch.cuda.set_device('cuda:0')

# Initialize Async Loader
raw_loader = DistributedDataLoader(args.input_bin, args.sequence_length)
train_loader = AsyncDataLoader(raw_loader)

model = GPT(config).bfloat16().cuda()

print("Converting Linear layers to Float8...")
# Only convert divisible dimensions (standard FP8 requirement)
convert_to_float8_training(model.transformer.h, 
                           module_filter_fn=lambda m, n: isinstance(m, torch.nn.Linear) 
                           and m.in_features % 16 == 0 and m.out_features % 16 == 0)

model = torch.compile(model)

# Parameter grouping
param_dict = {pn: p for pn, p in model.named_parameters()}
matrix_params = [p for p in param_dict.values() if p.ndim == 2]
scalar_params = [p for p in param_dict.values() if p.ndim < 2]

# Optimizers
opt_matrix = Muon(matrix_params, lr=0.05, momentum=0.95)
opt_scalar = torch.optim.Adam(scalar_params, lr=0.04, betas=(0.8, 0.95), fused=True)
opt_wte = torch.optim.Adam([model.transformer.wte.weight] + [v.weight for v in model.value_embeds], 
                           lr=0.6, betas=(0.8, 0.95), fused=True)
opt_head = torch.optim.Adam([model.lm_head.weight], lr=0.008, betas=(0.8, 0.95), fused=True)
optimizers = [opt_wte, opt_head, opt_matrix, opt_scalar]

def get_lr(it):
    if it < args.num_iterations - args.cooldown_iters: return 1.0 
    return (args.num_iterations - it) / args.cooldown_iters

schedulers = [torch.optim.lr_scheduler.LambdaLR(opt, get_lr) for opt in optimizers]

# -----------------------------------------------------------------------------
# 6. Training Loop
# -----------------------------------------------------------------------------

# Prefetch first batch
x, y = train_loader.next_batch()

t0 = time.time()
run_id = str(uuid.uuid4())
print(f"Run ID: {run_id}")
print("Step\tLoss\tTime(ms)\tkt/s\tWindow\tETA(m)")

for step in range(args.num_iterations + 1):
    # Dynamic Window Schedule
    window = 256 * ((step / args.num_iterations * (1792 - 256) + 256) // 256)
    attn_blocksize = torch.tensor(window, dtype=torch.int, device='cuda')

    model.train()
    
    # Gradient Accumulation Loop
    for i in range(args.batch_size):
        loss = model(x, y, attn_blocksize)
        
        # Async fetch next batch while backward pass runs
        x, y = train_loader.next_batch() 
        
        loss.backward()
        if i == args.batch_size - 1:
            train_loss = loss.detach()

    # Gradient Scaling & stepping
    for p in model.parameters():
        if p.grad is not None: 
            p.grad /= args.batch_size

    # Muon momentum schedule
    frac = min(step / 300, 1)
    opt_matrix.param_groups[0]['momentum'] = (1 - frac) * 0.85 + frac * 0.95

    for opt, sched in zip(optimizers, schedulers):
        opt.step()
        sched.step()
        
    model.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    dt = 1000 * (time.time() - t0)
    kt_s = (args.batch_size * args.sequence_length) / dt
    eta = ((args.num_iterations - step) * dt) / 1000 / 60
    
    print(f"{step+1}\t{train_loss.item():.3f}\t{dt:.0f}\t{kt_s:.1f}\t{int(window)}\t{eta:.1f}")
    t0 = time.time()
    
    if step == args.num_iterations: 
        break