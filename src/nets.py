"""Neural architectures: deep baselines + proposed candidates.
All models:  forward(x: (B,W,m) whitened residual window, ctx: dict) -> (class_logits (B,4), meas_logits (B,m))."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NC = 4


def prep(x):
    """Common input compression for every learned model: asinh keeps small residuals linear and tames outliers."""
    return torch.asinh(x)


def build_adj(kind, loc, f, t, self_loop=True):
    """Measurement graph: two RTU measurements are adjacent if they touch a common bus."""
    m = len(kind)
    buses = []
    for k, l in zip(kind, loc):
        buses.append({int(l)} if k <= 2 else {int(f[l]), int(t[l])})
    A = np.zeros((m, m), np.float32)
    by_bus = {}
    for i, bs in enumerate(buses):
        for b in bs:
            by_bus.setdefault(b, []).append(i)
    for lst in by_bus.values():
        for i in lst:
            A[i, lst] = 1
    if not self_loop:
        np.fill_diagonal(A, 0)
    return A


# ===================================================================================== deep baselines
class MLP(nn.Module):
    def __init__(self, m, W, h=512, cin=None):
        super().__init__()
        self.body = nn.Sequential(nn.Linear((cin or m) * W, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, h // 2), nn.GELU())
        self.hc, self.hm = nn.Linear(h // 2, NC), nn.Linear(h // 2, m)

    def forward(self, x, ctx=None):
        z = self.body(prep(x).flatten(1))
        return self.hc(z), self.hm(z)


class TCN(nn.Module):
    """Channels = measurements, causal dilated convolutions over the window."""
    def __init__(self, m, W, h=128, cin=None):
        super().__init__()
        self.inp = nn.Conv1d(cin or m, h, 1)
        self.blocks = nn.ModuleList([nn.Conv1d(h, h, 3, padding=0, dilation=d) for d in (1, 2, 4)])
        self.norm = nn.ModuleList([nn.GroupNorm(1, h) for _ in range(3)])
        self.hc, self.hm = nn.Linear(2 * h, NC), nn.Linear(2 * h, m)

    def forward(self, x, ctx=None):
        z = self.inp(prep(x).transpose(1, 2))
        for conv, nm, d in zip(self.blocks, self.norm, (1, 2, 4)):
            zz = F.pad(z, (2 * d, 0), mode="replicate")
            z = z + F.gelu(nm(conv(zz)))
        g = torch.cat([z[..., -1], z.amax(-1)], 1)
        return self.hc(g), self.hm(g)


class GRUNet(nn.Module):
    def __init__(self, m, W, h=128, cin=None):
        super().__init__()
        self.rnn = nn.GRU(cin or m, h, 2, batch_first=True, dropout=0.1)
        self.hc, self.hm = nn.Linear(h, NC), nn.Linear(h, m)

    def forward(self, x, ctx=None):
        o, _ = self.rnn(prep(x))
        g = o[:, -1]
        return self.hc(g), self.hm(g)


class TemporalTransformer(nn.Module):
    """Tokens = time steps (each a m-dim residual vector)."""
    def __init__(self, m, W, d=128, cin=None):
        super().__init__()
        self.inp = nn.Linear(cin or m, d); self.pos = nn.Parameter(torch.zeros(1, W, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2)
        self.hc, self.hm = nn.Linear(d, NC), nn.Linear(d, m)

    def forward(self, x, ctx=None):
        z = self.enc(self.inp(prep(x)) + self.pos)
        g = z[:, -1]
        return self.hc(g), self.hm(g)


class FTTransformer(nn.Module):
    """FT-Transformer style: one token per measurement (its W-step window embedded), attention across measurements."""
    def __init__(self, m, W, d=48, kinds=5, win=None):
        super().__init__()
        self.emb = nn.Linear(win or W, d); self.idx = nn.Parameter(torch.randn(1, m, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, 4, 2 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, 2)
        self.hc, self.hm = nn.Linear(d, NC), nn.Linear(d, 1)

    def forward(self, x, ctx=None):
        tok = self.emb(prep(x).transpose(1, 2)) + self.idx
        z = self.enc(torch.cat([self.cls.expand(len(tok), -1, -1), tok], 1))
        return self.hc(z[:, 0]), self.hm(z[:, 1:]).squeeze(-1)


class GraphBaseline(nn.Module):
    """GCN / GAT over the measurement graph; node features = window of that measurement."""
    def __init__(self, m, W, A, kind="gcn", d=64, win=None):
        super().__init__()
        self.kind_ = kind
        An = A / A.sum(1, keepdims=True)
        self.register_buffer("An", torch.tensor(An))
        self.inp = nn.Linear(win or W, d)
        if kind == "gcn":
            self.l1, self.l2 = nn.Linear(d, d), nn.Linear(d, d)
        else:
            from torch_geometric.nn import GATConv
            self.g1, self.g2 = GATConv(d, d // 4, heads=4), GATConv(d, d // 4, heads=4)
            ei = torch.tensor(np.array(np.nonzero(A)), dtype=torch.long)
            self.register_buffer("ei", ei)
        self.hn = nn.Linear(d, 1); self.hc = nn.Linear(2 * d, NC)

    def forward(self, x, ctx=None):
        B, W, m = x.shape
        h = F.gelu(self.inp(prep(x).transpose(1, 2)))                  # (B,m,d)
        if self.kind_ == "gcn":
            h = h + F.gelu(self.l1(self.An @ h))
            h = h + F.gelu(self.l2(self.An @ h))
        else:
            d = h.shape[-1]
            hf = h.reshape(B * m, d)
            off = (torch.arange(B, device=x.device) * m)[:, None, None]
            ei = (self.ei[None] + off).permute(1, 0, 2).reshape(2, -1)
            hf = hf + F.gelu(self.g1(hf, ei)); hf = hf + F.gelu(self.g2(hf, ei))
            h = hf.reshape(B, m, d)
        return self.hc(torch.cat([h.mean(1), h.amax(1)], 1)), self.hn(h).squeeze(-1)


class MambaLite(nn.Module):
    """Selective state-space (Mamba-style) baseline re-implemented in PyTorch; scan over the W-step window.
    Channels = measurements projected to d_model."""
    def __init__(self, m, W, d=96, n=16, layers=2, cin=None):
        super().__init__()
        self.inp = nn.Linear(cin or m, d)
        self.layers = nn.ModuleList([_MambaBlock(d, n) for _ in range(layers)])
        self.hc, self.hm = nn.Linear(d, NC), nn.Linear(d, m)

    def forward(self, x, ctx=None):
        z = self.inp(prep(x))
        for l in self.layers:
            z = l(z)
        g = z[:, -1]
        return self.hc(g), self.hm(g)


class _MambaBlock(nn.Module):
    def __init__(self, d, n, expand=2):
        super().__init__()
        di = d * expand
        self.norm = nn.LayerNorm(d)
        self.inp = nn.Linear(d, 2 * di)
        self.conv = nn.Conv1d(di, di, 3, padding=2, groups=di)
        self.xproj = nn.Linear(di, 1 + 2 * n)
        self.dt = nn.Linear(1, di)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, n + 1).float()).repeat(di, 1))
        self.D = nn.Parameter(torch.ones(di))
        self.out = nn.Linear(di, d)
        self.n = n

    def forward(self, x):
        B, L, _ = x.shape
        u, gate = self.inp(self.norm(x)).chunk(2, -1)
        u = F.silu(self.conv(u.transpose(1, 2))[..., :L].transpose(1, 2))
        dt_b_c = self.xproj(u)
        dt, Bm, Cm = dt_b_c[..., :1], dt_b_c[..., 1:1 + self.n], dt_b_c[..., 1 + self.n:]
        dt = F.softplus(self.dt(dt))                                   # (B,L,di)
        A = -torch.exp(self.A_log)                                     # (di,n)
        h = torch.zeros(B, u.shape[-1], self.n, device=x.device)
        ys = []
        for t in range(L):
            dA = torch.exp(dt[:, t, :, None] * A[None])
            dB = dt[:, t, :, None] * Bm[:, t, None, :]
            h = dA * h + dB * u[:, t, :, None]
            ys.append((h * Cm[:, t, None, :]).sum(-1) + self.D * u[:, t])
        y = torch.stack(ys, 1) * F.silu(gate)
        return x + self.out(y)


# ===================================================================================== proposed building blocks
class S4DConv(nn.Module):
    """Diagonal state-space convolution (S4D-Lin / real-imag init) with a W-tap kernel computed in closed form."""
    def __init__(self, d, n=8, W=12):
        super().__init__()
        self.W = W
        self.logdt = nn.Parameter(torch.rand(d) * (math.log(0.2) - math.log(0.01)) + math.log(0.01))
        self.logA_re = nn.Parameter(torch.log(0.5 * torch.ones(d, n)))
        self.A_im = nn.Parameter(math.pi * torch.arange(n).float().repeat(d, 1))
        self.C = nn.Parameter(torch.randn(d, n, 2) * 0.5)
        self.D = nn.Parameter(torch.randn(d))

    def kernel(self):
        dt = torch.exp(self.logdt)[:, None]
        A = -torch.exp(self.logA_re) + 1j * self.A_im                  # (d,n)
        C = torch.view_as_complex(self.C)
        dtA = A * dt
        Bbar = (torch.exp(dtA) - 1) / A
        l = torch.arange(self.W, device=A.device)
        K = 2 * ((C * Bbar)[..., None] * torch.exp(dtA[..., None] * l)).sum(1).real   # (d,W)
        return K

    def forward(self, u):
        """u: (B,d,W) -> causal conv with kernel."""
        K = self.kernel()
        W = u.shape[-1]
        idx = torch.arange(W, device=u.device)
        lag = idx[:, None] - idx[None, :]                              # t - s
        Tm = torch.where((lag >= 0)[None], K[:, lag.clamp(min=0)], torch.zeros(1, device=u.device))   # (d,W,W)
        return torch.einsum("dts,bds->bdt", Tm, u) + self.D[None, :, None] * u


class PerMeasEncoder(nn.Module):
    """Parameter-shared temporal encoder applied to every measurement; conditioned on measurement kind and on the
    robust noise scale (FiLM)."""
    def __init__(self, d=32, W=12, layers=2, kind="ssm", film=True, n_kinds=10):
        super().__init__()
        self.kind_ = kind
        self.inp = nn.Linear(4, d)
        self.kemb = nn.Embedding(n_kinds, d)
        self.film = nn.Linear(1 + d, 2 * d) if film else None
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.mix = nn.ModuleList([nn.Linear(d, 2 * d) for _ in range(layers)])
        if kind == "ssm":
            self.cores = nn.ModuleList([S4DConv(d, 8, W) for _ in range(layers)])
        else:
            self.cores = nn.ModuleList([nn.Conv1d(d, d, 3, padding=0, dilation=2 ** i, groups=d) for i in range(layers)])
        self.d = d
        self.W = W

    def forward(self, r, lag, tf, kinds, logs):
        """r: (B,m,W) raw whitened residual; lag: (B,m,W) clone-fingerprint feature; kinds: (m,) long; logs: (B,) log robust scale."""
        B, m, W = r.shape
        s = torch.exp(logs)[:, None, None].clamp(min=0.2)
        feats = torch.stack([torch.asinh(r), torch.asinh(r / s), torch.asinh(lag), tf], -1)       # (B,m,W,4)
        ke = self.kemb(kinds)
        if ke.dim() == 2:
            ke = ke[None]
        ke = ke.expand(B, m, -1)
        h = self.inp(feats) + ke[:, :, None, :]
        if self.film is not None:
            gb = self.film(torch.cat([logs[:, None, None].expand(B, m, 1), ke], -1))
            gamma, beta = gb.chunk(2, -1)
        for i in range(len(self.cores)):
            y = self.norms[i](h)
            if self.film is not None and i == 0:
                y = y * (1 + gamma[:, :, None, :]) + beta[:, :, None, :]
            y = y.reshape(B * m, W, self.d).transpose(1, 2)
            if self.kind_ == "ssm":
                y = self.cores[i](y)
            else:
                d = 2 ** i
                y = self.cores[i](F.pad(y, (2 * d, 0), mode="replicate"))
            y = F.gelu(y).transpose(1, 2).reshape(B, m, W, self.d)
            a, g = self.mix[i](y).chunk(2, -1)
            h = h + a * torch.sigmoid(g)
        return torch.cat([h[:, :, -1], h.amax(2)], -1)                     # (B,m,2d)


class WRNet(nn.Module):
    """Whitened-Residual temporal(-graph) network.  Variants (candidates / ablations):
       temporal in {'ssm','tcn','none'}, graph in {True,False}, film in {True,False}."""
    def __init__(self, m, W, A, kinds, vbit=None, d=32, temporal="ssm", graph=True, film=True, layers=2, glayers=2, attn=False, heads=4, lag=True, attn_bias=True,
                 prior=True, prior_drop=0.0):
        super().__init__()
        self.temporal, self.use_graph, self.use_attn, self.use_lag = temporal, graph, attn, lag
        self.register_buffer("kinds", torch.tensor(kinds, dtype=torch.long))
        self.register_buffer("vbit", torch.tensor(vbit if vbit is not None else np.zeros(len(kinds), int), dtype=torch.long))
        self.prior, self.prior_drop = prior, prior_drop
        if temporal == "none":
            self.enc0 = nn.Sequential(nn.Linear(W, d), nn.GELU(), nn.Linear(d, 2 * d))
            self.kemb = nn.Embedding(10, 2 * d)
        else:
            self.enc = PerMeasEncoder(d, W, layers, temporal, film)
        D2 = 2 * d
        An = A / A.sum(1, keepdims=True)
        self.register_buffer("An", torch.tensor(An))
        self.gl = nn.ModuleList([nn.Linear(D2, D2) for _ in range(glayers)]) if graph else None
        self.gn = nn.ModuleList([nn.LayerNorm(D2) for _ in range(glayers)]) if graph else None
        if attn:
            self.register_buffer("Abin", torch.tensor(A))
            self.beta = nn.Parameter(torch.tensor(2.0), requires_grad=attn_bias)
            if not attn_bias:
                self.beta.data.zero_()
            self.att = nn.ModuleList([nn.MultiheadAttention(D2, heads, batch_first=True) for _ in range(glayers)])
            self.an = nn.ModuleList([nn.LayerNorm(D2) for _ in range(glayers)])
            self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(D2), nn.Linear(D2, 2 * D2), nn.GELU(), nn.Linear(2 * D2, D2)) for _ in range(glayers)])
        self.ctx = nn.Linear(2 * D2 + 3, D2)
        self.node = nn.Sequential(nn.Linear(2 * D2, D2), nn.GELU(), nn.Linear(D2, 1))
        self.cls = nn.Sequential(nn.Linear(2 * D2 + 3, D2), nn.GELU(), nn.Linear(D2, NC))

    def set_system(self, A, kinds, vbit=None):
        """Swap the measurement graph / kinds (parameters are independent of the system size)."""
        dev = self.An.device
        An = A / A.sum(1, keepdims=True)
        self.An = torch.tensor(An, device=dev); self.kinds = torch.tensor(kinds, dtype=torch.long, device=dev)
        self.vbit = torch.tensor(vbit if vbit is not None else np.zeros(len(kinds), int), dtype=torch.long, device=dev)
        if self.use_attn:
            self.Abin = torch.tensor(A, device=dev)

    def forward(self, x, ctx=None):
        if x.dim() == 4:
            lag = x[..., 1].transpose(1, 2)
            tf = x[..., 4].transpose(1, 2) if x.shape[-1] > 4 else torch.zeros_like(lag)
            x = x[..., 0]
        else:
            lag = torch.full_like(x, 12.0).transpose(1, 2); tf = torch.zeros_like(lag)
        B, W, m = x.shape
        if self.prior:
            keep = torch.ones(B, 1, device=x.device)
            if self.training and self.prior_drop > 0:
                keep = (torch.rand(B, 1, device=x.device) > self.prior_drop).float()
            ki = self.kinds[None] + 5 * (self.vbit[None] * keep.long())
        else:
            ki = self.kinds
        r = x.transpose(1, 2)                                             # (B,m,W)
        logs = torch.log(torch.median(r[..., -1].abs(), dim=1).values / 0.6745 + 1e-3)   # robust noise-scale
        if self.temporal == "none":
            h = self.enc0(torch.asinh(r)) + self.kemb(ki).expand(B, m, -1)
        else:
            h = self.enc(r, lag if self.use_lag else torch.full_like(lag, 12.0), tf, ki, logs)
        if self.use_graph:
            for lin, nm in zip(self.gl, self.gn):
                h = h + F.gelu(lin(nm(self.An @ h)))
        if self.use_attn:
            bias = self.beta * self.Abin                                  # physics prior: bus-sharing measurements attend more
            for att, nm, ff in zip(self.att, self.an, self.ff):
                y = nm(h)
                h = h + att(y, y, y, attn_mask=bias, need_weights=False)[0]
                h = h + ff(h)
        s = torch.asinh(r[..., -1] / torch.exp(logs)[:, None].clamp(min=0.2))
        gfeat = torch.stack([logs, (s.abs() > 3).float().mean(1), s.abs().amax(1) / 10], -1)
        pooled = torch.cat([h.mean(1), h.amax(1), gfeat], -1)
        c = F.gelu(self.ctx(pooled))
        node = self.node(torch.cat([h, c[:, None].expand(-1, m, -1)], -1)).squeeze(-1)
        return self.cls(pooled), node
