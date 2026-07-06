"""FloodGNN — edge-conditioned message passing in plain PyTorch.

No torch-geometric: the two graph ops a GNN actually needs are gather (`h[src]`) and
scatter (`index_add_` / `scatter_reduce`), both core torch. That keeps the dependency
surface identical to the rest of Varuna (runs on the HF Space CPU image unchanged) and
every line of the message passing inspectable:

    message  m_uv = MLP([h_u, h_v, e_uv])          for every directed street edge u->v
    aggregate a_v = [mean_u m_uv, max_u m_uv]      over v's incoming edges
    update   h_v <- LayerNorm(h_v + MLP([h_v, a_v]))

Rainfall enters through FiLM (feature-wise linear modulation): a small MLP maps the storm
size to a per-layer (gamma, beta) that scales/shifts every node state. One network therefore
answers the *continuous* family of questions "where does an r-mm storm put water on the
streets?" instead of one model per storm.

Heads:
  node_depth [m]  — flood depth at the node's cell (evacuation risk at street corners)
  edge_depth [m]  — max depth along the street segment (what routing actually costs on)
  node_flow       — log1p(m³) drainage volume the street drain planner would send through
                    this node (where water *wants* to go -> drain corridor scoring)

`layers=0` degrades to a per-node MLP (still FiLM-conditioned) — the no-graph ablation that
proves message passing earns its keep.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .graph import EDGE_FEATURES, NODE_FEATURES


def mlp(sizes, act=nn.SiLU, dropout=0.0):
    """Linear stack with `act` between layers; the last layer stays linear."""
    layers = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(act())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def _aggregate(messages, dst, n_nodes):
    """Per-node [mean, max] of incoming messages; isolated nodes get zeros."""
    H = messages.shape[1]
    total = torch.zeros(n_nodes, H, device=messages.device, dtype=messages.dtype)
    total.index_add_(0, dst, messages)
    deg = torch.zeros(n_nodes, device=messages.device, dtype=messages.dtype)
    deg.index_add_(0, dst, torch.ones_like(dst, dtype=messages.dtype))
    mean = total / deg.clamp(min=1.0).unsqueeze(1)
    mx = torch.full((n_nodes, H), float("-inf"), device=messages.device, dtype=messages.dtype)
    mx.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, H), messages, reduce="amax",
                       include_self=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=1)


class MPLayer(nn.Module):
    def __init__(self, hidden, dropout=0.0):
        super().__init__()
        self.msg = mlp([3 * hidden, hidden, hidden], dropout=dropout)
        self.upd = mlp([3 * hidden, hidden, hidden], dropout=dropout)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, e, src, dst):
        m = self.msg(torch.cat([h[src], h[dst], e], dim=1))
        a = _aggregate(m, dst, h.shape[0])
        return self.norm(h + self.upd(torch.cat([h, a], dim=1)))


class FloodGNN(nn.Module):
    def __init__(self, node_dim=len(NODE_FEATURES), edge_dim=len(EDGE_FEATURES),
                 hidden=96, layers=4, dropout=0.0):
        super().__init__()
        self.cfg = dict(node_dim=node_dim, edge_dim=edge_dim, hidden=hidden,
                        layers=layers, dropout=dropout)
        self.enc_node = mlp([node_dim, hidden, hidden])
        self.enc_edge = mlp([edge_dim, hidden, hidden])
        self.mp = nn.ModuleList(MPLayer(hidden, dropout) for _ in range(layers))
        n_film = max(layers, 1)                                # layers=0 still FiLMs the encoding
        self.film = mlp([1, hidden, 2 * hidden * n_film])
        self.head_node = mlp([hidden, hidden, 1])
        self.head_flow = mlp([hidden, hidden, 1])
        self.head_edge = mlp([3 * hidden, hidden, 1])
        for head in (self.head_node, self.head_flow, self.head_edge):
            nn.init.constant_(head[-1].bias, -3.0)             # start ~dry: floods are sparse

    def forward(self, x, edge_index, edge_attr, rain_mm):
        """x [N,F], edge_index [2,E], edge_attr [E,3], rain_mm scalar (mm) -> dict of heads."""
        src, dst = edge_index[0], edge_index[1]
        r = torch.as_tensor([[float(rain_mm) / 100.0]], device=x.device, dtype=x.dtype)
        H = self.cfg["hidden"]
        film = self.film(r).view(-1, 2, H)                     # [n_film, (gamma|beta), H]

        h = self.enc_node(x)
        e = self.enc_edge(edge_attr)
        if not self.mp:                                        # ablation: no message passing
            h = (1.0 + film[0, 0]) * h + film[0, 1]
        for k, layer in enumerate(self.mp):
            h = (1.0 + film[k, 0]) * h + film[k, 1]
            h = layer(h, e, src, dst)

        sp = nn.functional.softplus
        return {
            "node_depth": sp(self.head_node(h)).squeeze(-1),
            "node_flow": sp(self.head_flow(h)).squeeze(-1),
            "edge_depth": sp(self.head_edge(torch.cat([h[src], h[dst], e], dim=1))).squeeze(-1),
        }


def save_checkpoint(model, path, extra=None):
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": model.cfg,
                "node_features": list(NODE_FEATURES), "edge_features": list(EDGE_FEATURES),
                **(extra or {})}, path)
    return path


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FloodGNN(**ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
