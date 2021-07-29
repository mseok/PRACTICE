import torch
import torch.nn as nn


class EGCL(nn.Module):
    r"""\
    Equivariant Graph Convolutional Layer
    \bold{m}_{ij}=\phi_e(\bold{h}^l_i,\bold{h}^l_j,
    ||\bold{x}^l_i-\bold{x}^l_j||^2, a_{ij})
    \bold{x}^{l+1}_i=\bold x^l_i +
    C\sum_{j\ne i}(\bold{x}^l_i-\bold{x}^l_j)\phi_x(\bold{m}_{ij})
    \bold{m}_i=\sum_{j\in N(i)}\bold{m}_{ij}
    \bold{h}^{l+1}_i=\phi_h(\bold{h}^l_i, \bold{m}_i)

    Input:
        node_feat:    Atom feature             [B,N,NF]
        pos:          Atom position            [B,N,3]
        adj:          Adjacency matrix         [B,N,N]
        valid:        Valid indices            [B,]

    Output:
        pos:        New atom position        [B,N,3]
        node_feat:  New atom feature         [B,N,3]
    """

    def __init__(self, n_node_feat, hidden_dim, infer=False, update_pos=False):
        super().__init__()
        self.edge_linear = nn.Sequential(
            nn.Linear(2 * n_node_feat + 1 + 1, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.coordinate_linear = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.node_linear = nn.Sequential(
            nn.Linear(n_node_feat + hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_node_feat, bias=False),
        )
        self.infer = infer
        if self.infer:
            self.inference_linear = nn.Sequential(
                nn.Linear(hidden_dim, 1, bias=False),
                nn.Sigmoid(),
            )
        self.update_pos = update_pos

    def forward(self, node_feat, pos, valid, adj):
        n_atoms = node_feat.shape[1]
        hi = node_feat.unsqueeze(2).repeat_interleave(n_atoms, 2)
        hj = node_feat.unsqueeze(1).repeat_interleave(n_atoms, 1)
        vec = self._compute_vec_from_pos(pos)
        _valid = valid.unsqueeze(-1).unsqueeze(-1).repeat_interleave(n_atoms, 2)
        dist = torch.pow(vec, 2).sum(dim=-1, keepdim=True)
        if len(adj.shape) == 3:
            adj = adj.unsqueeze(-1)
        edge_input = torch.cat([hi, hj, dist, adj], dim=-1)

        m_ij = self.edge_linear(edge_input)
        vec = self._normalize(vec)
        if self.update_pos:
            pos += (vec * self.coordinate_linear(m_ij) * _valid).mean(2)
        if self.infer:
            e_ij = self.inference_linear(m_ij)
            m_ij = (m_ij * e_ij)
        message = (m_ij * _valid).mean(2)
        node_input = torch.cat([node_feat, message], dim=-1)
        node_feat += self.node_linear(node_input)
        return pos, node_feat

    def _compute_vec_from_pos(self, pos):
        return pos.unsqueeze(2) - pos.unsqueeze(1)

    def _normalize(self, vec):
        norm = vec.norm(dim=-1, keepdim=True)
        normalized = vec / norm.clamp(min=1e-10)
        return normalized
