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
        node_feat:  New atom feature           [B,N,NF]
        pos:        New atom position          [B,N,3]
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
        _valid1 = valid.unsqueeze(2).repeat_interleave(n_atoms, 2)
        _valid2 = valid.unsqueeze(1).repeat_interleave(n_atoms, 1)
        _valid = (_valid1 * _valid2).unsqueeze(-1)
        dist = torch.pow(vec, 2).sum(dim=-1, keepdim=True)
        if len(adj.shape) == 3:
            adj = adj.unsqueeze(-1)
        edge_input = torch.cat([hi, hj, dist, adj], dim=-1)

        m_ij = self.edge_linear(edge_input)
        vec = self._normalize(vec)
        if self.update_pos:
            C = 1 / (valid.sum(-1) - 1)
            C = C.unsqueeze(-1).unsqueeze(-1)
            pos += C * (vec * self.coordinate_linear(m_ij) * _valid).sum(2)
        if self.infer:
            e_ij = self.inference_linear(m_ij)
            m_ij = (m_ij * e_ij)
        message = (m_ij * adj).sum(2)
        node_input = torch.cat([node_feat, message], dim=-1)
        node_feat += self.node_linear(node_input)
        return node_feat, pos

    def _compute_vec_from_pos(self, pos):
        return pos.unsqueeze(2) - pos.unsqueeze(1)

    def _normalize(self, vec):
        norm = vec.norm(dim=-1, keepdim=True)
        normalized = vec / norm.clamp(min=1e-10)
        return normalized


class InteractionLayer(nn.Module):
    r"""\
    Interaction layer from SchNet
    Contains continuous filter convolutional layer (CFConv)
    \bold x^{l+1}_i=(X^l *W^l)_i=\sum^{n_{atoms}}_{j=0}\bold x^l_j
    \circ W^l(\bold r_j - \bold r_i)

    Input:
        node_feat:    Atom feature             [B,N,NF]
        pos:          Atom position            [B,N,3]
        valid:        Valid indices            [B,]

    Output:
        node_feat:  New atom feature           [B,N,NF]
    """

    def __init__(self, in_dim, hidden_dim, out_dim, gamma, n_filters,
                 filter_spacing):
        super().__init__()
        self.filter_linear1 = nn.Linear(n_filters, hidden_dim, bias=False)
        self.filter_linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atomwise_linear1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.atomwise_linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.atomwise_linear3 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.act = SSP(a=0.5, b=0.5)
        self.gamma = gamma
        self.n_filters = n_filters
        self.filter_spacing = filter_spacing

    def forward(self, node_feat, pos, valid):
        _node_feat = self.atomwise_linear1(node_feat)
        _node_feat = self._cfconv(_node_feat, pos)
        _node_feat = self.atomwise_linear2(_node_feat)
        _node_feat = self.act(_node_feat)
        _node_feat = self.atomwise_linear3(_node_feat)
        node_feat = node_feat + _node_feat * valid.unsqueeze(-1)
        return node_feat

    def _compute_vec_from_pos(self, pos):
        return pos.unsqueeze(2) - pos.unsqueeze(1)

    def _cfconv(self, node_feat, pos):
        vec = self._compute_vec_from_pos(pos)
        dist = torch.sqrt(torch.pow(vec, 2).sum(-1))         # B,N,N
        filters = self._generate_filter_layers(dist)         # B,N,N,NF
        _node_feat = node_feat.unsqueeze(2)                  # B,N,1,NF
        _node_feat = (_node_feat * filters).sum(2)           # B,N,NF
        return _node_feat

    def _generate_filter_layers(self, dist):
        r"""\
        radial basis function
        e_k(\bold r_j - \bold r_i) = \exp(-\gamma ||d_{ij} - \mu_k||^2)
        """
        filter_centers = torch.Tensor(
            [self.filter_spacing * i for i in range(self.n_filters)]
        )
        filter_centers = filter_centers.to(dist.device)
        dist = dist.unsqueeze(-1).repeat_interleave(filter_centers.shape[0], -1)
        filter_centers = filter_centers.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        filters = torch.pow(dist - filter_centers, 2)
        filters = torch.exp(-self.gamma * filters)
        filters = self.filter_linear1(filters)               # B,N,N,NF
        filters = self.act(filters)
        filters = self.filter_linear2(filters)
        filters = self.act(filters)
        return filters


class SSP(nn.Module):
    r"""\
    Shifted SoftPlus
    ssp(x) = \ln (a*e^x + b)
    """
    def __init__(self, a=0.5, b=0.5):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, inp):
        return torch.log(self.a * torch.exp(inp) + self.b)
