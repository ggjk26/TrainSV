from __future__ import annotations

import struct

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .dnni import MATRIX_TYPES, Node, read_matrix, read_vector, sparsity_mask, write_matrix, write_vector


class MatrixParameter(nn.Module):

    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        self.weight = nn.Parameter(torch.from_numpy(read_matrix(node)))
        mask = sparsity_mask(node)
        self.register_buffer("mask", None if mask is None else torch.from_numpy(mask.astype(np.float32)))

    def forward(self) -> torch.Tensor:
        return self.weight if self.mask is None else self.weight * self.mask

    def export(self) -> None:
        write_matrix(self.node, self().detach().cpu().double().numpy())


class VectorParameter(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        self.node = node
        self.value = nn.Parameter(torch.from_numpy(read_vector(node)))

    def forward(self) -> torch.Tensor:
        return self.value

    def export(self) -> None:
        write_vector(self.node, self.value.detach().cpu().numpy())


def _ints(payload: bytes, count: int) -> tuple[int, ...]:
    if len(payload) != 4 * count:
        raise ValueError(f"expected {count} int32 values in payload, found {len(payload)} bytes")
    return struct.unpack(f"<{count}i", payload)


class Dense(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        if node.payload or not 1 <= len(node.children) <= 2:
            raise ValueError("modl0 requires an empty payload, a matrix and an optional bias")
        self.matrix = MatrixParameter(node.children[0])
        self.bias = VectorParameter(node.children[1]) if len(node.children) == 2 else None

    def forward(self, x, condition=None):
        return F.linear(x, self.matrix(), None if self.bias is None else self.bias())

    @property
    def shape(self):
        return tuple(self.matrix.weight.shape)


class Conv(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        kernel, self.stride, self.padding, self.dilation, groups = _ints(node.payload, 5)
        if groups != 1 or kernel <= 0:
            raise ValueError("modl1 supports groups=1 only")
        taps = [child for child in node.children if child.type in MATRIX_TYPES]
        biases = [child for child in node.children if child.type == "prim1"]
        if len(taps) != kernel or len(biases) > 1:
            raise ValueError("modl1 child count does not match its kernel")
        self.taps = nn.ModuleList(MatrixParameter(child) for child in taps)
        self.bias = VectorParameter(biases[0]) if biases else None

    @property
    def shape(self):
        return tuple(self.taps[0].weight.shape)

    @property
    def kernel(self):
        return len(self.taps)

    def forward(self, x, condition=None):
        weight = torch.stack([tap() for tap in self.taps], dim=2)
        y = F.conv1d(x.t().unsqueeze(0), weight, None if self.bias is None else self.bias(), self.stride, self.padding, self.dilation)
        return y.squeeze(0).t()


class Gru(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        if node.payload or len(node.children) != 12:
            raise ValueError("modl3 requires six matrix/bias pairs")
        self.matrices = nn.ModuleList(MatrixParameter(node.children[2 * index]) for index in range(6))
        self.biases = nn.ModuleList(VectorParameter(node.children[2 * index + 1]) for index in range(6))
        hidden, inputs = self.matrices[0].weight.shape
        object.__setattr__(self, "_kernel", nn.GRU(inputs, hidden))

    def forward(self, x, condition=None, reverse=False):
        parameters = {
            "weight_ih_l0": torch.cat([self.matrices[index]() for index in range(3)]),
            "weight_hh_l0": torch.cat([self.matrices[index]() for index in range(3, 6)]),
            "bias_ih_l0": torch.cat([self.biases[index]() for index in range(3)]),
            "bias_hh_l0": torch.cat([self.biases[index]() for index in range(3, 6)]),
        }
        if reverse:
            x = x.flip(0)
        output, _ = torch.func.functional_call(self._kernel, parameters, (x,))
        return output.flip(0) if reverse else output


class BiGru(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        if node.payload or len(node.children) != 2:
            raise ValueError("modl6 requires two GRU children")
        self.forward_gru = Gru(node.children[0])
        self.backward_gru = Gru(node.children[1])

    def forward(self, x, condition=None):
        return torch.cat([self.forward_gru(x), self.backward_gru(x, reverse=True)], dim=1)


class Activation(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        if node.children:
            raise ValueError("activation cannot have children")
        self.kind = node.type
        self.alpha = 0.0
        if self.kind in ("moda3", "moda4"):
            (self.alpha,) = struct.unpack("<f", node.payload)
        elif node.payload:
            raise ValueError("activation payload must be empty")

    def forward(self, x, condition=None):
        if self.kind == "moda0":
            return F.relu(x)
        if self.kind == "moda1":
            return torch.tanh(x)
        if self.kind == "moda2":
            return torch.sigmoid(x)
        if self.kind == "moda3":
            return torch.where(x > 0, x, x * self.alpha)
        if self.kind == "moda4":
            return F.relu(x) + torch.clamp(self.alpha * torch.expm1(x), max=0.0)
        if self.kind == "moda5":
            return x
        if self.kind == "moda7":
            return F.silu(x)
        raise ValueError(self.kind)


class GatedConv(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        self.input_channels, self.gate_channels, self.condition_channels = _ints(node.payload, 3)
        expected = 2 if self.condition_channels > 0 else 1
        if len(node.children) != expected or any(child.type != "modl1" for child in node.children):
            raise ValueError("_gnc1v0 requires one or two modl1 children")
        self.input_conv = Conv(node.children[0])
        self.condition_conv = Conv(node.children[1]) if expected == 2 else None

    def forward(self, x, condition=None):
        a = self.input_conv(x)
        if self.condition_conv is not None:
            a = a + self.condition_conv(condition)
        return torch.tanh(a[:, :self.gate_channels]) * torch.sigmoid(a[:, self.gate_channels:])


class ResidualStack(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        self.input_channels, self.hidden_channels, self.condition_channels = _ints(node.payload, 3)
        conditioned = self.condition_channels > 0
        if len(node.children) != (4 if conditioned else 3):
            raise ValueError("_ncwnv0 requires three groups and an optional condition projection")
        gates, skips, residuals = node.children[:3]
        if not len(gates.children) == len(skips.children) == len(residuals.children) > 0:
            raise ValueError("_ncwnv0 groups must have equal non-zero lengths")
        self.gates = nn.ModuleList(GatedConv(child) for child in gates.children)
        self.skips = nn.ModuleList(Conv(child) for child in skips.children)
        self.residuals = nn.ModuleList(Conv(child) for child in residuals.children)
        self.condition_projection = Conv(node.children[3]) if conditioned else None

    def forward(self, x, condition=None):
        if self.condition_projection is not None:
            output = self.condition_projection(condition)
        else:
            output = x.new_zeros(x.shape[0], self.hidden_channels)
        for gate, skip, residual in zip(self.gates, self.skips, self.residuals):
            r = residual(x)
            g = gate(x, condition)
            output = output + skip(g)
            x = g + r
        return output


class Sequence(nn.Module):
    def __init__(self, node: Node):
        super().__init__()
        if node.payload:
            raise ValueError("modm0 payload must be empty")
        layers = []
        for child in node.children:
            layer = build_network(child)
            layers.extend(layer.layers if isinstance(layer, Sequence) else [layer])
        self.layers = nn.ModuleList(layers)

    def forward(self, x, condition=None):
        for layer in self.layers:
            x = layer(x, condition)
        return x


_OPERATORS = {
    "modm0": Sequence, "modl0": Dense, "modl1": Conv, "modl3": Gru, "modl6": BiGru,
    "_gnc1v0": GatedConv, "_ncwnv0": ResidualStack,
    **{name: Activation for name in ("moda0", "moda1", "moda2", "moda3", "moda4", "moda5", "moda7")},
}


def build_network(node: Node) -> nn.Module:
    try:
        operator = _OPERATORS[node.type]
    except KeyError:
        raise ValueError(f"unsupported DNNI operator {node.type}") from None
    return operator(node)


def export_parameters(module: nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, (MatrixParameter, VectorParameter)):
            child.export()
