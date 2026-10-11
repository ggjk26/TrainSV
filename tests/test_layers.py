import numpy as np
import torch

import parity
from opensv_train.layers import Conv, Dense, GatedConv, Gru, ResidualStack, build_network
from opensv_train.voice import ROLES, load_voice

NETWORK_TYPES = {"modm0", "_ncwnv0", "modl6", "modl3", "modl0", "modl1"}


def top_level_networks(root):
    index = 0

    def visit(node, inside):
        nonlocal index
        here = index
        index += 1
        runnable = node.type in NETWORK_TYPES
        if runnable and not inside:
            yield here, node
        for child in node.children:
            yield from visit(child, inside or runnable)

    yield from visit(root, False)


def widths(module):
    for layer in module.modules():
        if isinstance(layer, (Dense, Conv)):
            input_width = layer.shape[1]
            break
        if isinstance(layer, (GatedConv, ResidualStack)):
            input_width = layer.input_channels
            break
        if isinstance(layer, Gru):
            input_width = layer.matrices[0].weight.shape[1]
            break
    condition = next((layer.condition_channels for layer in module.modules() if isinstance(layer, (GatedConv, ResidualStack)) and layer.condition_channels), 0)
    return input_width, condition


def test_networks_match_engine(voice_path):
    voice = load_voice(voice_path)
    generator = np.random.default_rng(0)
    checked = 0
    for role in ROLES:
        for index, node in top_level_networks(voice.models[role].root):
            network = build_network(node)
            input_width, condition_width = widths(network)
            frames = 37
            x = generator.standard_normal((frames, input_width)).astype(np.float32)
            arrays = {"input": x}
            if condition_width:
                arrays["condition"] = generator.standard_normal((frames, condition_width)).astype(np.float32)
            expected = parity.run({"voice": voice_path, "command": "dnni", "model": role, "node": index}, arrays)["output"]
            with torch.no_grad():
                actual = network(torch.from_numpy(x), None if not condition_width else torch.from_numpy(arrays["condition"])).numpy()
            assert actual.shape == expected.shape, (role, index)
            scale = max(1.0, float(np.abs(expected).max()))
            np.testing.assert_allclose(actual, expected, atol=2e-4 * scale, rtol=2e-4, err_msg=f"{role} node {index} ({node.type})")
            checked += 1
    assert checked > 20
