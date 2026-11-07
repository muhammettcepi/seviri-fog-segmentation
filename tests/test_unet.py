import pytest

torch = pytest.importorskip("torch")

from fogseg.unet import UNet


def test_output_has_the_input_size_for_odd_sizes():
    out = UNet(14, 4, 2)(torch.zeros(2, 14, 13, 22))
    assert out.shape == (2, 13, 22)
    out = UNet(14, 2, 4)(torch.zeros(1, 14, 241, 462))
    assert out.shape == (1, 241, 462)


def test_padding_keeps_pixels_in_place():
    net = UNet(1, 4, 2).eval()
    x = torch.randn(1, 1, 13, 22)
    padded = torch.nn.functional.pad(x, (0, 2, 0, 3))
    assert torch.allclose(net(x), net(padded)[:, :13, :22], atol=1e-5)


def test_weight_count_of_the_configured_net():
    weights = sum(p.numel() for p in UNet(14, 16, 4).parameters())
    assert 1_500_000 < weights < 2_500_000
