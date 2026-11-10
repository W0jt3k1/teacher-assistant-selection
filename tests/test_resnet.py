import pytest
import torch
import torch.nn as nn

from src.models.resnet import ResNet


@pytest.mark.parametrize("n", [1, 5, 18])
def test_forward(n):
    model = ResNet(n=n)
    X = torch.randn(4, 3, 32, 32)

    output = model(X)

    assert output.shape == (4, 100)


@pytest.mark.parametrize("n", [1, 5, 18])
def test_backward(n):
    model = ResNet(n=n)

    X = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 100, (4,))

    criterion = nn.CrossEntropyLoss()

    output = model(X)
    loss = criterion(output, y)

    loss.backward()

    assert torch.isfinite(loss)

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient: {name}"
            assert torch.isfinite(param.grad).all(), f"Invalid gradient: {name}"
