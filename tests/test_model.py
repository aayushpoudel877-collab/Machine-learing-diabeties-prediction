import torch
from sentinel_vision.model import build_model

def test_model_output_shape():
    assert build_model(6,False)(torch.randn(2,3,224,224)).shape==(2,6)

def test_model_has_trainable_head():
    assert any(p.requires_grad for p in build_model(6,False).fc.parameters())
