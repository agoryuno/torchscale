# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import pytest
import torch

from torchscale.architecture.config import EncoderConfig
from torchscale.architecture.encoder import Encoder

testcases = [
    {},
    {"vocab_size": 64000},
    {"activation_fn": "relu"},
    {"drop_path_rate": 0.1},
    {"encoder_normalize_before": False},
    {"no_scale_embedding": False},
    {"layernorm_embedding": True},
    {"rel_pos_buckets": 32, "max_rel_pos": 256},
    {"deepnorm": True, "subln": False, "encoder_normalize_before": False},
    {"bert_init": True},
    {"multiway": True},
    {"share_encoder_input_output_embed": True},
    {"checkpoint_activations": True},
    {"fsdp": True},
]


@pytest.mark.parametrize("args", testcases)
def test_encoder(args):
    config = EncoderConfig(**args)
    model = Encoder(config)
    token_embeddings = torch.rand(2, 10, config.encoder_embed_dim)
    model(src_tokens=None, token_embeddings=token_embeddings)


def test_embeds():
    """
    Add an embedding layer through the constructor
    and pass `src_tokens` into `forward()`.
    """
    from torchscale.component.embedding import TextEmbedding
    import numpy as np
    config = EncoderConfig(
                           vocab_size=64000, 
                           encoder_embed_dim=512,
                           encoder_attention_heads=8,
                           )
    model = Encoder(config, embed_tokens=TextEmbedding(config.vocab_size, config.encoder_embed_dim))
    src_tokens = torch.from_numpy(np.random.randint(1, config.vocab_size, 16)).unsqueeze(0)
    model(src_tokens)
