# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

"""
Adapted from:
1. https://github.com/meta-llama/codellama/blob/main/llama/model.py
2. https://github.com/naver-ai/rope-vit
3. https://github.com/lucidrains/rotary-embedding-torch
"""

from typing import Optional

import torch
from einops import rearrange, repeat
from torch import broadcast_tensors, nn


def init_t_xy(end_x: int, end_y: int, scale: float = 1.0, offset: int = 0, device=None):
    t = torch.arange(end_x * end_y, dtype=torch.float32, device=device)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    theta: float = 10000.0,
    scale_pos: float = 1.0,
    offset: int = 0,
    device=None,
):
    freqs_x = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )
    freqs_y = 1.0 / (
        theta ** (torch.arange(0, dim, 4, device=device)[: (dim // 4)].float() / dim)
    )

    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset, device=device)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[-2], x.shape[-1])
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    # Real-valued rotary embedding (no complex tensors). MPS doesn't support
    # several complex ops (e.g. repeat() / complex mul) and complex buffers are
    # often left on CPU, mixing devices. Doing the rotation with the real and
    # imaginary parts as plain real tensors is mathematically identical —
    # (a+bi)(c+di) = (ac - bd) + (ad + bc) i — and runs on CPU/MPS/CUDA alike.
    xq_r = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xq_re, xq_im = xq_r[..., 0], xq_r[..., 1]
    # Split freqs into real/imag while still on its own device, then move the
    # plain real tensors onto the query device (never a complex tensor on MPS).
    f_re = reshape_for_broadcast(freqs_cis.real, xq_re).to(xq.device)
    f_im = reshape_for_broadcast(freqs_cis.imag, xq_re).to(xq.device)
    xq_out = (
        torch.stack(
            [xq_re * f_re - xq_im * f_im, xq_re * f_im + xq_im * f_re], dim=-1
        )
        .flatten(3)
        .type_as(xq)
    )
    if xk.shape[-2] == 0:
        # no keys to rotate, due to dropout
        return xq_out, xk
    xk_r = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xk_re, xk_im = xk_r[..., 0], xk_r[..., 1]
    # repeat freqs along seq_len dim to match k seq_len (real tensors -> OK on MPS)
    if repeat_freqs_k:
        r = xk_re.shape[-2] // xq_re.shape[-2]
        f_re = f_re.repeat(*([1] * (f_re.ndim - 2)), r, 1)
        f_im = f_im.repeat(*([1] * (f_im.ndim - 2)), r, 1)
    xk_out = (
        torch.stack(
            [xk_re * f_re - xk_im * f_im, xk_re * f_im + xk_im * f_re], dim=-1
        )
        .flatten(3)
        .type_as(xk)
    )
    return xq_out, xk_out


def complex_mult(xq_real, xq_imag, freqs_cis_real, freqs_cis_imag):
    # Compute the real part of the product
    real_part = xq_real * freqs_cis_real - xq_imag * freqs_cis_imag
    # Compute the imaginary part of the product
    imag_part = xq_real * freqs_cis_imag + xq_imag * freqs_cis_real
    # Stack the real and imaginary parts along the last dimension
    return torch.stack([real_part, imag_part], dim=-1)


def apply_rotary_enc_real(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_real: torch.Tensor,
    freqs_cis_imag: torch.Tensor,
    repeat_freqs_k: bool = False,
):
    assert xk is not None
    assert xk.shape[-2] != 0

    xq_real = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 0]
    xq_imag = xq.float().reshape(*xq.shape[:-1], -1, 2)[..., 1]
    xk_real = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 0]
    xk_imag = xk.float().reshape(*xk.shape[:-1], -1, 2)[..., 1]
    freqs_cis_real = reshape_for_broadcast(freqs_cis_real, xq_real)
    freqs_cis_imag = reshape_for_broadcast(freqs_cis_imag, xq_imag)
    xq_out = complex_mult(xq_real, xq_imag, freqs_cis_real, freqs_cis_imag).flatten(3)
    if repeat_freqs_k:
        r = xk_real.shape[-2] // xq_real.shape[-2]
        freqs_cis_real = freqs_cis_real.repeat(*([1] * (freqs_cis_real.ndim - 2)), r, 1)
        freqs_cis_imag = freqs_cis_imag.repeat(*([1] * (freqs_cis_imag.ndim - 2)), r, 1)
    xk_out = complex_mult(xk_real, xk_imag, freqs_cis_real, freqs_cis_imag).flatten(3)
    # xq_out = torch.view_as_real(torch.complex(xq_real, xq_imag) * torch.complex(freqs_cis_real, freqs_cis_imag)).flatten(3)
    # xk_out = torch.view_as_real(torch.compelx(xk_real, xk_imag) * torch.complex(freqs_cis_real, freqs_cis_imag)).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


# rotary embedding helper functions
def broadcat(tensors, dim=-1):
    broadcasted_tensors = broadcast_tensors(*tensors)
    return torch.cat(broadcasted_tensors, dim=dim)


def rotate_half(x: torch.Tensor):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class VisionRotaryEmbeddingVE(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        pt_seq_len: Optional[int] = None,
        theta: float = 10000.0,
        offset: int = 1,  # specific to VE
    ):
        super().__init__()

        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        scale = 1.0
        if pt_seq_len is not None:
            scale = pt_seq_len / seq_len

        # offset of +1 following VE - even though for the
        # attention op only differences matter
        t = torch.arange(seq_len) * scale + offset

        freqs = torch.einsum("..., f -> ... f", t, freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        freqs = broadcat((freqs[None, :, :], freqs[:, None, :]), dim=-1)
        freqs_cos = freqs.cos().view(-1, freqs.shape[-1])
        freqs_sin = freqs.sin().view(-1, freqs.shape[-1])

        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def forward(self, t: torch.Tensor):
        return t * self.freqs_cos + rotate_half(t) * self.freqs_sin
