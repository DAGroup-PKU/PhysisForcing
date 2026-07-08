import logging

import torch
import torch.cuda.amp as amp

import project.distributed.unified_parallel as ulysses
from project.models.backbone.dit import rope_apply, sinusoidal_embedding_1d
from project.models.module import flash_attention
from project.utils.common import maybe_checkpoint

logger = logging.getLogger()


def ulysses_dit_forward(
    self,
    x,
    t,
    context,
    seq_len,
    y=None,
    cfg_scale=None,
    neg_embeds=None,
):
    """
    x:              A list of videos each with shape [C, T, H, W].
    t:              [B].
    context:        A list of text embeddings each with shape [L, C].
    """
    # params
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # embeddings
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # time embeddings
    if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
    with amp.autocast(dtype=torch.float32):
        bt = t.size(0)
        t = t.flatten()
        e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())

        e0 = self.time_projection(e).unflatten(2, (6, self.dim))
        assert e.dtype == torch.float32 and e0.dtype == torch.float32

        if self.config.use_cfg_emb:
            emb = self.cfg_embedder(cfg_scale, neg_embeds)
            assert emb.dtype == torch.float32
            e0 = e0 + emb

    # context
    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    # ulysses support
    sp_world = ulysses.get_unified_parallel_world_size()
    group = ulysses.get_unified_parallel_group()
    if seq_len % sp_world:
        padding_size = sp_world - (seq_len % sp_world)
        x = ulysses.pad_tensor(x, dim=1, padding_size=padding_size)
        # 对 e0 进行相同的填充和切片
        e0 = ulysses.pad_tensor(e0, dim=1, padding_size=padding_size)
    x = ulysses.Slice.apply(group, x, 1, True)
    e0 = ulysses.Slice.apply(group, e0, 1, True)

    # arguments
    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens)
        
    for block in self.blocks:
        x = maybe_checkpoint(block, x, enabled=self.gradient_checkpointing, **kwargs)

    # ulysses support
    x = ulysses.gather_outputs(x, gather_dim=1, padding_dim=1, unpad_dim_size=seq_len, scale_grad=True)

    # head
    x = self.head(x, e)

    # unpatchify
    x = self.unpatchify(x, grid_sizes)
    
    return [u.float() for u in x]


def ulysses_attn_forward(
    self,
    x,
    seq_lens,
    grid_sizes,
    freqs,
    dtype=torch.bfloat16
):
    b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
    seq_len = seq_lens.max()
    half_dtypes = (torch.float16, torch.bfloat16)

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # query, key, value function
    def qkv_fn(x):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        return q, k, v

    q, k, v = qkv_fn(x)

    # ulysses support
    sp_size = ulysses.get_unified_parallel_world_size()
    if n % sp_size:
        pad_size = sp_size - (n % sp_size)
        pad_size = pad_size * d
        pad_inner_dim = n * d + pad_size
        q = ulysses.pad_tensor(q, dim=2, padding_size=pad_size)
        k = ulysses.pad_tensor(k, dim=2, padding_size=pad_size)
        v = ulysses.pad_tensor(v, dim=2, padding_size=pad_size)
    else:
        pad_inner_dim = n * d

    qkv = torch.cat([q, k, v], dim=2)
    qkv = ulysses.gather_seq_scatter_heads_qkv(qkv, seq_dim=1, unpadded_dim_size=seq_len)
    q, k, v = qkv.split(pad_inner_dim // sp_size, dim=2)

    pad_n = pad_inner_dim // d
    pad_split_n = pad_n // sp_size
    q = q.view(b, seq_len, pad_split_n, d)
    k = k.view(b, seq_len, pad_split_n, d)
    v = v.view(b, seq_len, pad_split_n, d)

    q = rope_apply(q, grid_sizes, freqs)
    k = rope_apply(k, grid_sizes, freqs)

    x = flash_attention(
        q=half(q),
        k=half(k),
        v=half(v),
        k_lens=seq_lens,
        window_size=self.window_size
    )

    # ulysses support
    x = x.flatten(2)
    x = ulysses.gather_heads_scatter_seq(x, head_dim=2, seq_dim=1)
    if n % sp_size:
        x = ulysses.unpad_tensor(x, dim=2, padding_size=pad_size)

    x = self.o(x)
    return x
