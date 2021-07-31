import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from local_attention import LocalAttention


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed  # nn.Sequential()
        self.tgt_embed = tgt_embed  # nn.Sequential()
        self.generator = generator  # nn.Sequential()

    def forward(self, src, tgt, src_mask, tgt_mask, src_tgt=True):
        "Take in and process masked src and target sequences."
        re = self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)
        if not hasattr(self.encoder.layers[0].self_attn, 'attn'):
            return re
        return re

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


def attention(query, key, value, mask=None, dropout=None):
    """
        single attention calculation

        Args:
            query: vector Q
            key: vector K
            value: vector V
            mask: shows which grid would have a score of -1e9, matrix
            dropout: not used
            is_src: is source point net
            overlap: ratio of the point net after removing the non-relevant points. Only considered when is_src=True

        Returns:
            self-attention result, softmax result
        """
    b, h, num_v, _ = value.size()
    num_q = value.size(2)
    weight = torch.ones(size=(b, h, num_q, num_v), device=torch.cuda.current_device())
    weight = weight / num_v

    return torch.matmul(weight, value), weight


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=None):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


# DecoderLayer(self.emb_dims, c(attn), c(ff), self.dropout)
class EncoderLayer(nn.Module):
    # EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N)
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size  # 512

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 512//4
        self.h = h
        # clones: deepcopy
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # query=key=value=[B,1024,512]
        # 1) Do all the linear projections in batch from d_model => h x d_k

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]
        # [B,4,1024,128]=q=k=v
        # 2) Apply attention on all the projected vectors in batch.
        x = torch.mean(value, dim=2, keepdim=True)
        # x, self.attn = attention(query, key, value, mask=mask,dropout=self.dropout)

        # self.attn = torch.sum(self.attn,dim=1)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.norm = nn.Sequential()  # nn.BatchNorm1d(d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = None

    def forward(self, x):
        return self.w_2(self.norm(F.relu(self.w_1(x)).transpose(2, 1).contiguous()).transpose(2, 1).contiguous())


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def empty(tensor):
    return tensor.numel() == 0


# kernel functions

# transcribed from jax to pytorch from
# https://github.com/google-research/google-research/blob/master/performer/fast_self_attention/fast_self_attention.py

def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    ratio = (projection_matrix.shape[0] ** -0.5)

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    diag_data = data ** 2
    diag_data = torch.sum(diag_data, dim=-1)
    diag_data = (diag_data / 2.0) * (data_normalizer ** 2)
    diag_data = diag_data.unsqueeze(dim=-1)

    if is_query:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data -
                          torch.max(data_dash, dim=-1, keepdim=True).values) + eps)
    else:
        data_dash = ratio * (
                torch.exp(data_dash - diag_data - torch.max(data_dash)) + eps)

    return data_dash


def generalized_kernel(data, *, projection_matrix, kernel_fn=nn.ReLU(), kernel_epsilon=0.001, normalize_data=True,
                       device=None):
    b, h, *_ = data.shape

    data_normalizer = (data.shape[-1] ** -0.25) if normalize_data else 1.

    if projection_matrix is None:
        return kernel_fn(data_normalizer * data) + kernel_epsilon

    projection = repeat(projection_matrix, 'j d -> b h j d', b=b, h=h)

    data_dash = torch.einsum('...id,...jd->...ij', (data_normalizer * data), projection)

    data_prime = kernel_fn(data_dash) + kernel_epsilon
    return data_prime


# non-causal linear attention
def linear_attention(q, k, v):
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k.sum(dim=-2))
    context = torch.einsum('...nd,...ne->...de', k, v)
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
    return out


# efficient causal linear attention, created by EPFL
def causal_linear_attention(q, k, v):
    from fast_transformers.causal_product import CausalDotProduct
    return CausalDotProduct.apply(q, k, v)


# inefficient causal linear attention, without cuda code, for reader's reference
# not being used
def causal_linear_attention_noncuda(q, k, v):
    k_cumsum = k.cumsum(dim=-2)
    context = torch.einsum('...nd,...ne->...nde', k, v)
    context = context.cumsum(dim=-3)
    context /= k_cumsum.unsqueeze(dim=-1)
    out = torch.einsum('...nde,...nd->...ne', context, q)
    return out


def orthogonal_matrix_chunk(cols, qr_uniform_q=False, device=None):
    unstructured_block = torch.randn((cols, cols), device=device)
    q, r = torch.qr(unstructured_block.cpu(), some=True)
    q, r = map(lambda t: t.to(device), (q, r))

    # proposed by @Parskatt
    # to make sure Q is uniform https://arxiv.org/pdf/math-ph/0609050.pdf
    if qr_uniform_q:
        d = torch.diag(r, 0)
        q *= d.sign()
    return q.t()


def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling=0, qr_uniform_q=False, device=None):
    nb_full_blocks = int(nb_rows / nb_columns)

    block_list = []

    for _ in range(nb_full_blocks):
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q)

    remaining_rows = nb_rows - nb_full_blocks * nb_columns
    if remaining_rows > 0:
        q = orthogonal_matrix_chunk(nb_columns, qr_uniform_q=qr_uniform_q, device=device)
        block_list.append(q[:remaining_rows])

    final_matrix = torch.cat(block_list)

    if scaling == 0:
        multiplier = torch.randn((nb_rows, nb_columns), device=device).norm(dim=1)
    elif scaling == 1:
        multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device=device)
    else:
        raise ValueError(f'Invalid scaling {scaling}')

    return torch.diag(multiplier) @ final_matrix


class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features=None, redraw_projection=True, ortho_scaling=0, causal=False,
                 generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False):
        super().__init__()
        nb_features = default(nb_features, int(dim_heads * math.log(dim_heads)))

        self.dim_heads = dim_heads
        self.nb_features = nb_features
        self.ortho_scaling = ortho_scaling
        self.redraw_projection = redraw_projection
        self.qr_uniform_q = qr_uniform_q

        self.create_projection = gaussian_orthogonal_random_matrix

        self.generalized_attention = generalized_attention
        self.kernel_fn = kernel_fn

        if not redraw_projection:
            self.set_projection_matrix()

        self.causal = causal
        if causal:
            try:
                import fast_transformers.causal_product.causal_product_cuda
                self.causal_linear_fn = causal_linear_attention
            except ImportError:
                print(
                    'unable to import cuda code for auto-regressive Performer. will default to the memory inefficient non-cuda version')
                self.causal_linear_fn = causal_linear_attention_noncuda

    def set_projection_matrix(self, device):
        projection_matrix = self.create_projection(device=device, nb_rows=self.nb_features, nb_columns=self.dim_heads,
                                                   scaling=self.ortho_scaling, qr_uniform_q=self.qr_uniform_q)
        self.register_buffer('projection_matrix', projection_matrix)

    def forward(self, q, k, v):
        device = q.device

        if self.redraw_projection and not hasattr(self, 'projection_matrix'):
            projection_matrix = self.create_projection(device=device, nb_rows=self.nb_features,
                                                       nb_columns=self.dim_heads, scaling=self.ortho_scaling,
                                                       qr_uniform_q=self.qr_uniform_q)
        else:
            projection_matrix = self.projection_matrix

        if self.generalized_attention:
            q = generalized_kernel(q, kernel_fn=self.kernel_fn, projection_matrix=projection_matrix, device=device,
                                   is_query=True)
            k = generalized_kernel(k, kernel_fn=self.kernel_fn, projection_matrix=projection_matrix, device=device,
                                   is_query=False)
        else:
            q = softmax_kernel(q, projection_matrix=projection_matrix, device=device, is_query=True)
            k = softmax_kernel(k, projection_matrix=projection_matrix, device=device, is_query=False)

        attn_fn = linear_attention if not self.causal else self.causal_linear_fn
        out = attn_fn(q, k, v)
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, causal=False, heads=8, local_heads=0, local_window_size=256, nb_features=None,
                 redraw_projection=True, generalized_attention=False, kernel_fn=nn.ReLU(), qr_uniform_q=False,
                 dropout=0.):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.fast_attention = FastAttention(dim // heads, nb_features, redraw_projection, causal=causal,
                                            generalized_attention=generalized_attention, kernel_fn=kernel_fn,
                                            qr_uniform_q=qr_uniform_q)

        self.heads = heads
        self.d_k = dim // self.heads
        self.global_heads = heads - local_heads
        self.local_attn = LocalAttention(window_size=local_window_size, causal=causal, autopad=True, dropout=dropout,
                                         look_forward=int(not causal)) if local_heads > 0 else None

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        b, n, _, h, gh = *q.shape, self.heads, self.global_heads

        q, k, v = \
            [l(x).view(b, -1, self.heads, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip((self.to_q, self.to_k, self.to_v), (q, k, v))]

        (q, lq), (k, lk), (v, lv) = map(lambda t: (t[:, :gh], t[:, gh:]), (q, k, v))

        attn_outs = []

        if not empty(q):
            if exists(mask):
                global_mask = mask[:, None, :, None]
                k.masked_fill_(~global_mask, 0)

            out = self.fast_attention(q, k, v)
            attn_outs.append(out)

        if not empty(lq):
            out = self.local_attn(lq, lk, lv, input_mask=mask)
            attn_outs.append(out)

        out = torch.cat(attn_outs, dim=1)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)


class Transformer(nn.Module):
    def __init__(self, model_params):
        super(Transformer, self).__init__()
        self.emb_dims = model_params.emb_dims
        self.N = 1
        self.dropout = 0.0
        self.ff_dims = model_params.ff_dims
        self.n_heads = 4
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        src_attn = MultiHeadedAttention(self.n_heads, self.emb_dims)

        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.emb_dims, c(attn), c(src_attn), c(ff), self.dropout),
                                            self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None, src_tgt=False).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None, src_tgt=True).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding
