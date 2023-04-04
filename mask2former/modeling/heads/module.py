import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention module for both image and text
    """

    def __init__(self, q_dim, k_dim, embed_dim, num_heads, dropout=0.1, 
        clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_dim = q_dim
        self.k_dim = k_dim

        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** (-0.5)
        self.dropout = dropout

        self.q_proj = nn.Linear(self.q_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.k_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.q_dim)
        self.clamp_min_for_underflow = clamp_min_for_underflow
        self.clamp_max_for_overflow = clamp_max_for_overflow

        self._reset_parameters()

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    def forward(self, q, k, v, attention_mask=None, return_attention=False):
        if len(q.size()) == 3:
            bsz, tgt_len, embed_dim = q.size()
        elif len(q.size()) == 2:
            tgt_len, embed_dim = q.size()
            bsz = k.shape[0]
            q = q.expand(bsz, tgt_len, embed_dim)

        query_states = self.q_proj(q) * self.scale
        key_states = self._shape(self.k_proj(k), -1, bsz)
        value_states = self._shape(self.v_proj(v), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )

        if self.clamp_min_for_underflow:
            attn_weights = torch.clamp(attn_weights, min=-50000) # Do not increase -50000, data type half has quite limited range
        if self.clamp_max_for_overflow:
            attn_weights = torch.clamp(attn_weights, max=50000) # Do not increase 50000, data type half has quite limited range

        if attention_mask is not None:
            # [bsz, src_len]
            assert (attention_mask.dim() == 2)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            attention_mask = attention_mask.expand(bsz, 1, tgt_len, src_len)
            attention_mask = attention_mask.masked_fill(attention_mask == 0, -9e15)

            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if return_attention:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)


        return attn_output, attn_weights



class ContextInteraction(nn.Module):
    def __init__(self, q_dim, k_dim, embed_dim, num_heads, hidden_dim=None, dropout=0.1,
                 drop_path=.0, init_values=1e-1, use_layer_scale = False,
                 clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super(ContextInteraction, self).__init__()

        # pre_layer norm
        self.layer_norm_q_1 = nn.LayerNorm(q_dim)
        self.layer_norm_k_1 = nn.LayerNorm(k_dim)
        self.attn = MultiHeadAttention(q_dim=q_dim,
                                       k_dim=k_dim,
                                       embed_dim=embed_dim,
                                       num_heads=num_heads,
                                       clamp_min_for_underflow=clamp_min_for_underflow,
                                       clamp_max_for_overflow=clamp_max_for_overflow)

        # add layer scale for training stability
        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.gamma = nn.Parameter(init_values * torch.ones((q_dim)), requires_grad=True)


    def forward(self, mask_features, transformer_encoder_features, multi_scale_features, text_feat, attention_mask):

        q0, q1, q2 = multi_scale_features[0], multi_scale_features[1], multi_scale_features[2]
        q3 = mask_features
        k = text_feat
        v = text_feat

        output = []
        bs, _, h, w = q0.shape
        k = k.expand(bs, k.shape[0], k.shape[1])
        v = v.expand(bs, v.shape[0], v.shape[1])
        for q_index, q in enumerate([q0, q1, q2, q3]):
            bs, _, h, w = q.shape
            q = q.flatten(2).transpose(1, 2)
            q = self.layer_norm_q_1(q)
            k, v = self.layer_norm_k_1(k), self.layer_norm_k_1(v)
            delta_q = self.attn(q, k, v, attention_mask=attention_mask)[0]
            if self.use_layer_scale:
                q = q + self.drop_path(self.gamma * delta_q)
            else:
                q = q + delta_q
            q = q.transpose(1, 2).contiguous().view(bs, -1, h, w)
            output.append(q)

        transed_multi_scale_features = output[:3]
        transed_mask_features = output[3]

        return (transed_mask_features, transformer_encoder_features, transed_multi_scale_features)

