import torch
from torch import nn
from core.attention import MultiHeadedAttention
from core.embedding import PositionalEncoding
from core.modules import MultiLayeredSepConv1d
from typing import Tuple, Optional

class EncoderLayer(nn.Module):
    """Encoder layer module

    :param int size: input dim
    :param espnet.nets.pytorch_backend.core.attention.MultiHeadedAttention self_attn: self attention module
    :param espnet.nets.pytorch_backend.core.positionwise_feed_forward.PositionwiseFeedForward feed_forward:
        feed forward module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(self, size, self_attn, feed_forward, dropout_rate,
                 normalize_before=True, concat_after=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = torch.nn.LayerNorm(size)
        self.norm2 = torch.nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        #if self.concat_after:
        self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
          x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(self.self_attn(x, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)

        residual = x
        if self.normalize_before:
          x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)

        return x, mask

class Encoder(torch.nn.Module):
    """Transformer encoder module

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float attention_dropout_rate: dropout rate in attention
    :param float positional_dropout_rate: dropout rate after adding positional encoding
    :param str or torch.nn.Module input_layer: input layer type
    :param class pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied. i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)
    :param str positionwise_layer_type: linear of conv1d
    :param int positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
    :param int padding_idx: padding_idx for input_layer=embed
    """

    def __init__(self, idim: int,
                 attention_dim: int = 256,
                 attention_heads: int =2,
                 linear_units: int = 2048,
                 num_blocks: int = 4,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 attention_dropout_rate: float = 0.0,
                 input_layer: str = None,
                 pos_enc_class: torch.nn.Module = PositionalEncoding,
                 normalize_before: bool =True,
                 concat_after: bool =False,
                 positionwise_conv_kernel_sizes: list = [5, 25, 13, 9],
                 padding_idx: int =-1):

        super(Encoder, self).__init__()
        # if self.normalize_before:
        self.after_norm = torch.nn.LayerNorm(attention_dim)
        if isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer,
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer is None:
            self.embed = torch.nn.Sequential(
                pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise ValueError("unknown input_layer: " + input_layer)
        self.normalize_before = normalize_before

        # self.encoders = repeat(
        #     4,
        #     lambda: EncoderLayer(
        #         attention_dim,
        #         MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
        #         positionwise_layer(*positionwise_layer_args),
        #         dropout_rate,
        #         normalize_before,
        #         concat_after
        #     )
        # )
        self.encoders_ = nn.ModuleList([
            EncoderLayer(
                attention_dim,
                MultiHeadedAttention(attention_heads, attention_dim, attention_dropout_rate),
                MultiLayeredSepConv1d(attention_dim, linear_units, positionwise_conv_kernel_sizes[i], dropout_rate),
                dropout_rate,
                normalize_before,
                concat_after
            ) for i in range(num_blocks)])




    def forward(self, xs: torch.Tensor, masks: Optional[torch.Tensor] = None):
        """Embed positions in tensor

        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        # if isinstance(self.embed, Conv2dSubsampling):
        #     xs, masks = self.embed(xs, masks)
        # else:
        xs = self.embed(xs)

        #xs, masks = self.encoders_(xs, masks)
        for encoder in self.encoders_:
            xs, masks = encoder(xs, masks)
        if self.normalize_before:
            xs = self.after_norm(xs)

        return xs, masks
