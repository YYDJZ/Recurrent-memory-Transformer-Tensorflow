from collections import namedtuple
from functools import wraps
from packaging import version

import tensorflow as tf
from tensorflow import keras
from einops import rearrange
import math
# constants

Config = namedtuple('EfficientAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)
def maskedfill(t, mask, value):
    mask = tf.cast(mask, tf.bool)  # 将掩码转换为布尔类型
    neg_mask = tf.logical_not(mask)  # 反转掩码
    masked_tensor = tf.where(mask, value, t)  # 根据掩码条件进行替换
    filled_tensor = tf.where(neg_mask, t, masked_tensor)  # 根据反转的掩码条件进行填充

    return filled_tensor
#Since tensorflow does not have inner function of self-attention calculation, we have to inplement it by ourselves.
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> tf.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.shape[-2], key.shape[-2]
    scale_factor = 1 / math.sqrt(query.shape[-1]) if scale is None else scale
    attn_bias = tf.zeros((L, S), dtype=query.dtype)
    if is_causal:
        assert attn_mask is None
        temp_mask = tf.ones((L, S), dtype=tf.bool)
        temp_mask=tf.linalg.band_part(temp_mask,-1,0)
        attn_bias=maskedfill(attn_bias,tf.logical_not(temp_mask),float("-inf"))
        attn_bias=tf.cast(attn_bias,dtype=query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == tf.bool:
            attn_mask=maskedfill(attn_mask,tf.logical_not(attn_mask),True)
        else:
            attn_bias += attn_mask
    attn_weight = query @ tf.transpose(key, perm=[0,1,3,2]) * scale_factor
    attn_weight += attn_bias
    attn_weight = tf.nn.softmax(attn_weight, axis=-1)
    attn_weight = tf.nn.dropout(attn_weight, dropout_p)
    return attn_weight @ value


class Attend(keras.Model):
    def __init__(
        self,
        dropout = 0.,
        causal = False,
        use_flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = keras.layers.Dropout(dropout)

        self.causal = causal
        #self.mask = tf.Variable(initial_value=None, trainable=False, dtype=tf.float32)
        self.mask=None

        self.use_flash = use_flash


    def get_mask(self, n):
        if exists(self.mask) and self.mask.shape[-1] >= n:
            return self.mask[:n, :n]
        self.mask = tf.ones((n, n), dtype=tf.bool)
        mask = tf.ones((n, n), dtype=tf.bool)
        mask=tf.linalg.band_part(mask, 0, -1)
        self.mask.assign(mask)
        return mask

    def flash_attn(self, q, k, v, mask = None):
        _, heads, q_len, _, k_len = *q.shape, k.shape[-2]

        # Check if mask exists and expand to compatible shape
        #The dimension of mask should be 4
        # The mask is B L, so it would have to be expanded to B H N L,In tf the function(rank) is used to calculate the dimension
        if exists(mask):
            if tf.rank(mask)!= 4:
                mask = rearrange(mask, 'b j -> b 1 1 j')

            t_shape = [mask.shape[0], heads, q_len, mask.shape[3]]
            #mask = mask.expand(-1, heads, q_len, -1)
            mask=tf.broadcast_to(mask,t_shape)

        out = scaled_dot_product_attention(
                q, k, v,
                attn_mask = mask,
                dropout_p = self.dropout if self.trainable else 0.,
                is_causal = self.causal
            )
        return out

    def call(self, q, k, v, mask=None):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        both pytorch and tensorflow has inner einsum
        """

        n = q.shape[-2]

        scale = q.shape[-1] ** -0.5

        if self.use_flash:
            return self.flash_attn(q, k, v, mask=mask)


        sim = tf.einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # key padding mask

        if exists(mask):
            if tf.rank(mask)!=4:
                mask = rearrange(mask, 'b j -> b 1 1 j')
            #sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)
            sim=maskedfill(sim,~mask,-1*sim.dtype.max)
        # causal mask

        if self.causal:
            causal_mask = self.get_mask(n,)
            #sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
            sim = maskedfill(sim, causal_mask, -1 * sim.dtype.max)

        # attention

        attn=tf.nn.softmax(sim,axis=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = tf.einsum("b h i j, b h j d -> b h i d", attn, v)

        return out