import math
from functools import partial
from itertools import zip_longest
from contextlib import nullcontext
import tensorflow as tf
from tensorflow import keras
from einops import rearrange, repeat, pack, unpack
import numpy as np
from typing import Optional, Tuple,List
linear=partial(keras.layers.Dense,use_bias=False)
from attend import  Attend


#This part of code references https://github.com/lucidrains/recurrent-memory-transformer-pytorch Original code inplement the function by pytorch,
#We inplement in Tensorflow instead.

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t


class EmptyContextManager:
    def __enter__(self):
        pass  # 在进入上下文时不执行任何操作

    def __exit__(self, exc_type, exc_value, traceback):
        pass  # 在退出上下文时不执行任何操作


#用于切换到验证模式
def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        is_training = self.trainable
        self.trainable=False
        out = fn(self, *args, **kwargs)
        self.trainable=is_training
        return out
    return inner


def default(*vals):
    for val in vals:
        if exists(val):
            return val
    return None

def divisible_by(n, d):
    return (n % d) == 0


def log(x, eps = 1e-20):
    return tf.math.log(tf.maximum(x,eps))


def gumbel_noise(x):
    noise = tf.random.uniform(tf.shape(x), minval=0, maxval=1, dtype=tf.float32)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., axis = -1):
    return tf.argmax(((t / max(temperature, 1e-10)) + gumbel_noise(t)),axis=axis)

def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    values, indices = tf.nn.top_k(logits, k=k)
    probs = tf.fill(tf.shape(logits), float('-inf'))
    probs=np.array(probs)
    for i in range(len(probs)):
            probs[i][indices[i][0]]=values[i][0]
    return tf.convert_to_tensor(probs)


def top_k(logits, thres=0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    values, indices = tf.nn.top_k(logits, k=k)
    probs = tf.fill(tf.shape(logits), float('-inf'))
    probs=np.array(probs)
    for i in range(len(probs)):
            probs[i][indices[i][0]]=values[i][0]
    return tf.convert_to_tensor(probs)

#Token shift function
def token_shift_fn(t, ps):
    read_mem, t, write_mem = unpack(t, ps, 'b * d')
    shape=tf.shape(t)
    split_size = shape[-1]//2
    t0 = t[..., :split_size]
    t_shift = t[..., split_size:]
    padd=[]
    for i in range(len(t_shift.shape)-2):
        padd.append([0,0])
    padd.append([1,0])
    padd.append([0,0])
    t_shift = tf.pad(t_shift, paddings=tf.constant(padd), mode="CONSTANT",constant_values = 0.)
    t_shift=t_shift[:,:-1,:]
    t0 = tf.concat([t0, t_shift], axis = -1)
    return tf.concat([read_mem, t0, write_mem], axis = -2)
    


def frac_gradient(x, frac = 1.):
    if frac == 1.:
        return x
    newx=x[:]
    return x * frac + newx * (1. - frac)


# Rotary embedding

class RotaryEmbedding(tf.keras.Model):
    def __init__(self, dim, theta = 32768):
        super().__init__()
        ts=tf.cast(tf.range(0, dim, 2),dtype=tf.float32)
        inv_freq = 1. / (theta ** (ts / dim))
        self.buffer_inv_freq=tf.Variable(inv_freq,trainable=False)

    def call(self, positions):
        freqs = tf.einsum('i , j -> i j', positions, self.inv_freq)
        freqs = tf.concat([freqs, freqs], axis= -1)
        return freqs

def rotate_half(x):
    shape=tf.shape(x)
    split_size = shape[-1]//2
    x1 = x[..., :split_size]
    x2 = x[..., split_size:]
    return tf.concat([-x2, x1], axis=-1)

def apply_rotary_pos_emb(pos, t):
    return (t * tf.math.cos(pos)) + (rotate_half(t) * tf.math.sin(pos))


class RMSNorm(tf.keras.Model):
    def __init__(self, dim):
        super(RMSNorm,self).__init__()
        self.scale = dim ** 0.5
        self.gamma = tf.Variable(tf.ones((dim)))

    def call(self, x):
        return tf.nn.l2_normalize(x,  axis= -1) * self.scale * self.gamma

class GEGLU(keras.Model):
    def call(self, x):
        shape = tf.shape(x)
        split_size = shape[-1] // 2
        x0 = x[..., :split_size]
        gate = x[..., split_size:]
        return x0 * tf.nn.gelu(gate)

def FeedForward(dim, mult = 4):
    dim_inner = int(dim * mult * 2 / 3)
    return tf.keras.Sequential([
        linear(units=dim_inner * 2, use_bias = False),
        GEGLU(),
        RMSNorm(dim_inner),
        linear( units=dim,use_bias = False)]
    )

class Attention(keras.Model):
    def __init__(
        self,
        *,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        use_flash_attn = False,
        use_custom_causal_attn_mask = False
    ):
        super(Attention,self).__init__()
        dim_inner = dim_head * heads
        self.heads = heads

        self.attend = Attend(
            causal = causal and not use_custom_causal_attn_mask,
            dropout = dropout,
            use_flash = use_flash_attn
        )

        self.null_kv = tf.Variable(tf.random.normal((2, heads, dim_head)))

        self.to_q = linear(units=dim_inner,)
        self.to_kv = linear( units=dim_inner * 2,)
        self.to_out = linear(units=dim_inner,)
    
    
    def call(
    self,
    x,
    rotary_emb: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
    mask = None,
    xl_memories = None):
    
        h = self.heads
    
        q = self.to_q(x)
        tokv=self.to_kv(x)
        shape = tf.shape(tokv)
        split_size = shape[-1] // 2
        k = tokv[..., :split_size]
        v = tokv[..., split_size:]
        del tokv
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        # add a null key / value
        # to protect against an entirely masked out sequence
        # as well as giving attention ability to attend to nothing
    
        nk, nv = map(lambda t: repeat(t, 'h d -> b h 1 d', b = x.shape[0]), self.null_kv)
        
        k =tf.concat([nk, k], axis = -2)
        v = tf.concat([nv, v], axis = -2)
    
        if exists(mask):
            padd=[]
            for i in range(tf.rank(mask)-1):
                padd.append([0,0])
            padd.append([1,0])
            mask = tf.pad(mask, paddings=padd, constant_values= True)
    
        # manage memories
        next_xl_memories = tf.stack([k, v])
    
        if exists(xl_memories):
            kx, vx = xl_memories
            k = tf.concat([kx, k], axis = -2)
            v = tf.concat([vx, v], axis = -2)
    
            if exists(mask):
                #mask = F.pad(mask, (xl_memories.shape[-2], 0), value = True)
                padd = []
                for i in range(tf.rank(mask) - 1):
                    padd.append([0, 0])
                padd.append([xl_memories.shape[-2], 0])
                mask = tf.pad(mask, paddings=padd, constant_values=True)
    
        if exists(rotary_emb):
            q_rotary_emb, k_rotary_emb = rotary_emb
    
            q = apply_rotary_pos_emb(q_rotary_emb, q)
            k = apply_rotary_pos_emb(k_rotary_emb, k)
    
        out = self.attend(q, k, v, mask = mask)
    
        out = rearrange(out, 'b h n d -> b n (h d)')
    
        return self.to_out(out), next_xl_memories




class RecurrentMemoryTransformer(tf.keras.Model):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        num_memory_tokens,
        seq_len,
        causal = True,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        use_flash_attn = False,
        ignore_index = -1,
        abs_pos_emb = True,
        rotary_pos_emb = False,
        token_shift = True,
        use_xl_memories = True,
        xl_mem_len = None,
        enhanced_xl_recurrence = False,      # add simple method for enhancing receptive field of xl memories, from ernie-doc paper
        emb_gradient_frac = 0.1,             # trick from cogview paper that leads to a bit more stability
        memory_not_causal = True,            # flash attention behaves a bit more optimally if causal mask is not explicitly passed in - but if the memories perform better without a causal mask, it is necessary to have this turned on
        add_write_to_next_write_mem = False, # add the write memories of previous step to the next write step - thanks to @IcarusWizard for pointing out this discrepancy
        next_write_mem_stop_grad = True,     # whether to stop gradient of previous read memory -> next write memory
        always_have_read_memories = True,    # whether to always have read memories, even on the first step, so to make the model onnx-able
        resi_dual_scale = 1.,                # in the case of overflows in fp16 on the prenorm branch, set this to a value less than 1.
    ):

        super(RecurrentMemoryTransformer,self).__init__()
        self.causal = causal
        self.seq_len = seq_len

        self.emb_gradient_frac = emb_gradient_frac
        assert 0 < resi_dual_scale <= 1., 'The resiDual scale must be within 0 and 1'
        self.resi_dual_scale = resi_dual_scale
        assert num_memory_tokens > 0,'Nums of memory tokens must be greater than 0'
        self.token_emb = tf.keras.layers.Embedding(num_tokens, dim)

        assert any([abs_pos_emb, rotary_pos_emb, token_shift])

        self.pos_emb = tf.keras.layers.Embedding(seq_len, dim) if abs_pos_emb else None

        self.rotary_pos_emb = RotaryEmbedding(dim_head) if rotary_pos_emb else None

        self.maybe_token_shift = token_shift_fn if token_shift else identity

        self.num_memory_tokens = num_memory_tokens

        self.read_memory_emb = tf.Variable(tf.random.normal((num_memory_tokens, dim),stddev=0.02))

        self.memory_tokens = tf.Variable(tf.random.normal((num_memory_tokens, dim),stddev=0.02))

        # xl memories

        xl_mem_len = default(xl_mem_len, seq_len)
        assert xl_mem_len <= seq_len,"The xl_men_len must not exceed seq_len"
        self.xl_mem_len = xl_mem_len

        self.use_xl_memories = use_xl_memories
        self.enhanced_xl_recurrence = enhanced_xl_recurrence

        # the tensorflow seems not to have modellist,so we have to build model in another way.



        self.depth=depth
        for i in range(depth*4):

            if i%4==0:
                exec(f"self.lays{i}=Attention(dim=dim,dim_head=dim_head,causal=causal,heads=heads,use_flash_attn=use_flash_attn,use_custom_causal_attn_mask=memory_not_causal)")
            elif i%4==1:
                exec(f"self.lays{i}=RMSNorm(dim)")
            elif i%4==2:
                exec(f"self.lays{i}=FeedForward(dim=dim, mult=ff_mult)")
            else:
                exec(f"self.lays{i}=RMSNorm(dim)")
        self.norm = RMSNorm(dim)
        self.to_logits = keras.layers.Dense(num_tokens)

        self.ignore_index = ignore_index

        # whether to use custom attention mask if causal and memory should not be causal

        self.use_custom_causal_attn_mask = causal and memory_not_causal


        self.add_write_to_next_write_mem = add_write_to_next_write_mem
        self.next_write_mem_stop_grad = next_write_mem_stop_grad

        # allow for attending to raw read memory positional embeddings on first step
        # hack to make it onnx-able and should not hurt

        self.always_have_read_memories = always_have_read_memories

    def init_memory(self, batch):
        return repeat(self.memory_tokens, 'm d -> b m d', b = batch)

    def call(
        self,
        x,
        read_memories = None,
        *,
        mask = None,
        labels = None,
        xl_memories: Optional[List[tf.Tensor]] = None,
        mask_out_read_memories = False   # in the case one is passing in 0s for read memories, for onnx-able model
    ):
        has_xl_memories = exists(xl_memories) and len(xl_memories) > 0

        # Need to deal with device
        b, n, device, mem_length, return_loss = *x.shape, x.device, self.num_memory_tokens, exists(labels)

        assert n <= self.seq_len
        pos = tf.range(n)
        # maybe absolute positional embedding
        x = self.token_emb(x)
        
        if exists(self.pos_emb):
            x = tf.cast(x,dtype=tf.float32) + self.pos_emb(pos)

        x = frac_gradient(x, self.emb_gradient_frac)
        write_memories = self.init_memory(b)
        
        # DETACH in pytorch is equal to stop gradient in tensorflow
        if exists(read_memories):
            if tf.rank(read_memories) == 2:
                read_memories = repeat(read_memories, 'n d -> b n d', b = b)

            read_mem_length = mem_length
            read_memories = read_memories + self.read_memory_emb
        elif self.always_have_read_memories:
            read_mem_length = mem_length
            read_memories = repeat(self.read_memory_emb, 'n d -> b n d', b = b)
        else:
            read_mem_length = 0
            read_memories = x[:, 0:0]

        # concat to main sequence using einop's pack
        x, ps = pack([read_memories, x, write_memories], 'b * d')
        # take care of mask
        if exists(mask):
            padd=[]
            for i in range(tf.rank(mask)-1):
                padd.append([0,0])
            padd.append([read_mem_length, mem_length])
            mask = tf.pad(mask,paddings=tf.constant(padd), constant_values=True)

        # custom causal mask, if needed

        if self.use_custom_causal_attn_mask:
            causal_mask = tf.ones((n, n),dtype = tf.bool)
            causal_mask=tf.linalg.band_part(causal_mask, -1, 0)
            padd=[]
            for i in range(tf.rank(causal_mask)-2):
                padd.append([0,0])
            padd.append([read_mem_length,0])
            padd.append([0,mem_length])
            causal_mask = tf.pad(causal_mask, paddings=padd, constant_values= False)
            padd=[]
            for i in range(tf.rank(causal_mask)-2):
                padd.append([0,0])
            padd.append([0,mem_length])
            padd.append([read_mem_length,0])
            causal_mask = tf.pad(causal_mask, paddings=padd, constant_values = True)

            causal_mask = rearrange(causal_mask, 'i j -> 1 1 i j')

            if exists(mask):
                mask = rearrange(mask, 'b j -> b 1 1 j')
                mask = mask & causal_mask
            else:
                mask = causal_mask

        if read_mem_length > 0 and mask_out_read_memories:
            read_mem_mask = tf.range(x.shape[-2]) < read_mem_length

            if exists(mask):
                mask = mask & ~read_mem_mask
            else:
                mask = read_mem_mask



        rotary_emb = None
#Executes rotary embedding.Offset main positions by 10000(like traditional positional embedding), and keep all memories at position 0,
        if exists(self.rotary_pos_emb):
            mem_rel_dist = 10000

            q_pos = pos + mem_rel_dist

            if has_xl_memories:
                xl_mem_length = xl_memories[0].shape[-2]
                q_pos += xl_mem_length
                
            padd=[]
            for i in range(tf.rank(q_pos)-1):
                padd.append([0,0])
            padd.append([read_mem_length, mem_length])
            q_pos = tf.pad(q_pos, paddings=tf.constant(padd), value = 0)
            q_rotary_emb = self.rotary_pos_emb(q_pos)




            # the order of the keys are - [xl memories] [read memories] [main sequence] [ write memories]

            if has_xl_memories:
                k_pos = tf.range(xl_mem_length) + mem_rel_dist
                k_pos = tf.concat([k_pos, q_pos], axis = -1)
            else:
                k_pos = q_pos

            k_rotary_emb = self.rotary_pos_emb(k_pos)

            rotary_emb = (q_rotary_emb, k_rotary_emb)

        shift_fn = partial(self.maybe_token_shift, ps = ps)

        # prepare xl memories

        xl_memories = default(xl_memories, [])
        xl_memories_iter = iter(xl_memories)
        new_xl_memories = []

        residual = x * self.resi_dual_scale
        for i in range(self.depth):
            lays=[]
            exec(f"lays.append(self.lays{i*4})")
            exec(f"lays.append(self.lays{i * 4+1})")
            exec(f"lays.append(self.lays{i * 4+2})")
            exec(f"lays.append(self.lays{i * 4+3})")
            attn=lays[0]
            attn_post_norm=lays[1]
            ff=lays[2]
            ff_post_norm=lays[3]

            attn_out, xl_memories = attn(shift_fn(x), mask=mask, xl_memories=next(xl_memories_iter, None),
                                         rotary_emb=rotary_emb)
            new_xl_memories.append(xl_memories)
            x = attn_post_norm(x + attn_out)

            residual = residual + attn_out * self.resi_dual_scale
            ff_out = ff(shift_fn(x))
            x = ff_post_norm(x + ff_out)
            residual = residual + ff_out * self.resi_dual_scale
        next_xl_memories = None

        if self.use_xl_memories:
            next_xl_memories=[tf.stop_gradient(t[..., -self.xl_mem_len:, :]) for t in new_xl_memories]
        # Residual connect

        x = x + self.norm(residual)

        # split out memories using unpack

        read_memories, x, write_memories = unpack(x, ps, 'b * d')
        logits = self.to_logits(x)
        if not return_loss:
            return logits, write_memories, next_xl_memories

        #iglabels=masked_targets = tf.where(tf.equal(labels, self.ignore_index), -1 * tf.ones_like(labels), labels)
        #logits=rearrange(logits, 'b n c -> b c n')
        mask = tf.not_equal(labels, self.ignore_index)
        
        # Set the invalid label index to -1
        labels_masked = tf.where(mask, labels, -1)
        
        loss =  tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels_masked,)

        loss=tf.reduce_mean(loss)
        return loss, write_memories, next_xl_memories

#Use wrapper could manage segments. When construcing RecurrentMemoryTransformerWrapper, the trainable parameters are actually contained
# in Recurrent memory transformer.

class RecurrentMemoryTransformerWrapper(tf.keras.Model):
    def __init__(
        self,
        transformer: RecurrentMemoryTransformer,
        truncate_at_step = None  # number of steps before detaching memories (truncated bptt). with memory replay checkpointing, there should be no memory issues, but in case of instability, as reported in initial paper
    ):
        super().__init__()
        self.transformer = transformer
        self.seq_len = transformer.seq_len
        self.truncate_at_step = truncate_at_step

    @eval_decorator
    def generate(
            self,
            prime,
            *,
            length,
            memories=None,
            xl_memories: Optional[List[tf.Tensor]] = None,
            temperature=1.,
            filter_thres=0.9
    ):

    #Since tensorflow will not track gradient unless using gradienttape,so we do not need a function like @torch.no_grad()

        assert self.transformer.causal, 'only autoregressive transformers can generate'

        start_len, seq_len = prime.shape[-1], self.seq_len

        assert length >= start_len

        print(len(prime[0]),seq_len,start_len)
        *past_segments, curr_segment = np.split(prime,indices_or_sections=max(int(len(prime[0])//seq_len),1),axis=-1)
        print(past_segments)
        # catch memories up to the current segment

        for past_segment in past_segments:
            _, memories, xl_memories = self.transformer(past_segment, memories, xl_memories=xl_memories)

        # sample for the remaining length

        for ind in range(length - start_len):
            logits, next_memories, next_xl_memories = self.transformer(curr_segment, memories,xl_memories=xl_memories)

            logits = logits[:, -1]

            filtered_logits = top_k(logits, thres=filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature=temperature)
            sampled = rearrange(sampled, 'b -> b 1')

            curr_segment = tf.concat([curr_segment, sampled], axis=-1)

            if divisible_by(curr_segment.shape[-1] - 1, seq_len):
                memories = next_memories
                xl_memories = next_xl_memories

                past_segment, curr_segment = curr_segment[..., :seq_len], curr_segment[..., -1:]
                past_segments.append(past_segment)
        # add current segment to all segments

        past_segments.append(curr_segment)

        # reconcat all segments,tf.concat is equal to torch.cat
        #print("past:segments:",past_segments)


        output = tf.concat(past_segments, axis=-1)
        print(output.shape)
        output = output[:, start_len:]
        return output


    def call(
        self,
        x,
        memories = None,
        *,
        mask = None,
        xl_memories: Optional[List[tf.Tensor]] = None,
        return_loss = False,
        labels = None,
        truncate_at_step = None,         # if set, this would override the truncate_at_step at init
        memory_replay_backprop = False,  # whether to have the class do the backwards pass memory efficiently
        mrbp_loss_weight = 1.,    # if using memory replay backprop with gradient accumulation, scale loss by this factor ex. (1. / <num grad accum steps>)
        tp=None
    ):
        seq_len, truncate_at_step = self.seq_len, default(truncate_at_step, self.truncate_at_step)

        labels = None

        x, labels = x[:, :-1], x[:, 1:]

        # segment input

        segments = tf.split(x, int((x.shape[-1]) / seq_len), axis=-1)
        
        total_length = x.shape[-1]
        num_segments = len(segments)
        segment_length_frac = tuple(map(lambda t: t.shape[-1] / total_length, segments))

        # default values

        label_segments = mask_segments = (None,)

        if exists(labels):
            label_segments = tf.split(labels,int(labels.shape[-1]/seq_len), axis=-1)
            # take care of the mask

        if exists(mask):
            mask_segments = tf.split(mask,int(mask.shape[-1]/seq_len), axis=-1)
            # keep replay buffer

        replay_buffer = [memories]

        # replay buffer for xl memories

        xl_segments = [xl_memories]
  # decide context of forward depending on whether doing memory-replay-backprop

        forward_context = nullcontext() if not memory_replay_backprop else EmptyContextManager()

        # forward and get all outputs (can be either loss or logits)

        logits = []
        losses = []

        for step, (segment, mask_segment, label_segment, loss_weight) in enumerate(zip_longest(segments, mask_segments, label_segments, segment_length_frac)):

            with forward_context:
                output, memories, xl_memories = self.transformer.call(segment, memories, mask = mask_segment, labels = label_segment)


                if exists(truncate_at_step) and divisible_by(step + 1, truncate_at_step):
                    memories = memories.detach()

            replay_buffer.append(memories)

            xl_segments.append(xl_memories)

            if return_loss:
                losses.append(output * loss_weight)
            else:
                logits.append(output)

        if memory_replay_backprop:
            memories_grad = tf.Variable(tf.zeros_like(replay_buffer[-1]))
            self.memories_grad=memories_grad
            reversed_inputs = zip_longest(*map(reversed, [
                range(num_segments),
                segments,
                replay_buffer[:-1],
                xl_segments[:-1],
                mask_segments,
                label_segments,
                segment_length_frac,
            ]))

            total_loss=0
            for step, segment, segment_memories, segment_xl_memories, mask_segment, label_segment, loss_weight in reversed_inputs:
                is_first = step == 0

                if exists(segment_memories):

                    segment_memories=tf.Variable(segment_memories)
                opt1=tf.keras.optimizers.Adam(learning_rate=0.01)
                #opt2=tf.keras.optimizers.Adam(learning_rate=0.001)

                loss, next_segment_memories, _ = self.transformer(segment, segment_memories, mask=mask_segment,xl_memories=segment_xl_memories,
                    labels=label_segment)
                weighted_loss = loss * loss_weight * mrbp_loss_weight

                    #weighted_loss.backward(retain_graph=True)

                    #next_segment_memories.backward(memories_grad)
                if exists(segment_memories):
                    sggrad=tp.gradient(weighted_loss,segment_memories)
                memograd=tp.gradient(next_segment_memories,memories_grad)

                total_loss += weighted_loss

                if memograd  is not None:
                    opt1.apply_gradients(zip(memograd,[memories_grad]))
                if is_first:
                    continue

                if exists(truncate_at_step) and divisible_by(step, truncate_at_step):
                    memories_grad.assign(tf.zeros_like(memories_grad))
                else:
                    memories_grad.assign(sggrad)
            return total_loss,
        if not return_loss:
            logits = tf.concat(logits, axis = -2)
            return logits, memories

        # otherwise return the sum of losses

        return sum(losses), memories

