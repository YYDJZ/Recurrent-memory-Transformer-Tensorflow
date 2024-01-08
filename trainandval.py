

import gzip
import tqdm
import numpy as np
import tensorflow as tf

from  utils.recurrent_memory_transformer import RecurrentMemoryTransformer, RecurrentMemoryTransformerWrapper

NUM_EPOCHES = 1000 #Training epoches
BATCH_SIZE = 4
GRADIENT_ACCUMULATE_EVERY = 2
VALIDATE_EVERY = 20 #How many epoch validate
PRIME_LENGTH = 128
GENERATE_EVERY = 100
GENERATE_LENGTH = 256
SEQ_LEN = 256 # The length of input sequence

import random




with gzip.open("./data/enwik8.gz") as file:
    print(file)
    data = np.frombuffer(file.read(int(95e6)), dtype=np.uint8).copy()
    np_train, np_valid = np.split(data, [int(90e6)])


#Tensorflow does not have function of dataloader, so we use equvilant way to generate input data.

def generatesample(dataset, seq_len):
    rand_start = random.randint(0, dataset.size - seq_len - 1, )
    seq = dataset[rand_start:rand_start + seq_len + 1]
    return seq


def generatesamples(dataset, batch, seq_len):
    lst = []
    for i in range(batch):
        rand_start = random.randint(0, dataset.size - seq_len, )
        seq = dataset[rand_start:rand_start + seq_len + 1]
        lst.append(seq)
    return tf.stack(lst)


def decode_token(token):
    return str(chr(max(32, token)))


def decode_tokens(tokens):
    return "".join(list(map(decode_token, tokens)))


model0 = RecurrentMemoryTransformer(
    num_tokens=256,
    dim=512,
    depth=3,
    dim_head=64,
    heads=8,
    seq_len=256,
    use_flash_attn=True,
    num_memory_tokens=64,
    use_xl_memories=True,
    xl_mem_len=256
)

model = RecurrentMemoryTransformerWrapper(model0)

trainloss = []

validloss = []

for i in tqdm.tqdm(range(NUM_EPOCHES), mininterval=10.0, desc="training"):
    # model.train()
    model.trainable = True
    model0.trainable = True
    total_loss = tf.constant([0]).numpy()[0]
    for _ in range(GRADIENT_ACCUMULATE_EVERY):
        model = RecurrentMemoryTransformerWrapper(model0)
        opt1 = tf.keras.optimizers.Adam(learning_rate=0.0001)
        sp = generatesamples(np_train, BATCH_SIZE, SEQ_LEN)
        with tf.GradientTape(persistent=True) as tape:
            loss = model(
                sp,
                memory_replay_backprop=True,
                mrbp_loss_weight=1. / GRADIENT_ACCUMULATE_EVERY,
                tp=tape
            )
            print(loss)
        gd = tape.gradient(loss, model0.trainable_variables)
        opt1.apply_gradients(zip(gd, model0.trainable_variables))

        total_loss += loss
    trainloss.append(total_loss)
    print(f"training loss: {total_loss}")


    """Since the original code place the backward in forward() and in tensorflow the backward must be execueted when optimizing, put
    the optimizer in the call()function of our model"""

    if (i+1)  % VALIDATE_EVERY == 0:
        # model.eval()
        model.trainable = False
        model0.trainable = False
        loss, _ = model(generatesamples(np_valid, BATCH_SIZE, SEQ_LEN), return_loss=True)
        print(f"validation loss: {loss}")
        validloss.append(loss)
        ls = sorted(trainloss[-10:])[1:-1]
        print(sum(ls) / len(ls))

    if i % GENERATE_EVERY == 0:
        #model.eval()
        model.trainable=False
        model0.trainable=False
        inp = generatesample(np_valid,SEQ_LEN)[:PRIME_LENGTH]
        prime = decode_tokens(inp)
        print(f"%s \n\n %s", (prime, "*" * 100))

        sample = model.generate(inp[None, :], length = GENERATE_LENGTH)
        output_str = decode_tokens(sample[0])
        print("output_str:")
        print(output_str, "\n")

#Save the relevant data.

with open('trainloss.txt', 'w') as f:
    for item in trainloss:
        f.write("%s\n" % item)

with open('valloss.txt', 'w') as f:
    for item in validloss:
        f.write("%s\n" % item)