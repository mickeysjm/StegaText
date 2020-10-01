import torch
import numpy as np
import bitarray

from transformers import GPT2LMHeadModel, GPT2Tokenizer

def decode(self, token_ids, **kwargs):
    filtered_tokens = self.convert_ids_to_tokens(token_ids)
    text = self.convert_tokens_to_string(filtered_tokens)
    return text
GPT2Tokenizer.decode = decode

def _convert_token_to_id(self, token):
    return self.encoder.get(token, 0)
GPT2Tokenizer._convert_token_to_id = _convert_token_to_id


def limit_past(past):
    past = list(past)
    for i in range(len(past)):
        past[i] = past[i][:, :, :, -1022:]
    return past

def kl(q, logq, logp):
    res = q*(logq-logp)/0.69315
    res[q==0] = 0
    return res.sum().item() # in bits

def entropy(q, logq):
    res = q*logq/0.69315
    res[q==0] = 0
    return -res.sum().item() # in bits

# e.g. [0, 1, 1, 1] looks like 1110=14
def bits2int(bits):
    res = 0
    for i, bit in enumerate(bits):
        res += bit*(2**i)
    return res

# LSB -> MSB
def int2bits(inp, num_bits):
    if num_bits == 0:
        return []
    strlist = ('{0:0%db}'%num_bits).format(inp)
    return [int(strval) for strval in reversed(strlist)]

def is_sent_finish(token_idx, enc):
    token = enc.decoder[token_idx]
    return '.' in token or '!' in token or '?' in token

def num_same_from_beg(bits1, bits2):
    assert len(bits1) == len(bits2)
    for i in range(len(bits1)):
        if bits1[i] != bits2[i]:
            break

    return i

def encode_context(raw_text, enc):
    context_tokens = [enc.encoder['<|endoftext|>']] + enc.encode(raw_text)
    return context_tokens

# Use gpt2-medium for 345M param model
# Use gpt2-large for 774M param model
def get_model(seed=1234, model_name='gpt2', device_id="0"):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    enc = GPT2Tokenizer.from_pretrained(model_name)
    enc.unk_token = None
    enc.bos_token = None
    enc.eos_token = None
    
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    #model.double()

    return enc, model, device

enc32_itoc = ['\0', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '.', ',', "'", '!', ' ']
enc32_ctoi = {k: v for v, k in enumerate(enc32_itoc)}
def enc32(text):
    bits = []
    for c in text:
        bits.extend(int2bits(enc32_ctoi[c], 5))
    return bits

def dec32(bits):
    text = ''
    for i in range(0, len(bits), 5):
        c = enc32_itoc[bits2int(bits[i:i+5])]
        if c == '\0':
            break
        text += c
    return text

# message should be bit string
# encoded should be text string
def expansion_ratio(message, encoded):
    message_bits = len(message)
    encoded_ba = bitarray.bitarray()
    encoded_ba.frombytes(encoded.encode('utf-8'))
    encoded_bits = len(encoded_ba.tolist())
    return encoded_bits/message_bits

def get_output_file_name(args):
    dataset_path = args["dataset_path"]
    encryption_method = args["encrypt"]
    steganography_method = args["encode"]
    if steganography_method in ["bins", "huffman"]:
        block_size = args["block_size"]
        output_name = f"{dataset_path}/results_{encryption_method}_{steganography_method}_block_{block_size}.json"
    elif steganography_method == "patient-huffman":
        block_size = args["block_size"]
        epsilon = args["epsilon"]
        output_name = f"{dataset_path}/results_{encryption_method}_{steganography_method}_block_{block_size}_epsilon_{epsilon}.json"
    elif steganography_method == "arithmetic":
        precision = args["precision"]
        temp = args["temp"]
        topK = args["topK"]
        output_name = f"{dataset_path}/results_{encryption_method}_{steganography_method}_precision_{precision}_temp_{temp}_topK_{topK}.json"
    elif steganography_method == "saac":
        precision = args["precision"]
        temp = args["temp"]
        delta = args["delta"]
        output_name = f"{dataset_path}/results_{encryption_method}_{steganography_method}_precision_{precision}_temp_{temp}_delta_{delta}.json"
    else:
        output_name = ""
    return output_name
