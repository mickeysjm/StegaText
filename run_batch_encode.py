import numpy as np
import bitarray
import sys
import re
import math
import argparse

from utils import get_model, encode_context, get_output_file_name
from block_baseline import get_bins, encode_block, decode_block
from huffman_baseline import encode_huffman, decode_huffman
from patient_huffman_baseline import encode_patient_huffman, decode_patient_huffman
from arithmetic_baseline import encode_arithmetic, decode_arithmetic
from saac import encode_saac, decode_saac
from tqdm import tqdm
import json
import time


def plaintext2bits(plaintext, context, model, enc, model_type, encryption_method, device, **args):
    """Encryption step, convert plaintext to message bits.
    
    Args:
        plaintext (str): secret message
        context (str): context message for arithmetic based encryption, typically ""
        model (LM model or None): a LM model for arithmetic based encrytion or None for unicode encryption
        enc (LM encoder or None): an pretrained encoder for rithmetic based encrytion or None for unicode encryption
        model_type (str): name of LM model or "" for unicode encryption
        encryption_method (str): name of encryption method
    
    Returns:
        message: a list of bit
        info: a dictionary of encryption information
    """
    assert encryption_method in {"utf8", "arithmetic"}, f"Unsupported encryption method: {encryption_method}"
    assert model_type in {"utf8", "gpt2"}, f"Unsupported model type: {model_type}"

    n_words = len(plaintext.split(" "))
    if encryption_method == "utf8":
        n_subwords = n_words
        ba = bitarray.bitarray()
        ba.frombytes(plaintext.encode('utf-8'))
        message = ba.tolist()
    elif encryption_method == "arithmetic":
        n_subwords = len(enc.tokenize(plaintext))
        if model_type == "gpt2":
            context_tokens = [enc.encoder['<|endoftext|>']] + enc.encode(context)
            plaintext += '<eos>'
            message = decode_arithmetic(model, enc, plaintext, context_tokens, device=device, precision=40, topk=60000)
    n_bits = len(message)
    info = {"n_words": n_words, "n_subwords": n_subwords, "n_bits": n_bits}
    return message, info


def bits2covertext(message, context, model, enc, model_type, steganography_method, device, **args):
    """Steganography step, convert message bits to cover text.
    
    Args:
        message ([int]): a list of message bits
        context (str): a context string
        model (LM model): a LM model 
        enc (LM encoder): a LM encoder
        model_type (str): name of LM model
        steganography_method (str): name of steganography method

    Returns:
        covertext: a str of cover text
        info: a dictionary of steganography information
    """
    assert steganography_method in {"bins", "huffman", "patient-huffman", "arithmetic", "saac"}, f"Unsupported steganography method: {steganography_method}"
    assert model_type in {"gpt2"}, f"Unsupported model type: {model_type}"

    # for huffman and bins coding
    block_size = args.get("block_size", 4)

    # for bins coding
    bin2words = args.get("bin2words", {})
    words2bin = args.get("words2bin", {})
    if steganography_method == "bins":
        assert bin2words != {} and words2bin != {}, "For steganography method bin, must specify bin2words and words2bin"

    # for arithmetic coding
    precision = args.get("precision", 26)
    temp = args.get("temp", 1.0)

    # for adaptive arithmetic coding
    nucleus = args.get("nucleus", 0.99)

    # all other shared parameters
    topk = args.get("topK", 300)
    finish_sent = args.get("finish_sent", False)  # whether or not to force finish sent. If so, stats displayed will be for non-finished sentence

    # encode context
    if model_type == "gpt2":
        context_tokens = [enc.encoder['<|endoftext|>']] + enc.encode(context)

    Hq = 0
    n_bits = len(message)
    if steganography_method == 'bins':
        out, nll, kl, words_per_bit = encode_block(model, enc, message, context_tokens, block_size, bin2words, words2bin, device=device, finish_sent=finish_sent)
        info = {"n_bits": n_bits, "ppl": math.exp(nll), "kl": kl, "words_per_bit": words_per_bit, "bits_per_word": 1.0/words_per_bit, "Hq": Hq/0.69315}
    elif steganography_method == 'huffman':
        out, nll, kl, words_per_bit = encode_huffman(model, enc, message, context_tokens, block_size, device=device, finish_sent=finish_sent)
        info = {"n_bits": n_bits, "ppl": math.exp(nll), "kl": kl, "words_per_bit": words_per_bit, "bits_per_word": 1.0/words_per_bit, "Hq": Hq/0.69315}
    elif steganography_method == 'patient-huffman':
        out, nll, kl, words_per_bit = encode_patient_huffman(model, enc, message, context_tokens, block_size, delta=1.0, device=device, finish_sent=finish_sent)
        info = {"n_bits": n_bits, "ppl": math.exp(nll), "kl": kl, "words_per_bit": words_per_bit, "bits_per_word": 1.0/words_per_bit, "Hq": Hq/0.69315}
    elif steganography_method == 'arithmetic':
        out, nll, kl, words_per_bit, Hq, kl_list = encode_arithmetic(model, enc, message, context_tokens, device=device, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk)
        info = {"n_bits": n_bits, "ppl": math.exp(nll), "kl": kl, "words_per_bit": words_per_bit, "bits_per_word": 1.0/words_per_bit, "Hq": Hq/0.69315, "kl_list": kl_list}
    elif steganography_method == 'saac':
        out, nll, kl, words_per_bit, Hq, topk_list, case_studies = encode_saac(model, enc, message, context_tokens, device=device, temp=temp, finish_sent=finish_sent, precision=precision, topk=topk, nucleus=nucleus)
        info = {"n_bits": n_bits, "ppl": math.exp(nll), "kl": kl, "words_per_bit": words_per_bit, "bits_per_word": 1.0/words_per_bit, "Hq": Hq/0.69315, "topk_list": topk_list, "case_studies": case_studies}
    
    covertext = enc.decode(out)
    return covertext, info


def main(args):
    # process hyperparameters
    args = vars(args)
    dataset = args['dataset']
    dataset_path = args['dataset_path']
    lm_model = args['lm']
    device = args['device']
    encryption_method = args["encrypt"]
    use_cached_encryption_results = (encryption_method == "cached")
    steganography_method = args["encode"]
    precision = args["precision"]
    temp = args["temp"]
    topK = args["topK"]
    block_size = args["block_size"]
    nucleus = args["nucleus"]    
    delta = args["delta"]
    if delta:
        nucleus = 2**(-1.0*delta)
    print("Loading large LM to GPU, please wait for a few seconds...")
    enc, model, device = get_model(model_name=lm_model, device_id=device)

    # load plaintext dataset
    if dataset != "random":
        with open(f"{dataset_path}/plaintext.txt", "r") as fin:
            plaintexts = [line.strip() for line in fin.readlines() if line.strip() != ""]
        print(f"Encoding {len(plaintexts)} plaintexts")
    bin2words, words2bin = get_bins(len(enc.encoder), block_size)
    args["bin2words"] = bin2words
    args["words2bin"] = words2bin

    # encryption
    print(f"Encryption Algorithm: {encryption_method}")
    if use_cached_encryption_results:
        print("Load existing encrypted messages")
        encryption_infos = []
        messages = []
        with open(f"{dataset_path}/message_bits.txt", "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    messages.append(eval(line))
    else:
        encryption_infos = []
        encryption_context = ""
        messages = []
        for plaintext in tqdm(plaintexts, desc="encrypting"):
            message, info = plaintext2bits(plaintext, encryption_context, model, enc, lm_model, encryption_method, device)
            messages.append(message)
            encryption_infos.append(info)
        with open(f"{dataset_path}/message_bits.txt", "w") as fout:
            for message in messages:
                fout.write(str(message))
                fout.write("\n")

    # steganography encoding 
    encoding_infos = []
    encoding_context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."
    covertexts = [] 
    print(f"Steganography Encoding Algorithm: {steganography_method}")
    start = time.time()
    for message in tqdm(messages[:100], desc="encoding"):
        covertext, info = bits2covertext(message, encoding_context, model, enc, lm_model, steganography_method, device, bin2words=bin2words, words2bin=words2bin,
            precision=precision, temp=temp, topK=topK, block_size=block_size, nucleus=nucleus)
        covertexts.append(covertext)
        encoding_infos.append(info)
    end = time.time()
    efficiency = (end-start) / 100
    print(f"Use {efficiency} per example")

    results = {
        "encrpytion_infos": encryption_infos,
        "encoding_infos": encoding_infos,
        "covertexts": covertexts
    }
    output_name = get_output_file_name(args)
    with open(output_name, "w") as fout:
        json.dump(results, fout, indent=4, sort_keys=True, separators=(',', ': '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", type=str, default="drug", choices=["drug", "cnn_dm", "covid_19", "random"])
    parser.add_argument("-dataset_path", type=str, default="../drug/")
    parser.add_argument("-encrypt", type=str, default="cached", choices=["utf8", "arithmetic", "cached"])
    parser.add_argument("-encode", type=str, default="saac", choices=["bins", "huffman", "patient-huffman", "arithmetic", "saac"])
    parser.add_argument("-lm", type=str, default="gpt2")
    parser.add_argument("-device", type=str, default="0", help="your gpu device id")
    parser.add_argument("-block_size", type=int, default=4, help="block_size for bin/huffman encoding method")
    parser.add_argument("-precision", type=int, default=26, help="precision for arithmetic encoding method")
    parser.add_argument("-temp", type=float, default=1.0, help="temperature for arithmetic/huffman encoding method")
    parser.add_argument("-topK", type=int, default=50, help="topK for arithmetic encoding method")
    parser.add_argument("-nucleus", type=float, default=0.99, help="neclues for adaptive arithmetic encoding method")
    parser.add_argument("-delta", type=float, default=0.01, help="delta for adaptive arithmetic encoding method")
    args = parser.parse_args()
    main(args)
