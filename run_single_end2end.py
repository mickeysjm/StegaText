import numpy as np
import bitarray
import sys
import re
import math
import argparse

from utils import get_model, encode_context
from block_baseline import get_bins, encode_block, decode_block
from huffman_baseline import encode_huffman, decode_huffman
from patient_huffman_baseline import encode_patient_huffman, decode_patient_huffman
from arithmetic_baseline import encode_arithmetic, decode_arithmetic
from saac import encode_saac, decode_saac


def main(args):
    # get model hyperparameters
    args = vars(args)
    lm_model = args['lm']
    device = args['device']
    encryption_method = args["encrypt"]
    steganography_method = args["encode"]
    precision = args["precision"]
    temp = args["temp"]
    topk = args["topK"]
    block_size = args["block_size"]
    nucleus = args["nucleus"]    
    delta = args["delta"]
    if delta:
        nucleus = 2**(-1.0*delta)

    # get plaintext
    if args["plaintext"] == "":
        plaintext = "molly ultra caps capped at 180mgs will have you flying for hrs clean come down 99 of the time ."
    else:
        plaintext = args["plaintext"]

    # get steganography encoding context
    if args["context"] == "":
        context = args["context"]
    else:
        context = "Washington received his initial military training and command with the Virginia Regiment during the French and Indian War. He was later elected to the Virginia House of Burgesses and was named a delegate to the Continental Congress, where he was appointed Commanding General of the nation's Continental Army. Washington led American forces, allied with France, in the defeat of the British at Yorktown. Once victory for the United States was in hand in 1783, Washington resigned his commission."

    # start steganography pipeline
    print("Loading large LM to GPU, please wait for a few seconds...")
    enc, model, device = get_model(model_name=lm_model, device_id=device)
    
    # Encryption: encrypt secret plaintext to message bits
    print(f"Plaintext: {plaintext}")
    print(f"Encryption method: {encryption_method}")
    if encryption_method == "utf8":
        ba = bitarray.bitarray()
        ba.frombytes(plaintext.encode('utf-8'))
        message = ba.tolist()
    elif encryption_method == "arithmetic":
        message_ctx = [enc.encoder['<|endoftext|>']]
        plaintext += '<eos>'
        message = decode_arithmetic(model, enc, plaintext, message_ctx, device=device, precision=40, topk=60000)
    print(f"Encrypted message bits: {message}")

    # Steganography Encoding: encode message bits to covertext
    print(f"Steganography encoding method: {steganography_method}")
    context_tokens = encode_context(context, enc)
    if steganography_method == 'bins':
        bin2words, words2bin = get_bins(len(enc.encoder), block_size)
        out, nll, kl, words_per_bit = encode_block(model, enc, message, context_tokens, block_size, bin2words, words2bin, device=device)
    elif steganography_method == 'huffman':
        out, nll, kl, words_per_bit = encode_huffman(model, enc, message, context_tokens, block_size, device=device)
    elif steganography_method == 'patient-huffman':
        out, nll, kl, words_per_bit = encode_patient_huffman(model, enc, message, context_tokens, block_size, device=device, delta=1.0)
    elif steganography_method == 'arithmetic':
        out, nll, kl, words_per_bit, Hq, kl_list = encode_arithmetic(model, enc, message, context_tokens, device=device, temp=temp, precision=precision, topk=topk)
    elif steganography_method == 'saac':
        out, nll, kl, words_per_bit, Hq, topk_list, case_studies = encode_saac(model, enc, message, context_tokens, device=device, temp=temp, precision=precision, topk=topk, nucleus=nucleus)
    covertext = enc.decode(out)
    print(f"Encoded covertext: {covertext}")
    print(f"kl: {kl}, bits/words: {1.0/words_per_bit}")
    
    # Steganography Decoding: decode covertext to message bits
    if steganography_method == 'bins':
        message_rec = decode_block(model, enc, covertext, context_tokens, block_size, bin2words, words2bin)
    elif steganography_method == 'huffman':
        message_rec = decode_huffman(model, enc, covertext, context_tokens, block_size)
    elif steganography_method == 'patient-huffman':
        print("Not implemented yet")
        exit(-1)
    elif steganography_method == 'arithmetic':
        message_rec = decode_arithmetic(model, enc, covertext, context_tokens, device=device, temp=temp, precision=precision, topk=topk)
    elif steganography_method == 'saac':
        message_rec = decode_saac(model, enc, covertext, context_tokens, device=device, temp=temp, precision=precision, topk=topk, nucleus=nucleus)
    print(f"Decoded message bits: {message_rec}")

    # Decryption: map message bits back to original text
    if encryption_method == "utf8":
        message_rec = [bool(item) for item in message_rec]
        ba = bitarray.bitarray(message_rec)
        reconst = ba.tobytes().decode('utf-8', 'ignore')
    elif encryption_method == "arithmetic":
        reconst = encode_arithmetic(model, enc, message_rec, message_ctx, device=device, precision=40, topk=60000)
        reconst = enc.decode(reconst[0])
    print("Recovered plaintext:", reconst)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-plaintext", type=str, default="", help="your secret plaintext, use a double-quotes if necessary")
    parser.add_argument("-context", type=str, default="", help="context used for steganography, use a double-quotes if necessary")
    parser.add_argument("-encrypt", type=str, default="arithmetic", choices=["arithmetic", "utf8"])
    parser.add_argument("-encode", type=str, default="bins", choices=["bins", "huffman", "patient-huffman", "arithmetic", "saac"])
    parser.add_argument("-lm", type=str, default="gpt2")
    parser.add_argument("-device", type=str, default="0", help="your gpu device id")
    parser.add_argument("-block_size", type=int, default=4, help="block_size for bin/huffman encoding method")
    parser.add_argument("-precision", type=int, default=26, help="precision for arithmetic encoding method")
    parser.add_argument("-temp", type=float, default=1.0, help="temperature for arithemtic/huffman encoding method")
    parser.add_argument("-topK", type=int, default=50, help="topK for arithemtic encoding method")
    parser.add_argument("-nucleus", type=float, default=0.95, help="neclues for adaptive arithemtic encoding method")
    parser.add_argument("-delta", type=float, default=0.01, help="delta for adaptive arithemtic encoding method")
    args = parser.parse_args()
    main(args)
