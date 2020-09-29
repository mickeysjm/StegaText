# StegaText

This repo contains the implementations of several linguistic steganography methods in paper "Near-imperceptible Neural Linguistic Steganography via Self-Adjusting Arithmetic Coding" published in EMNLP 2020.

## Dependency

You need to install all dependent librarys in `requirements.txt` file. Besides, you need to download the `gpt2-medium` model (345M parameter) from [transformers library](https://huggingface.co/transformers/pretrained_models.html)

## Included Implementations

1. `block_baseline.py`: implementations of baseline method `Bin-LM` in the paper.
2. `huffman_baseline.py`: implementations of baseline method `RNN-Stega` in the paper.
3. `arithmetic_baseline.py`: implementations of baseline method `Arithmetic` in the paper.
4. `patient_huffman_baselin.py`: implementations of baseline method `Patient-Huffman` in the paper.
5. `saac.py`: implementations of our proposed method `SAAC` in the paper.
6. `run_single_end2end.py`: a script to run though the entire steganography pipeline (i.e., encryption -> encoding -> decoding -> decryption) on `one plaintext`.
7. `run_batch_encode.py`: a script to run the encryption+encoding steps on `a batch of plaintexts`.
8. `run_all.sh`: a script to run all above methods


