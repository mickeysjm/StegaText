# run batch steganography encoding (just encryption plus encoding steps)
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt cached -encode bins -block_size 4 
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt cached -encode huffman -block_size 7
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt cached -encode arithmetic -topK 300
python run_batch_encode.py -dataset drug -dataset_path ./datasets/drug -encrypt cached -encode saac -delta 0.01
# run single steganography whole pipeline (encryption -- encoding -- decoding -- decryption)
python run_single_end2end.py -encode bins
python run_single_end2end.py -encode huffman
python run_single_end2end.py -encode arithmetic
python run_single_end2end.py -encode saac