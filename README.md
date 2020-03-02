# txt2numpy
A series of scripts to convert a set of txt files into a set of numpy arrays stored as h5py files

About
=====
If you have a set of txt files as for OpenWebText and PG19, the first thing you may want to do is to convert them into a set of numpy arrays tokenized in some way. Here, I will provide a set of scripts to perform this efficiently. It allows my laptop to process the entirety of OpenWebText within an hour. You can modify them accordingly to suit your needs. There are two types of resulting numpy arrays: (1) the one-dimensional array that tokenizes all the tokens, and it would look like `<SEP>(DOC1)<SEP>(DOC2)<SEP>...`, (2) the array that records the location of separator. This format is suitable for long-range language modeling such as Reformer. 

If you want a 2D numpy array (e.g. for shorter-range language model as BERT), each of whose row is a document, padded and curtailed accordingly for varying size, then you can segment this first array into each document using the second array. While this process is simple, I didn't provide a script for this, as the main target of this repo is for long-range language modeling. 

I also provided a script that converts the first array in a way suitable for adaptive input/softmax. For example, one can attempt GPT-2 BPE with adaptive input/softmax, which I observed to be fast and perform well.

Requirements
-----------
* [huggingface/tokenizers](https://github.com/huggingface/tokenizers). 
* h5py 
  
Concatenating txt files
-----------
For this, all you need is to bash a command as follows (this is an example for GP19):

  `echo ./train/*.txt | xargs awk 'FNR==1{print "<|endoftext|>"}1' > ./train.txt`
This will produce a txt file that would look as `<|endoftext|>(DOC1)<|endoftext|>(DOC2)<|endoftext|>...`. Contrary to intuition, `<|endoftext|>` notes the beginning of a document rather than the end (this was to follow the convention of GPT2 BPE), so you may want to replace it with something as `<|startoftext|>` if you prefer, though this doesn't matter. If you do so, please replace likewise in the remaining scripts.  

Tokenization
-----------
For this, it suffices to call `python3 tokenize.py`. It produces a h5py file containing the aforementioned two types of numpy arrays. If you want to use the default GPT2 BPE, please store `vocab.json` and `gpt2-merges.txt` in the same directory. If you want to try a customized vocabulary, please refer to [huggingface/tokenizers](https://github.com/huggingface/tokenizers) for their instruction. You can then modify my `tokenize.py` accordingly. In particular, you need to modify the dtype accordingly if your custom vocabulary size exceeds the limit of the default dtype. If by any chance it outputs MemoryError, you can decrease the default value for `MAX_TOKEN_LENGTH` and `FREQ`. 

For adaptive input/softmax
==========
For this, it suffices to call `python3 adaptive.py`. It retokenizes the array of the first type in a way compatible with adaptive input/softmax, while the array containing the document location information is kept intact. It also outputs how this mapping was performed, which can be used for converting the tokens back into the original vocabulary such as GPT2 BPE. 

Caveats
----------
Though I use PyTorch, I do not use its DataLoader class. So, I'm not sure how h5py file works with DataLoader. If by any chance it does not work well, you may want to convert it into other format. 
