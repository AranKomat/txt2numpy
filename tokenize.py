from tokenizers import ByteLevelBPETokenizer
tokenizer = ByteLevelBPETokenizer('vocab.json', 'gpt2-merges.txt')
import numpy as np
from time import time
import h5py
KEYS = ['tokens', 'docs']
MAX_TOKEN_LENGTH = 400000000
FREQ = 10000
SLASH_N = 198 # the token for '\n' in the case of gpt2 bpe
SEP = 50256  # the token for '<|endoftext|>' in the case of gpt2 bpe

class Counter(object):
	def __init__(self):
		self.init()
		self.idx_saved = {'tokens': 0, 'docs': 0}
		
	def init(self):
		self.keys = KEYS
		self.max_token_length = MAX_TOKEN_LENGTH
		self.obj = {'tokens': np.zeros(self.max_token_length, dtype=np.uint16), 'docs': np.zeros(self.max_token_length//100, dtype=np.int32)}
		self.idx = {'tokens': 0, 'docs': 0}
		self.tmp = []
		self.save_cond = False
	
	def create_dataset(self):
		with h5py.File('./tokenized', "w") as f:
			tokens = f.create_dataset('tokens', shape=(0,), maxshape=(None,), chunks=(1000000,), dtype='u2')		
			docs = f.create_dataset('docs', shape=(0,), maxshape=(None,), chunks=(1000000,), dtype='i4')

	def repeat(func):
		def wrapper(*args):
			outputs = {el: None for el in KEYS}
			for k in KEYS:
				args_ = [arg[k] if idx>0 else arg for idx, arg in enumerate(args)]
				out = func(*args_)
				if out is not None:
					outputs[k] = out
			if out is not None:
				return outputs
		return wrapper
			
	@repeat
	def add(self, obj, idx, seq):
		length = len(seq)
		obj[idx:idx+length] = np.array(seq, dtype=np.int32)
		return idx + length
					
			
	def encoding(self):
		tmp = ''.join(self.tmp).split('<|endoftext|>')
		encoded = tokenizer.encode_batch(tmp)
		self.tmp = []
		seq = []
		for elm in encoded: #eliminate \n
			tmp = elm.ids
			if len(tmp) > 0 and tmp[0] == SLASH_N:
				seq += [tmp[1:]]
			else:
				seq += [tmp]
		seqs = self.clean(seq)
		seqs = {self.keys[i]: seqs[i] for i in range(len(self.keys))} 
		self.idx = self.add(self.obj, self.idx, seqs)
		if self.max_token_length*0.9 < self.idx['tokens']:
			self.save_cond = True
	
	def clean(self, seq):
		new_seq = []
		docs = []
		count = self.idx['tokens']
		seqlen = len(seq)
		for idx, elm in enumerate(seq):
			elmlen = len(elm)
			new_seq += elm
			count += elmlen
			if (idx != seqlen-1) or (elmlen == 0): 
				new_seq += [SEP]
				docs += [count+self.idx_saved['tokens']] #local_idx for doc + global token_idx
				count += 1
		return new_seq, docs
		
	@repeat
	def curtail(self, obj, idx):
		return obj[:idx]
	
	def save(self):
		self.obj = self.curtail(self.obj, self.idx)

		with h5py.File('./tokenized', "a") as f:
			tokens = f['tokens']
			docs = f['docs']
			token_idx = self.idx['tokens']
			doc_idx = self.idx['docs'] 
			token_idx_saved = self.idx_saved['tokens']
			doc_idx_saved = self.idx_saved['docs']
			self.idx_saved['tokens'] += token_idx
			self.idx_saved['docs'] += doc_idx
			tokens.resize((self.idx_saved['tokens'],))
			docs.resize((self.idx_saved['docs'],))
			tokens[token_idx_saved:] = self.obj['tokens']
			docs[doc_idx_saved:] = self.obj['docs']
		self.init()
	
t = time()
c = Counter()
count = 0
with open('./train.txt') as f:
	c.create_dataset()
	count = 0
	for line in f: 
		if c.save_cond:
			c.save()
		count += 1
		c.tmp += [line]
		if (count+1) % FREQ == 0:
			c.encoding()
			print(count, time()-t, c.idx)
	if c.tmp:
		c.encoding()
	
	c.save()
		
	
