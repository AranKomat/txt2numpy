import numpy as np
import h5py
from time import time
vocab_size = 50257
with h5py.File('./tokenized', "r") as f:
	x = f['tokens'][:10000000]
	y = np.bincount(x)
	if len(y) != vocab_size:
		NotImplementedError
	map = (-y).argsort().argsort()
	print(x, np.take(map,x))
	with h5py.File('./map', "w") as g:
		_ = g.create_dataset('map', shape=(vocab_size,), dtype='i4')
	with h5py.File('./map', "a") as g:
		tmp = g['map']
		tmp = map

t = time()
count = 0
n = 100000000
with h5py.File('./tokenized', "r") as f:
	tokens_in = f['tokens']
	docs_in = f['docs']	
	tokens_len = tokens_in.len()
	docs_len = docs_in.len()	
	with h5py.File('./tokenized2', "w") as g:
		_ = g.create_dataset('tokens', shape=(tokens_len,), chunks=(1000000,), dtype='u2')
		_ = g.create_dataset('docs', shape=(docs_len,), chunks=(10000,), dtype='i4')
	with h5py.File('./tokenized2', "a") as g:
		tokens_out = g['tokens']
		docs_out = g['docs']
		docs_out = docs_in	
		for idx in range(tokens_len // n):
			tokens_out[idx*n:(idx+1)*n] = np.take(map, tokens_in[idx*n:(idx+1)*n])
			print(count, time()-t)
			count += 1
		if idx*n < tokens_len:
			tokens_out[(idx+1)*n:] = np.take(map, tokens_in[(idx+1)*n:])
			
