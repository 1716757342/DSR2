from dnc import DNC
import torch
import time

rnn = DNC(
  input_size=64,
  hidden_size=128,
  rnn_type='lstm',
  num_layers=1,
  nr_cells=5,
  cell_size=10,
  read_heads=2,
  batch_first=True,
  gpu_id=-1
)

(controller_hidden, memory, read_vectors) = (None, None, None)

start_time = time.time()
output, (controller_hidden, memory, read_vectors) = \
  rnn(torch.randn(2000, 64), (controller_hidden, memory, read_vectors), reset_experience=True)
print(output.shape)
print(time.time()-start_time)