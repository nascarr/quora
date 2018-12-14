import torch.optim as optim
from functools import partial

Adam = optim.Adam

AdamW = partial(optim.Adam, betas=(0.9,0.99))
