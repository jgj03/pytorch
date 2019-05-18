# Need a GPU to get this to run!

import torch
cpu = torch.device('cpu')
gpu = torch.device('cuda')
cpu, gpu

# not specifying data in cpu or gpu
x = torch.tensor([1.5])
x, x.device

# on cpu
y = torch.tensor([2.5], device=cpu)
y, y.device

# on gpu
z = torch.tensor([3.5], device=gpu)
z, z.device

# cant mix all together
x + y + z # ERROR RuntimeError

# Can work on the gpu
a = x.to(gpu) + y.to(gpu) + z
a

# MOving back to CPU
b = a.to(cpu)
b, b.device
