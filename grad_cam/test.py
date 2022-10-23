import numpy as np
import torch

a = torch.rand(64,64)
b = torch.reshape(a,(1,4096))
c = torch.squeeze(b,dim=0)

d = a[0:32,0:32]
e = a[32:64,32:64]

print(d.size())
print(e.size())

# print(a)
print(b.size())
print(c.size())