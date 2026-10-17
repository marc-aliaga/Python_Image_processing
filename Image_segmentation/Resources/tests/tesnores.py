import torch
import numpy as np 

t1 = torch.tensor(4.)
t2 = torch.tensor([1, 2, 3, 4])
t3 = torch.tensor([[5., 6],
                   [7,  8],
                   [9, 10]])
print(t3)
reshape = torch.reshape(t3, (2, 3))
print(reshape)

x = torch.tensor([1, 2, 3, 4])
x_stacked = torch.stack([x, x, x, x], dim=0)
print(x_stacked)

####permutations

x_original = torch.rand(size=(2, 2, 3))
x_permuted = x_original.permute(2, 0, 1)
print(f" original: {x_original}")
print(f" permuted: {x_permuted}")

###derivative calc

x = torch.tensor(2., requires_grad= True)
y = x ** 4
print(y)

y.backward()
print("dy/dx:", x.grad)
