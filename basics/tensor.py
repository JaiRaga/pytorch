import torch
import numpy as np

# create a tensor
t = torch.empty(2, 2, 1, 3)

print("tensor", t, type(t))

# create a tensor with random value
random_tensor = torch.rand(1,2)
print("random tensor", random_tensor, type(random_tensor))

# create a tensor with zeros and ones
zeros = torch.zeros(2, 2)
ones = torch.ones(2, 2)
print("zeros:", zeros)
print("ones", ones, ones.dtype)
ones_int = torch.ones(2, 2, dtype=torch.int)
print("ones int type", ones_int, ones_int.size())

# alternate way to create tensor
alt_tensor = torch.tensor([2, 10, 5])
print("tensor:", alt_tensor, alt_tensor.dtype)

# tensor addition(same for multiplication, subtraction, division) --> torch.add, torch.sub, torch.mul, troch.div
x = torch.tensor([2,2])
z = torch.tensor([1,1])
y = torch.empty(2)
print(x, y, z, x.dtype, y.dtype, z.dtype)

int_add = x + z

print("integer addition", int_add, int_add.dtype)

mix_add = x + y

print("mixed addition", mix_add, mix_add.dtype)

add = torch.add(x, y)

print("addition", add, add.dtype)

# inplace addition
y.add_(x)
print("addition", y, y.dtype)

# division
division = torch.div(z, x)
div2 = z / x
print("Division:", division, div2)

# accessing elements
a = torch.rand(5, 3)
print("Accessing: Original", a)
print("Accessing column one:", a[:, 1]) # access the first column
print("Accessing first row:", a[1, :])
print("Accessing one element:", a[1, 1], a[1,1].item()) # .item() only works when there is one element in the tensor

# reshaping tensor
print("Accessing: Original", a)
b = a.view(15)
print("reshaped tensor", b, b.size()) # returns 1-d tensor
b = a.view(-1, 5) # -1 indicates pytorch will figure the right shape
print("reshap determined by pytorch", b, b.size())

# changing a tensor into numpy array
print("ones tensor", ones, type(ones))
ones_np = ones.numpy()
print("ones numpy", ones_np, type(ones_np))

# Note: if the tensor is on the cpu and not the gpu then
# both the objects share the same memory location, so if
# we change one we will also change the other

ones.add_(4)
print('tensor', ones)
print('np', ones_np)

# changing numpy array to tensor
ones_numpy = np.ones(5)
print("numpy ones:", ones_numpy)
ones_tensor = torch.from_numpy(ones_numpy)
print("tensor ones:", ones_tensor)
ones_numpy += 1
print("numpy ones:", ones_numpy)
print("tensor ones:", ones_tensor) # refer the above note

# to use gpu
if torch.cuda.is_available():
  device = torch.device('cuda')
  x = torch.ones(5, device=device)
  y = torch.ones(5)
  y = y.to(device) # moves y from cpu to gpu
  z = x + y
  z = z.to("cpu") # numpy array can only be stored in cpu






