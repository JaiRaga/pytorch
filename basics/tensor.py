import torch

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




