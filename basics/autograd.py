import torch

a = torch.randn(3, requires_grad=True)
print("a:", a)

b = a+2
print("b:", b)
c = a*2
print("c:", c)
d = a.mean()
print("d:", d)

d.backward() # (dd/da)
print("gradient: ", a.grad)

# vector jacobian product(multiplying tensor with a vector)
v = torch.tensor([0.1, 1, 0.001], dtype=torch.float32)
print("vector v", v)
c.backward(v)
#print(c.backward(v))
print(a.grad)

# prevent pytorch from tracking gradient
# a.requires_grad_(False)
# print("a:", a)
# a = a.detach()
# print("a:", a)
with torch.no_grad():
  b = a*1
  print(b)

# same code outside the block
b = a*1
print(b)

weights = torch.ones(4, requires_grad=True)

for epoch in range(4):
  model_output = (weights * 3).sum()
  print('model output:', model_output)
  model_output.backward()
  print('weights:', weights.grad)

  # empty the gradients
  weights.grad.zero_()