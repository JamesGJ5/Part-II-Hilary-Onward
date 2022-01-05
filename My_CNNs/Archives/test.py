import torch
arr1 = torch.Tensor([1.5, 2.5, 3.5])
arr2 = torch.Tensor([2, 3, 4])
arr3 = arr1 * arr2
print(arr3)
arr3.unsqueeze_(-1)
print(arr3)