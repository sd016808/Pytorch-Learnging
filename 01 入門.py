# 什麼是 Pytorch ？
# Numpy 的替代品，支援 GPU 運算
# 提供深度學習的模型框架，支援訓練、評估、預測

import torch
import numpy as np

np.array([1,2 ,3]) # ndarray
'''
array([1, 2, 3])
'''

torch.tensor([1,2,3]) # tensor
'''
tensor([1, 2, 3])
'''

# tensor 相關的函數 & 運算
torch.empty(5, 3) # 建立一個 5*3 的空矩陣
'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
'''
torch.randn(5, 3) # 建立一個 5*3 的隨機矩陣
'''
tensor([[ 0.4912,  0.3587,  0.5275],
        [ 0.0292,  0.6747,  1.7571],
        [-0.9482,  1.6986, -0.2130],
        [ 1.3560,  0.9019, -0.3862],
        [ 0.2087,  1.1592,  0.1219]])
'''
torch.zeros(5, 3, dtype=torch.long) # 建立一個 5*3 的全 0 矩陣 dtype 可以指定數據型態
'''
tensor([[0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]])
'''


# 基於已存在的 Tensor 操作
x = torch.tensor([1,2,3], dtype=torch.float)

# 建立一個 5*3 的全 1 矩陣
# Returns a Tensor of size size filled with 1. By default, the returned Tensor has the same torch.dtype and torch.device as this tensor.
x.new_ones(5, 3) 
'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]])
'''

torch.zeros_like(x, dtype=torch.long) # 建立一個和 x 相同大小的全 0 矩陣
'''
tensor([0, 0, 0])
'''

# 建立一個和 x 相同的隨機矩陣
# Returns a tensor with the same size as input that is filled with random numbers from a normal distribution with mean 0 and variance 1.
torch.randn_like(x, dtype=torch.float) 
'''tensor([-0.6243, -1.3635, -0.9666])'''

# 查看維度 
np.array([1, 2, 3]).shape 
'''
(3,)
'''
torch.tensor([1, 2, 3]).size() # shape() 和 size() 都可以查看維度
'''
torch.Size([3])
'''
torch.randn(5, 3).size()
'''
torch.Size([5, 3])
'''
torch.randn(5, 3).size()[1]
'''
3
'''

# 運算 加減乘除
x = torch.ones(5, 3)
y = torch.ones(5, 3)
print(x + y)
z = torch.ones(5, 3)
print(torch.add(x, y, out = z)) # z 原本的值會被覆蓋
print(z)
w = torch.add(x, y)
print(w)

y.add_(x) # in-place operation  一旦有'_' 就是 in-place operation

# 取得 tensor 的值
np.array([[1, 2, 3], [4, 5, 6]])[1]
'''
array([4, 5, 6])
'''
np.array([[1, 2, 3], [4, 5, 6]])[:, 2] # 不指定行 默認為全部
'''
array([3, 6])
'''

torch.tensor([[1, 2, 3], [4, 5, 6]])[1]
'''
tensor([4, 5, 6])
'''
torch.tensor([[1, 2, 3], [4, 5, 6]])[:, 2] # 不指定行 默認為全部
'''
tensor([3, 6])
'''

# 更改維度 reshape() 也可以更改維度
x.view(15)
'''
tensor([0.0552, 0.5683, 0.1646, 0.6544, 0.9681, 0.4681, 0.3195, 0.5535, 0.1881,
        0.9806, 0.8222, 0.3280, 0.9549, 0.6637, 0.9176])
        '''
x.view(-1)
'''
tensor([0.0552, 0.5683, 0.1646, 0.6544, 0.9681, 0.4681, 0.3195, 0.5535, 0.1881,
        0.9806, 0.8222, 0.3280, 0.9549, 0.6637, 0.9176])
        '''
x.view(15, 1)
'''
tensor([[0.0552],
        [0.5683],
        [0.1646],
        [0.6544],
        [0.9681],
        [0.4681],
        [0.3195],
        [0.5535],
        [0.1881],
        [0.9806],
        [0.8222],
        [0.3280],
        [0.9549],
        [0.6637],
        [0.9176]])
'''
x.view(3, 5)
'''
tensor([[0.0552, 0.5683, 0.1646, 0.6544, 0.9681],
        [0.4681, 0.3195, 0.5535, 0.1881, 0.9806],
        [0.8222, 0.3280, 0.9549, 0.6637, 0.9176]])
'''
x.view(-1, 5)  # -1 表示自動推斷剩餘的數量
'''
tensor([[0.0552, 0.5683, 0.1646, 0.6544, 0.9681],
        [0.4681, 0.3195, 0.5535, 0.1881, 0.9806],
        [0.8222, 0.3280, 0.9549, 0.6637, 0.9176]])
'''

# 取數值
x[3, 0]
'''
tensor(0.9806)
'''
x[3, 0].item() # 非常常用 反向傳遞 打印日誌 畫圖 等等
'''
0.9806369543075562
'''

# 拼接
x = torch.tensor([[1, 2, 3]])
torch.cat((x, x, x), 0) # 按照維度 0 拼接
'''
tensor([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])
'''

torch.cat((x, x, x), 1) # 按照維度 1 拼接
'''
tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3]])
'''

# Tensor 和 Numpy 的轉換
# Tensor -> Numpy
x = torch.ones(5, 3)
y = x.numpy() # 轉換後一樣共用記憶體
print(y)
'''
array([[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]])
 '''
x.add_(1)
print(y)
'''
array([[2. 2. 2.]
 [2. 2. 2.]
 [2. 2. 2.]
 [2. 2. 2.]
 [2. 2. 2.]])
'''

# Numpy -> Tensor
x = np.ones((5, 3))
y = torch.from_numpy(x) # 轉換後一樣共用記憶體
print(y)
'''
tensor([[1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.],
        [1., 1., 1.]], dtype=torch.float64)
'''
np.add(x, 1, out=x)
print(y)
'''
tensor([[2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.],
        [2., 2., 2.]], dtype=torch.float64)
'''


