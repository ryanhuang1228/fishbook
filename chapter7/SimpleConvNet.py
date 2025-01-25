import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from chapter5.TwoLayerNet import Relu, Affine, SoftmaxWithLoss

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_data.shape
    out_h = int(1 + (H + 2 * pad - filter_h) / stride)  # 输出高
    out_w = int(1 + (W + 2 * pad - filter_w) / stride)  # 输出宽

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')  # 填充0
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))  # 按照数据序号、通道、滤波器长、滤波器宽、输出长、输出宽
    # 对滤波器长和滤波器宽遍历，获取每一个（滤波器长，滤波器宽）坐标的数据集，大小为(输出长，输出宽)
    for i in range(filter_h):
        for j in range(filter_w):
            # 前两个:分别对应 数据序号和通道
            col[:, :, i, j, :, :] = img[:, :, i:i+stride*out_h:stride, j:j+stride*out_w:stride]
    
    col = col.transpose(0, 4, 5, 1, 2, 3)   # 因为col的每一行代表了通道(1)x滤波器长(2)x滤波器宽(3)的便利数据，因此把输出长(4)、输出宽(5)提前
    col = col.reshape(N*out_h*out_w, -1)    # 方便后续reshape为(数据序号(0)x输出长(4)x输出宽(5), 通道(1)x滤波器长(2)x滤波器宽(3))大小的矩阵
    return col

def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = int(1 + (H + 2 * pad - filter_h) / stride)  # 输出高
    out_w = int(1 + (W + 2 * pad - filter_w) / stride)  # 输出宽
    
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H+2*pad, W+2*pad))
    for i in range(filter_h):
        for j in range(filter_w):
            '''
            想象一下一个5x5的滤波器在7x7的矩阵中移动
            可以先想想滤波器的第一个点在矩阵中的移动
            filter_h, filter_w 保存了第n个点在矩阵中遍历的值，大小为(out_h, out_w)
            '''
            img[:, :, i:i+stride*out_h:stride, j:j+stride*out_w:stride] = col[:, :, i, j,:, :]
    return img[:, :, pad: H+pad, pad: W+pad]

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # W: (FN, C, FH, FW)
        self.b = b  # b: (FN, ) 详见图7-12 一个滤波器一个偏置
        self.stride = stride
        self.pad = pad

        # 反向传播数据
        self.x = None
        self.col = None
        self.col_W = None
        
    def forward(self, x):
        FN, C, FH, FW = self.W.shape    # 滤波器的数量，通道，高，宽
        N, C, H, W = x.shape       # 数据的数量，通道，高，宽
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)  # 输出高
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)  # 输出宽
        
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T    # 按列展开各个滤波器的权重 col_W: (, FN)
        
        out = np.dot(col, col_W) + self.b   # self.b: (FN, ) 广播
        # out: (N*out_h*out_w, FN) 
        out = out.reshape(N, out_h, out_w, -1)
        out = out.transpose(0, 3, 1, 2)  # 将轴变为数量, 滤波器数量, 高, 宽
        
        self.x = x
        self.col = col
        self.col_W = col_W
        return out
    
    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)  # self.dW: (-1, FN)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)  # 转置后再reshape

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)
        return dx

class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None
        
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2 * self.pad - self.pool_h) / self.stride)  # 输出高
        out_w = int(1 + (W + 2 * self.pad - self.pool_w) / self.stride)  # 输出宽

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)  # 该reshape与书上不符合，此时的排布为(1,2,3,1,2,3)通道依次循环
        
        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)  # 此时结果与书上相符，原因在于通道C是最后一个轴，与上述相符

        self.x = x
        self.arg_max = arg_max
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)   # 把通道换到最后一个轴
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))     # 事实上池化会缩小输入矩阵，因此这里需要扩大成输入矩阵大小
        '''对于取最大值的数传递导数，其余导数为0'''
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        '''N, H, W, C - > N, H, W, C, Pool_size'''
        dmax = dmax.reshape(dout.shape + (pool_size,))
        '''将矩阵整理成NxHxW行的形式，方便使用col2im'''
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class SimpleConvNet:
    def __init__(self, input_dim=(1, 28, 28), 
                conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]

        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))
        
        self.params = {}
        '''卷积层参数'''
        self.params["W1"] = weight_init_std * np.random.randn(filter_num, input_dim[0],
                                                              filter_size, filter_size)
        self.params["b1"] = np.zeros(filter_num)
        '''全连接层参数'''
        self.params["W2"] = weight_init_std * np.random.rand(pool_output_size, hidden_size)
        self.params["b2"] = np.zeros(hidden_size)
        '''输出层参数'''
        self.params["W3"] = weight_init_std * np.random.rand(hidden_size, output_size)
        self.params["b3"] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers["Conv1"] = Convolution(self.params["W1"], self.params["b1"],
                                           conv_param["stride"], conv_param["pad"])
        self.layers["Relu1"] = Relu()
        self.layers["Pool1"] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers["Affine1"] = Affine(self.params["W2"], self.params["b2"])
        self.layers["Relu2"] = Relu()
        self.layers["Affine2"] = Affine(self.params["W3"], self.params["b3"])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        
        for layer in layers:
            dout = layer.backward(dout)
        
        # 梯度更新
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t) / float(x.shape[0])
        
        return accuracy