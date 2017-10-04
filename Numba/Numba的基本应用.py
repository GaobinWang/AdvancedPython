# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 14:31:16 2017

@author: Lesile
"""

"""
Example1:使用 jit 加速 Python 低效的 for 语句
jit 的全称是 Just-in-time，在 numba 里面则特指 Just-in-time compilation（即时编译）。
例子：给数组中的每个数加上一个常数 c：
"""
import numba as nb
import numpy as np
import matplotlib.pyplot as plt


# 普通的 for
def add1(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs

# list comprehension
def add2(x, c):
    return [xx + c for xx in x]

# 使用 jit 加速后的 for
@nb.jit(nopython=True)
def add_with_jit(x, c):
    rs = [0.] * len(x)
    for i, xx in enumerate(x):
        rs[i] = xx + c
    return rs

y = np.random.random(10**5).astype(np.float32)
x = y.tolist()

assert np.allclose(add1(x, 1), add2(x, 1), add_with_jit(x, 1))
%timeit add1(x, 1)
%timeit add2(x, 1)
%timeit add_with_jit(x, 1)
print(np.allclose(add1(x, 1),add2(x, 1), 1))

"""
###numba不支持 list comprehension，详情可参见这里
* jit能够加速的不限于for，但一般而言加速for会比较常见、效果也比较显著。
* jit会在某种程度上“预编译”你的代码，这意味着它会在某种程度上固定住各个变量的数据类型；
所以在jit下定义数组时，如果想要使用的是float数组的话，就不能用[0] * len(x)定义、而应该像上面那样在0后面加一个小数点：[0.] * len(x)
"""

"""
Example2:使用 vectorize 实现 numpy 的 Ufunc 功能
"""
#虽然jit确实能让我们代码加速不少，但比之numpy的Ufunc还是要差很多
assert np.allclose(y + 1, add_with_jit(x, 1))
%timeit add_with_jit(x, 1)
%timeit y + 1
#可以看到几乎有 200 倍的差距，这当然是无法忍受的。
#为此，我们可以用vectorize来定义出类似于Ufunc的函数：
@nb.vectorize(nopython=True)
def add_with_vec(yy, c):
    return yy + c

assert np.allclose(y + 1, add_with_vec(y, 1), add_with_vec(y, 1.))
%timeit add_with_vec(y, 1)
%timeit add_with_vec(y, 1.)
%timeit y + 1
%timeit y + 1.
#虽然还是慢了 2 倍左右，但已经好很多了

#此外，vectorize最炫酷的地方在于，它可以“并行”：
@nb.vectorize("float32(float32, float32)", target="parallel", nopython=True)
def add_with_vec(y, c):
    return y + c

assert np.allclose(y+1, add_with_vec(y,1.))
%timeit add_with_vec(y, 1.)
%timeit y + 1

#似乎还变慢了；不过如果使用 Intel Distribution for Python 的话，会发现parallel版本甚至会比numpy原生的版本要稍快一些

###parallel总会更好的例子
# 将数组所有元素限制在某个区间[a, b]内
# 小于 a 则置为 a，大于 b 则置为 b
# 经典应用：ReLU
 
@nb.vectorize("float32(float32, float32, float32)", target="parallel", nopython=True)
def clip_with_parallel(y, a, b):
    if y < a:
        return a
    if y > b:
        return b
    return y

@nb.vectorize("float32(float32, float32, float32)", nopython=True)
def clip(y, a, b):
    if y < a:
        return a
    if y > b:
        return b
    return y

assert np.allclose(np.clip(y, 0.1, 0.9), clip(y, 0.1, 0.9), clip_with_parallel(y, 0.1, 0.9))
%timeit clip_with_parallel(y, 0.1, 0.9)
%timeit clip(y, 0.1, 0.9)
%timeit np.clip(y, 0.1, 0.9)
#这个栗子中的性能提升就是实打实的了。总之，使用parallel时不能一概而论，还是要做些实验
#?为何我的电脑上并行处理较慢呢？

"""
需要指出的是，vectorize中的参数target一共有三种取值：cpu（默认）、parallel和cuda。关于选择哪个取值，官方文档上有很好的说明：
A general guideline is to choose different targets for different data sizes and algorithms. 
The “cpu” target works well for small data sizes (approx. less than 1KB) and low compute intensity algorithms. 
It has the least amount of overhead. 
The “parallel” target works well for medium data sizes (approx. less than 1MB). 
Threading adds a small delay. The “cuda” target works well for big data sizes (approx. greater than 1MB) and high compute intensity algorithms.
 Transfering memory to and from the GPU adds significant overhead.
"""

"""
Example3:使用 jit(nogil=True) 实现高效并发（多线程）
我们知道，Python 中由于有 GIL 的存在，所以多线程用起来非常不舒服。
不过 numba 的 jit 里面有一项参数叫 nogil，想来聪明的观众老爷们都猜到了它是干什么的了……
下面就来看一个栗子：

这个栗子有点长，不过我们只需要知道如下两点即可：
* make_single_task和make_multi_task分别生成单线程函数和多线程函数
* 生成的函数会调用相应的kernel来完成计算
"""
import math
from concurrent.futures import ThreadPoolExecutor

# 计算类似于 Sigmoid 的函数
def np_func(a, b):
    return 1 / (a + np.exp(-b))

# 参数中的 result 代表的即是我们想要的结果，后同
# 第一个 kernel，nogil 参数设为了 False
@nb.jit(nopython=True, nogil=False)
def kernel1(result, a, b):
    for i in range(len(result)):
        result[i] = 1 / (a[i] + math.exp(-b[i]))

# 第二个 kernel，nogil 参数设为了 True
@nb.jit(nopython=True, nogil=True)
def kernel2(result, a, b):
    for i in range(len(result)):
        result[i] = 1 / (a[i] + math.exp(-b[i]))

def make_single_task(kernel):
    def func(length, *args):
        result = np.empty(length, dtype=np.float32)
        kernel(result, *args)
        return result
    return func

def make_multi_task(kernel, n_thread):
    def func(length, *args):
        result = np.empty(length, dtype=np.float32)
        args = (result,) + args
        # 将每个线程接受的参数定义好
        chunk_size = (length + n_thread - 1) // n_thread
        chunks = [[arg[i*chunk_size:(i+1)*chunk_size] for i in range(n_thread)] for arg in args]
        # 利用 ThreadPoolExecutor 进行并发
        with ThreadPoolExecutor(max_workers=n_thread) as e:
            for _ in e.map(kernel, *chunks):
                pass
        return result
    return func

length = 10 ** 6
a = np.random.rand(length).astype(np.float32)
b = np.random.rand(length).astype(np.float32)

nb_func1 = make_single_task(kernel1)
nb_func2 = make_multi_task(kernel1, 4)
nb_func3 = make_single_task(kernel2)
nb_func4 = make_multi_task(kernel2, 4)

rs_np = np_func(a, b)
rs_nb1 = nb_func1(length, a, b)
rs_nb2 = nb_func2(length, a, b)
rs_nb3 = nb_func3(length, a, b)
rs_nb4 = nb_func4(length, a, b)
assert np.allclose(rs_np, rs_nb1, rs_nb2, rs_nb3, rs_nb4)
%timeit np_func(a, b)
%timeit nb_func1(length, a, b)
%timeit nb_func2(length, a, b)
%timeit nb_func3(length, a, b)
%timeit nb_func4(length, a, b)

#一般来说，数据量越大、并发的效果越明显。反之，数据量小的时候，并发很有可能会降低性能

"""
Example4:numba 的应用实例 —— 卷积与池化

"""
import numba as nb
import numpy as np

# 普通的卷积
def conv_kernel(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(n_filters):
                    rs[i, q, j, p] += np.sum(w[q] * window)
    return rs

# 简单地加了个 jit 后的卷积
@nb.jit(nopython=True)
def jit_conv_kernel(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                window = x[i, ..., j:j+filter_height, p:p+filter_width]
                for q in range(n_filters):
                    rs[i, q, j, p] += np.sum(w[q] * window)

# 卷积操作的封装
def conv(x, w, kernel, args):
    n, n_filters = args[0], args[4]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    kernel(x, w, rs, *args)
    return rs

# 64 个 3 x 28 x 28 的图像输入（模拟 mnist）
x = np.random.randn(64, 3, 28, 28).astype(np.float32)
# 16 个 5 x 5 的 kernel
w = np.random.randn(16, x.shape[1], 5, 5).astype(np.float32)

n, n_channels, height, width = x.shape
n_filters, _, filter_height, filter_width = w.shape
out_h = height - filter_height + 1
out_w = width - filter_width + 1
args = (n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w)

print(np.linalg.norm((conv(x, w, conv_kernel, args) - conv(x, w, jit_conv_kernel, args)).ravel()))
%timeit conv(x, w, conv_kernel, args)
%timeit conv(x, w, jit_conv_kernel, args)
#可以看到，仅仅是加了一个jit、速度就直接提升了十多倍


##同时需要特别注意的是，使用jit和使用纯numpy进行编程的很大一点不同就是，不要畏惧用for；事实上一般来说，代码“长得越像 C”、速度就会越快：
@nb.jit(nopython=True)
def jit_conv_kernel2(x, w, rs, n, n_channels, height, width, n_filters, filter_height, filter_width, out_h, out_w):
    for i in range(n):
        for j in range(out_h):
            for p in range(out_w):
                for q in range(n_filters):
                    for r in range(n_channels):
                        for s in range(filter_height):
                            for t in range(filter_width):
                                rs[i, q, j, p] += x[i, r, j+s, p+t] * w[q, r, s, t]
                                
assert np.allclose(conv(x, w, jit_conv_kernel, args), conv(x, w, jit_conv_kernel, args))
%timeit conv(x, w, jit_conv_kernel, args)
%timeit conv(x, w, jit_conv_kernel2, args)

###池化操作
# 普通的 MaxPool
def max_pool_kernel(x, rs, *args):
    n, n_channels, pool_height, pool_width, out_h, out_w = args
    for i in range(n):
        for j in range(n_channels):
            for p in range(out_h):
                for q in range(out_w):
                    window = x[i, j, p:p+pool_height, q:q+pool_width]
                    rs[i, j, p, q] += np.max(window)

# 简单地加了个 jit 后的 MaxPool
@nb.jit(nopython=True)
def jit_max_pool_kernel(x, rs, *args):
    n, n_channels, pool_height, pool_width, out_h, out_w = args
    for i in range(n):
        for j in range(n_channels):
            for p in range(out_h):
                for q in range(out_w):
                    window = x[i, j, p:p+pool_height, q:q+pool_width]
                    rs[i, j, p, q] += np.max(window)

# 不惧用 for 的、“更像 C”的 MaxPool
@nb.jit(nopython=True)
def jit_max_pool_kernel2(x, rs, *args):
    n, n_channels, pool_height, pool_width, out_h, out_w = args
    for i in range(n):
        for j in range(n_channels):
            for p in range(out_h):
                for q in range(out_w):
                    _max = x[i, j, p, q]
                    for r in range(pool_height):
                        for s in range(pool_width):
                            _tmp = x[i, j, p+r, q+s]
                            if _tmp > _max:
                                _max = _tmp
                    rs[i, j, p, q] += _max

# MaxPool 的封装
def max_pool(x, kernel, args):
    n, n_channels = args[:2]
    out_h, out_w = args[-2:]
    rs = np.zeros([n, n_filters, out_h, out_w], dtype=np.float32)
    kernel(x, rs, *args)
    return rs

pool_height, pool_width = 2, 2
n, n_channels, height, width = x.shape
out_h = height - pool_height + 1
out_w = width - pool_width + 1
args = (n, n_channels, pool_height, pool_width, out_h, out_w)

assert np.allclose(max_pool(x, max_pool_kernel, args), max_pool(x, jit_max_pool_kernel, args))
assert np.allclose(max_pool(x, jit_max_pool_kernel, args), max_pool(x, jit_max_pool_kernel2, args))
%timeit max_pool(x, max_pool_kernel, args)
%timeit max_pool(x, jit_max_pool_kernel, args)
%timeit max_pool(x, jit_max_pool_kernel2, args)

