# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 15:48:12 2017

@author: Lesile

功能:用Python解决优化问题.其参考链接如下：
https://docs.scipy.org/doc/scipy/reference/optimize.html
"""

"""
#scipy.optimize.minimize
参考链接:https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
功能简介:解决多变量函数的有约束优化问题(局部最优化),其问题形式如下:
    minimize f(x) subject to
    g_i(x) >= 0,  i = 1,...,m
    h_j(x)  = 0,  j = 1,...,p
用法:minimize(fun, x0, args=(), method=None,jac=None, 
            hess=None, hessp=None,bounds=None, 
            constraints=(),tol=None, callback=None, 
            options=None)
参数简介：
fun:目标函数(callable)
x0:初始值(ndarray)
args:目标函数的额外参数或者其雅克比行列式(tuple, optional)
method:优化算法,默认值为'BFGS'(str or callable, optional)
jac：目标函数的梯度(callable, optional)
hess:目标函数的Hessian矩阵(callable, optional)
bounds:优化变量的上下界‘(only for L-BFGS-B, TNC and SLSQP)’(sequence, optional)
constraints:有约束优化问题的约束条件(dict or sequence of dict, optional)
ps:Constraints definition (only for COBYLA and SLSQP))
tol:算法终止条件的精度(float, optional)
options:优化算法的选项(dict, optional)
callback:Called after each iteration(callable, optional)

constraints可选的参数:
type:不等式的类型,‘eq’代表等式;‘ineq’代表不等式. 
fun:约束函数(callable)
jac:约束函数的梯度(only for SLSQP)
args:约束函数中的额外参数
ps:默认等式约束等于0;不等式约束非负.(COBYLA仅支持不等式约束)

options的可选参数(以SLSQP为例):
ftol:目标函数终止准则中的精度目标(float)
eps:jacobian数值近似过程中用到的步长(float)
disp:是否打印收敛信息(bool)
maxiter:最大迭代次数(int)

返回对象res参数简介:
x	(ndarray):最优函数值对应的自变量的值
success(bool):优化结果是否成功
status(int)：状态
message(str):优化结果信息
fun:目标函数最优解
jac:目标函数最后的Jacobian
hess:目标函数最后的Hessian
nfev(int):目标函数被估计的次数
njev(int):目标函数的Jacobian被估计的次数
nhev(int):目标函数的Hessian被估计的次数
nit(int):优化器的迭代次数
"""

###Example
from scipy.optimize import minimize, rosen, rosen_der
x0 = [1.3, 0.7, 0.8, 1.9, 1.2]
res = minimize(rosen, x0, method='Nelder-Mead', tol=1e-6)
res.x
print(res.message)

###Example
fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
cons = ({'type': 'ineq', 'fun': lambda x:  x[0] - 2 * x[1] + 2},
        {'type': 'ineq', 'fun': lambda x: -x[0] - 2 * x[1] + 6},
        {'type': 'ineq', 'fun': lambda x: -x[0] + 2 * x[1] + 2})
bnds = ((0, 1000000), (0, 1000000))
res = minimize(fun, (2, 0), method='SLSQP', bounds=bnds,
               constraints = cons)
res.x  #x的数值
res.fun #
print(res.message)
