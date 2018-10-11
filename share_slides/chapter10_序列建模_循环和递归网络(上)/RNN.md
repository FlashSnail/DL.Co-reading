# RNN基本原理

- #### RNN基本框架

  ![1539165242340](C:\Users\lixian5\AppData\Roaming\Typora\typora-user-images\1539165242340.png)

- 图为RNN基础框架的计算图：

  ![img](file:///c:\users\lixian5\documents\jddongdong\jimenterprise\lixian283\temp\jdonline20181010131324.png)

- #### RNN正向传播

  ##### 正向传播公式：

  $$
  h_t=\phi(Ux_{t}+Wh_{t-1}+b)，\phi是tanh()函数
  $$

  $$
  o_t=Vh_{t}+c
  $$

  $$
  \hat{y}_t=\varphi(o_t)，\varphi是softmax()函数
  $$

  $$
  L_t=f(\hat{y}_t,y_t)
  $$

  $$
  L=\sum_{i=1}^{\tau}{L_t}
  $$

  ###### 维度定义，例子：

  x,h都是向量，用**黑体字母**表示；而U、V是**矩阵**，用大写字母表示。**向量的下标**表示**时刻**，例如，表示在t时刻向量h的值。

  我们假设输入向量x的维度是m，输出向量h的维度是n，则矩阵U的维度是nxm，矩阵W的维度是nxn
  $$
  \begin{align}
  \begin{bmatrix}
  h_1^t\\
  h_2^t\\
  .\\.\\
  h_n^t\\
  \end{bmatrix}=\phi(
  \begin{bmatrix}
  u_{11} u_{12} ... u_{1m}\\
  u_{21} u_{22} ... u_{2m}\\
  .\\.\\
  u_{n1} u_{n2} ... u_{nm}\\
  \end{bmatrix}
  \begin{bmatrix}
  x_1\\
  x_2\\
  .\\.\\
  x_m\\
  \end{bmatrix}+
  \begin{bmatrix}
  w_{11} w_{12} ... w_{1n}\\
  w_{21} w_{22} ... w_{2n}\\
  .\\.\\
  w_{n1} w_{n2} ... w_{nn}\\
  \end{bmatrix}
  \begin{bmatrix}
  h_1^{t-1}\\
  h_2^{t-1}\\
  .\\.\\
  h_n^{t-1}\\
  \end{bmatrix})
  \end{align}
  $$

- #### RNN反向传播

  反向传播求导：

  ##### 1.对o(t)求导：L对o(t)只有一个路径，所以不需要τ个相加

  $$
  \frac{\partial L}{\partial L_t}=1
  \\\frac{\partial L}{\partial o_t}=\frac{\partial L}{\partial L_t}\frac{\partial L_t}{\partial o_t}=\hat{y}_t-y_t
  $$

  ##### 2.对h(t)求导：分两种情况,t不等于τ的时候会传2条路径，所以要相加

  $$
  当 t=\tau时:
  \\\frac{\partial L}{\partial h_{\tau}}=\frac{\partial L}{\partial o_{\tau}}\frac{\partial o_{\tau}}{\partial h_{\tau}}=V^T\frac{\partial L}{\partial o_{\tau}}
  \\当t\neq\tau时：
  \\\frac{\partial L}{\partial h_{t}}=\frac{\partial L}{\partial o_{t}}\frac{\partial o_{t}}{\partial h_{t}}+\frac{\partial L}{\partial h_{t+1}}\frac{\partial h_{t+1}}{\partial h_{t}}\\
  =(\frac{\partial o_t}{\partial h_t})^T\frac{\partial L}{\partial o_{t}}+(\frac{\partial h_{t+1}}{\partial h_{t}})^T\frac{\partial L}{\partial h_{t+1}}\\
  =V^T\frac{\partial L}{\partial o_{t}}+W^T(\frac{\partial L}{\partial h_{t+1}})diag(1-(h_{t+1})^2)
  $$



  ##### 对V，c的求导：V,c是对于τ个路径来说是共享的，所以需要进行τ次的叠加

$$
o_t=Vh_{t}+c
$$

![1539163933102](D:\02.Structure\markdown笔记\%5CUsers%5Clixian5%5CAppData%5CRoaming%5CTypora%5Ctypora-user-images%5C1539163933102.png)

$$
\frac{\partial L}{\partial c}=\sum_{t=1}^{\tau}{\frac{\partial L}{\partial o_t}}\frac{\partial o_t}{\partial c}=\sum_{t=1}^{\tau}{\frac{\partial L}{\partial o_t}}
$$

$$
\frac{\partial L}{\partial V}=\sum_{t=1}^{\tau}{\frac{\partial L}{\partial o_t}\frac{\partial o_t}{\partial V}}\\
=\sum_{t=1}^{\tau}{\frac{\partial L}{\partial o_t}}(h_t)^T
$$

##### 对W,U的求导：W,U是对于τ个路径来说是共享的，所以需要进行τ次的叠加

$$
h_t=\phi(Ux_{t}+Wh_{t-1}+b)，\phi是tanh()函数
$$

![1539164026838](C:\Users\lixian5\AppData\Roaming\Typora\typora-user-images\1539164026838.png)
$$
\frac{\partial L}{\partial W}=\sum_{t=1}^{\tau}{\frac{\partial L}{\partial h_t}}\frac{\partial h_t}{\partial W}\\
=\sum_{t=1}^{\tau}{diag(1-(h_t)^2)}\frac{\partial L}{\partial h_t}(h_{t-1})^T
$$

$$
\frac{\partial L}{\partial U}=\sum_{t=1}^{\tau}{\frac{\partial L}{\partial h_t}}\frac{\partial h_t}{\partial U}\\
=\sum_{t=1}^{\tau}{diag(1-(h_t)^2)}\frac{\partial L}{\partial h_t}(x_{t})^T
$$



- #### 参考文件

  [循环神经网络(RNN)模型与前向反向传播算法](https://www.cnblogs.com/pinard/p/6509630.html)

  [数学 · RNN（二）· BPTT 算法](https://zhuanlan.zhihu.com/p/26892413)

  [详解循环神经网络(Recurrent Neural Network)](https://www.jianshu.com/p/39a99c88a565)


