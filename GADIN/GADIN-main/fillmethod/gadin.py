# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO 创新点实验项目，在GAIN代码的基础上改进
# TODO 特殊改进部分 搜索 hmf
'''GAIN function.
Date: 2020/02/28
Reference: J. Yoon, J. Jordon, M. van der Schaar, "GAIN: Missing Data 
           Imputation using Generative Adversarial Nets," ICML, 2018.
Paper Link: http://proceedings.mlr.press/v80/yoon18a/yoon18a.pdf
Contact: jsyoon0823@gmail.com
'''

# Necessary packages
#import tensorflow as tf
##IF USING TF 2 use following import to still use TF < 2.0 Functionalities
# TODO 注意tf 版本
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()

import tensorflow as tf
import numpy as np
# TODO 进步条模块
from tqdm import tqdm

# TODO python 如何调用其他.py里的方法
from fillmethod.utils import normalization, renormalization, rounding
from fillmethod.utils import xavier_init
from fillmethod.utils import binary_sampler, uniform_sampler, sample_batch_index
from sklearn.impute import KNNImputer
# TODO 根据value值，进行字典排序
# d是字典 ，num是取字典前num个最大value值对应的key
# return list
def sort_by_value(d, num):
  items = d.items()
  backitems = [[v[1], v[0]] for v in items]
  backitems.sort()
  # 得 list倒序排列
  backitems_list = backitems[::-1]
  d_list = []
  for i in range(len(backitems_list)):
    d_list.append(int(backitems_list[i][1]))
  return d_list[0:num]

# TODO hmf :gadin
def gadin (data_x, gain_parameters):
  '''Impute missing values in data_x

  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations

  Returns:
    - imputed_data: imputed data
  '''
  # Define mask matrix TODO 定义 M掩码矩阵
  data_m = 1-np.isnan(data_x)

  # System parameters TODO 模型参数：样本量，提示率，a，迭代次数
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  noise_num = gain_parameters['noise_num']

  # Other parameters TODO 矩阵行数和列数
  no, dim = data_x.shape

  # Hidden state dimensions TODO 隐藏层维度 = 属性个数
  h_dim = int(dim)

  # Normalization TODO 数据归一化，空换为0，就是随机噪声
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)  # todo 这是用0，下面用 KNN

  #
  # TODO hmf 在这里将随机噪声用KNN填补，不用0:改动 *************************** 还不如用0
  # KI = KNNImputer(n_neighbors=3, weights="uniform") # TODO KNN预填补
  # norm_data_x = KI.fit_transform(norm_data)
  # TODO hmf 在这里将随机噪声用KNN填补，不用0:改动 ***************************

  ## GAIN architecture
  # Input placeholders
  # Data vector
  # TODO 数据类型，行不定，列确定=属性列数 （placeholder是优化代码的）
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])

  # Discriminator variables
  # TODO 改进4：hmf 可考虑去掉 H，有些情况下，效果好**********************************************************************
  # TODO 初始化网络参数变量 通过xavier_init 形式，w1不同，节点数是其他的两倍  hmf 可考虑去掉 H，有些情况下，效果好
  # D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_W1 = tf.Variable(xavier_init([dim, h_dim])) # Data + Hint as inputs TODO 不乘2 hmf
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

  D_W2 = tf.Variable(xavier_init([h_dim, h_dim-2]))# TODO hmf 适当减少隐藏层节点数：注意前者是输入个数，后者是输出个数
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim-2]))

  D_W3 = tf.Variable(xavier_init([h_dim-2, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs

  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))
  # G_W1 = tf.Variable(xavier_init([dim, h_dim]))  # TODO 不乘2 hmf
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))

  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))

  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

  ## GAIN functions
  # TODO G和D 使用的都是 四层 除输入层是属性个数两倍，每层结点数相同的全连接神经网络 *******************************
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    # TODO tf.concat 是拼接向量或矩阵 axis = 1表示 矩阵左右拼接，=0表示矩阵上下拼接
    inputs = tf.concat(values = [x, m], axis = 1)
    # inputs = x  # TODO 输入 hmf
    # TODO tf.multiply 两个矩阵中对应元素相乘
    # TODO tf.matmul 将矩阵a乘以矩阵b，生成a * b 矩阵相乘
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob

  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint

    # inputs = tf.concat(values = [x, h], axis = 1)
    inputs = x  # TODO 这里没用到h  hmf
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

  ## GAIN structure
  # Generator
  # TODO 生成器预测样本
  G_sample = generator(X, M)

  # Combine with observed data
  # TODO 填补矩阵
  Hat_X = X * M + G_sample * (1-M)

  # Discriminator
  # TODO 概率矩阵
  D_prob = discriminator(Hat_X, H)

  ## GAIN loss
  # TODO tf.reduce_mean 计算均值
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8))
  # TODO 尝试修改 LOSS hmf 改动：
  # D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8))

  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))

  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss

  ## GAIN solver
  # TODO 根据损失函数，优化网络参数
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

  ## Iterations
  # TODO 执行tf
  sess = tf.Session()
  # TODO 有tf.Variable 的情况下，必须用下面的run形式
  sess.run(tf.global_variables_initializer())


  # Start Iterations
  # TODO 部分样本，迭代过程，优化G和D的过程
  # TODO tqdm 是显示进度条
  # TODO hmf 改进1：样本每次随机取，改为固定样本迭代训练：意义不大，多样本差距效果差不多

  # TODO 1 *********************************************************多样本差距不大，但多数下不好 begain
  # Sample batch
  # # TODO 每次随机取部分样本，以及对应 X，M
  # batch_idx = sample_batch_index(no, batch_size)
  # X_mb = norm_data_x[batch_idx, :]
  # M_mb = data_m[batch_idx, :]
  #
  # # Sample random vectors
  # Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
  # # Sample hint vectors
  # # TODO 提示率：生成数据时使用
  # H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
  # H_mb = M_mb * H_mb_temp
  #
  # # Combine random vectors with observed vectors
  # # TODO X 与 Z 是相加的关系
  # X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
  # TODO 1 ********************************************************* end

  for it in tqdm(range(iterations)):
    # TODO 1 ********************************************************* begain
    # # Sample batch
    # # TODO 每次随机取部分样本，以及对应 X，M
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]
    M_mb = data_m[batch_idx, :]

    # Sample random vectors
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
    # Sample hint vectors
    # TODO 提示率：生成数据时使用
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp

    # Combine random vectors with observed vectors
    # TODO X 与 Z 是相加的关系
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    # TODO 1 ********************************************************* end

    # TODO 2 *********************************************************加噪，效果好些 begain（2.1改进，分情况使用）

    # TODO 改进2：缺失率前三或前五，分别增加噪声，迭代：这个效果好些  hmf
    # TODO 制造噪声,矩阵某一列为0， X和 M都加噪声 ：效果不好，可测试高缺失情况  hmf

    # TODO 找到数据缺失率，并循环
    # TODO 每次循环后，数据还原，保证每次循环只有一个为0的列 hmf:但这效果不如不还原 ********2.1改进版：待考虑（看实际情况，删减或保留）
    X_mb_copy = X_mb
    M_mb_copy = M_mb
    #TODO** ** ** ** 2.1改进版

    m_dict = {}
    for i in range(M_mb.shape[1]):
      num = 0
      for j in M_mb[:, i]:
        if j == 0:
          num = num + 1
      m_dict[str(i)] = num
    # 取前5个，缺失率最大的列下标 TODO 这个2或5,可作为参数 hmf
    d_list = sort_by_value(m_dict, noise_num)
    # TODO 每次for循环后，应该有个数据还原，这就需要改变参数 hmf
    for i in d_list:
      X_mb[:, i] = X_mb[:, i] * 0
      M_mb[:, i] = M_mb[:, i] * 0

      H_mb = M_mb * H_mb_temp  #TODO 是否去掉 H。 hmf:去掉效果差些，但差距不算太大
      # TODO run执行上面对应程序
      _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                feed_dict={M: M_mb, X: X_mb, H: H_mb})
      _, G_loss_curr, MSE_loss_curr = \
        sess.run([G_solver, G_loss_temp, MSE_loss],
                 feed_dict={X: X_mb, M: M_mb, H: H_mb})
    #TODO 每次循环后，数据还原，保证每次循环只有一个为0的列 hmf:但这效果不如不还原 *********2.1 改进版：待考虑
      X_mb = X_mb_copy
      M_mb = M_mb_copy
    #TODO** ** ** ** 2.1改进版
    # TODO 2 ********************************************************* end

    # # TODO 3 *********************************************************加噪+交替 begain：加交替，效果不好 begain
    # # TODO 改进3：加噪+交替 hmf:效果不好
    # # # TODO ********2.1 改进版：待考虑
    # X_mb_copy = X_mb
    # M_mb_copy = M_mb
    # H_mb_copy = H_mb
    # # TODO ********2.1
    # m_dict = {}
    # for i in range(M_mb.shape[1]):
    #   num = 0
    #   for j in M_mb[:, i]:
    #     if j == 0:
    #       num = num + 1
    #   m_dict[str(i)] = num
    # # 取前5个，缺失率最大的列下标 TODO 这个5,可作为参数 hmf
    # d_list = sort_by_value(m_dict, 4)
    #
    # for i in range(len(d_list)):
    #   X_mb[:, d_list[i]] = X_mb[:, d_list[i]] * 0
    #   M_mb[:, d_list[i]] = M_mb[:, d_list[i]] * 0
    #   H_mb = M_mb * H_mb_temp
    #   # TODO run执行上面对应程序
    #   _, D_loss_curr = sess.run([D_solver, D_loss_temp],
    #                             feed_dict={M: M_mb, X: X_mb, H: H_mb})
    #   _, G_loss_curr, MSE_loss_curr = \
    #     sess.run([G_solver, G_loss_temp, MSE_loss],
    #              feed_dict={X: X_mb, M: M_mb, H: H_mb})
    #
    #   # TODO 每次循环后，数据还原，保证每次循环只有一个为0的列 hmf:但这效果不如不还原 *****hmf 2.1 改进版 待考虑
    #   X_mb = X_mb_copy
    #   M_mb = M_mb_copy
    #   # TODO ********2.1
    #
    #   # # # TODO 使用G生成试试，交替填补
    #   imputed_data_for = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    #   imputed_data_for = M_mb * X_mb + (1 - M_mb) * imputed_data_for
    #   X_mb[:, d_list[i]] = imputed_data_for[:, d_list[i]]

    # TODO 3 ********************************************************* end


    # TODO 2 ********************************************************* begain
    # # TODO run执行上面对应程序
    # _, D_loss_curr = sess.run([D_solver, D_loss_temp],
    #                           feed_dict = {M: M_mb, X: X_mb, H: H_mb})
    # _, G_loss_curr, MSE_loss_curr = \
    # sess.run([G_solver, G_loss_temp, MSE_loss],
    #          feed_dict = {X: X_mb, M: M_mb, H: H_mb})
    # TODO 2 *********************************************************  end


  ## Return imputed data
  Z_mb = uniform_sampler(0, 0.01, no, dim)
  M_mb = data_m
  # TODO 数据矩阵
  X_mb = norm_data_x
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

  # TODO 运行生成器
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]

  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

  # Renormalization TODO 反归一化
  imputed_data = renormalization(imputed_data, norm_parameters)

  # Rounding TODO 对分类变量进行四舍五入的估算
  imputed_data = rounding(imputed_data, data_x)

  return imputed_data



# TODO hmf :gadin_model 主要返回的是误差值
def gadin_model (data_x, gain_parameters):
  '''Impute missing values in data_x

  Args:
    - data_x: original data with missing values
    - gain_parameters: GAIN network parameters:
      - batch_size: Batch size
      - hint_rate: Hint rate
      - alpha: Hyperparameter
      - iterations: Iterations

  Returns:
    - imputed_data: imputed data
  '''
  # todo 误差值保存到list
  rmse_list = []

  # Define mask matrix TODO 定义 M掩码矩阵
  data_m = 1-np.isnan(data_x)

  # System parameters TODO 模型参数：样本量，提示率，a，迭代次数
  batch_size = gain_parameters['batch_size']
  hint_rate = gain_parameters['hint_rate']
  alpha = gain_parameters['alpha']
  iterations = gain_parameters['iterations']
  noise_num = gain_parameters['noise_num']

  # Other parameters TODO 矩阵行数和列数
  no, dim = data_x.shape

  # Hidden state dimensions TODO 隐藏层维度 = 属性个数
  h_dim = int(dim)

  # Normalization TODO 数据归一化，空换为0，就是随机噪声
  norm_data, norm_parameters = normalization(data_x)
  norm_data_x = np.nan_to_num(norm_data, 0)  # todo 这是用0，下面用 KNN

  #
  # TODO hmf 在这里将随机噪声用KNN填补，不用0:改动 *************************** 还不如用0
  # KI = KNNImputer(n_neighbors=3, weights="uniform") # TODO KNN预填补
  # norm_data_x = KI.fit_transform(norm_data)
  # TODO hmf 在这里将随机噪声用KNN填补，不用0:改动 ***************************

  ## GAIN architecture
  # Input placeholders
  # Data vector
  # TODO 数据类型，行不定，列确定=属性列数 （placeholder是优化代码的）
  X = tf.placeholder(tf.float32, shape = [None, dim])
  # Mask vector
  M = tf.placeholder(tf.float32, shape = [None, dim])
  # Hint vector
  H = tf.placeholder(tf.float32, shape = [None, dim])

  # Discriminator variables
  # TODO 改进4：hmf 可考虑去掉 H，有些情况下，效果好**********************************************************************
  # TODO 初始化网络参数变量 通过xavier_init 形式，w1不同，节点数是其他的两倍  hmf 可考虑去掉 H，有些情况下，效果好
  # D_W1 = tf.Variable(xavier_init([dim*2, h_dim])) # Data + Hint as inputs
  D_W1 = tf.Variable(xavier_init([dim, h_dim])) # Data + Hint as inputs TODO 不乘2 hmf
  D_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

  D_W2 = tf.Variable(xavier_init([h_dim, h_dim-2]))# TODO hmf 适当减少隐藏层节点数：注意前者是输入个数，后者是输出个数
  D_b2 = tf.Variable(tf.zeros(shape = [h_dim-2]))

  D_W3 = tf.Variable(xavier_init([h_dim-2, dim]))
  D_b3 = tf.Variable(tf.zeros(shape = [dim]))  # Multi-variate outputs

  theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]


  #Generator variables
  # Data + Mask as inputs (Random noise is in missing components)
  G_W1 = tf.Variable(xavier_init([dim*2, h_dim]))
  # G_W1 = tf.Variable(xavier_init([dim, h_dim]))  # TODO 不乘2 hmf
  G_b1 = tf.Variable(tf.zeros(shape = [h_dim]))

  G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
  G_b2 = tf.Variable(tf.zeros(shape = [h_dim]))

  G_W3 = tf.Variable(xavier_init([h_dim, dim]))
  G_b3 = tf.Variable(tf.zeros(shape = [dim]))

  theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

  ## GAIN functions
  # TODO G和D 使用的都是 四层 除输入层是属性个数两倍，每层结点数相同的全连接神经网络 *******************************
  # Generator
  def generator(x,m):
    # Concatenate Mask and Data
    # TODO tf.concat 是拼接向量或矩阵 axis = 1表示 矩阵左右拼接，=0表示矩阵上下拼接
    inputs = tf.concat(values = [x, m], axis = 1)
    # inputs = x  # TODO 输入 hmf
    # TODO tf.multiply 两个矩阵中对应元素相乘
    # TODO tf.matmul 将矩阵a乘以矩阵b，生成a * b 矩阵相乘
    G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
    # MinMax normalized output
    G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
    return G_prob

  # Discriminator
  def discriminator(x, h):
    # Concatenate Data and Hint

    # inputs = tf.concat(values = [x, h], axis = 1)
    inputs = x  # TODO 这里没用到h  hmf
    D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_logit = tf.matmul(D_h2, D_W3) + D_b3
    D_prob = tf.nn.sigmoid(D_logit)
    return D_prob

  ## GAIN structure
  # Generator
  # TODO 生成器预测样本
  G_sample = generator(X, M)

  # Combine with observed data
  # TODO 填补矩阵
  Hat_X = X * M + G_sample * (1-M)

  # Discriminator
  # TODO 概率矩阵
  D_prob = discriminator(Hat_X, H)

  ## GAIN loss
  # TODO tf.reduce_mean 计算均值
  D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) \
                                + (1-M) * tf.log(1. - D_prob + 1e-8))
  # TODO 尝试修改 LOSS hmf 改动：
  # D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8))

  G_loss_temp = -tf.reduce_mean((1-M) * tf.log(D_prob + 1e-8))

  MSE_loss = \
  tf.reduce_mean((M * X - M * G_sample)**2) / tf.reduce_mean(M)

  D_loss = D_loss_temp
  G_loss = G_loss_temp + alpha * MSE_loss

  ## GAIN solver
  # TODO 根据损失函数，优化网络参数
  D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
  G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

  ## Iterations
  # TODO 执行tf
  sess = tf.Session()
  # TODO 有tf.Variable 的情况下，必须用下面的run形式
  sess.run(tf.global_variables_initializer())


  # Start Iterations
  # TODO 部分样本，迭代过程，优化G和D的过程
  # TODO tqdm 是显示进度条
  # TODO hmf 改进1：样本每次随机取，改为固定样本迭代训练：意义不大，多样本差距效果差不多

  # TODO 1 *********************************************************多样本差距不大，但多数下不好 begain
  # Sample batch
  # # TODO 每次随机取部分样本，以及对应 X，M
  # batch_idx = sample_batch_index(no, batch_size)
  # X_mb = norm_data_x[batch_idx, :]
  # M_mb = data_m[batch_idx, :]
  #
  # # Sample random vectors
  # Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
  # # Sample hint vectors
  # # TODO 提示率：生成数据时使用
  # H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
  # H_mb = M_mb * H_mb_temp
  #
  # # Combine random vectors with observed vectors
  # # TODO X 与 Z 是相加的关系
  # X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
  # TODO 1 ********************************************************* end

  for it in tqdm(range(iterations)):
    # TODO 1 ********************************************************* begain
    # # Sample batch
    # # TODO 每次随机取部分样本，以及对应 X，M
    batch_idx = sample_batch_index(no, batch_size)
    X_mb = norm_data_x[batch_idx, :]
    M_mb = data_m[batch_idx, :]

    # Sample random vectors
    Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
    # Sample hint vectors
    # TODO 提示率：生成数据时使用
    H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
    H_mb = M_mb * H_mb_temp

    # Combine random vectors with observed vectors
    # TODO X 与 Z 是相加的关系
    X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    # TODO 1 ********************************************************* end

    # TODO 2 *********************************************************加噪，效果好些 begain（2.1改进，分情况使用）

    # TODO 改进2：缺失率前三或前五，分别增加噪声，迭代：这个效果好些  hmf
    # TODO 制造噪声,矩阵某一列为0， X和 M都加噪声 ：效果不好，可测试高缺失情况  hmf

    # TODO 找到数据缺失率，并循环
    # TODO 每次循环后，数据还原，保证每次循环只有一个为0的列 hmf:但这效果不如不还原 ********2.1改进版：待考虑（看实际情况，删减或保留）
    X_mb_copy = X_mb
    M_mb_copy = M_mb
    #TODO** ** ** ** 2.1改进版

    m_dict = {}
    for i in range(M_mb.shape[1]):
      num = 0
      for j in M_mb[:, i]:
        if j == 0:
          num = num + 1
      m_dict[str(i)] = num
    # 取前5个，缺失率最大的列下标 TODO 这个2或5,可作为参数 hmf
    d_list = sort_by_value(m_dict, noise_num)
    # TODO 每次for循环后，应该有个数据还原，这就需要改变参数 hmf
    for i in d_list:
      X_mb[:, i] = X_mb[:, i] * 0
      M_mb[:, i] = M_mb[:, i] * 0

      H_mb = M_mb * H_mb_temp  #TODO 是否去掉 H。 hmf:去掉效果差些，但差距不算太大
      # TODO run执行上面对应程序
      _, D_loss_curr = sess.run([D_solver, D_loss_temp],
                                feed_dict={M: M_mb, X: X_mb, H: H_mb})
      _, G_loss_curr, MSE_loss_curr = \
        sess.run([G_solver, G_loss_temp, MSE_loss],
                 feed_dict={X: X_mb, M: M_mb, H: H_mb})
    #TODO 每次循环后，数据还原，保证每次循环只有一个为0的列 hmf:但这效果不如不还原 *********2.1 改进版：待考虑
      X_mb = X_mb_copy
      M_mb = M_mb_copy

    rmse_list.append(round(MSE_loss_curr, 4))

  ## Return imputed data
  Z_mb = uniform_sampler(0, 0.01, no, dim)
  M_mb = data_m
  # TODO 数据矩阵
  X_mb = norm_data_x
  X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

  # todo 尝试保存模型
  # saver = tf.train.Saver()
  # saver.save(sess, f"F:/model_save")

  # TODO 运行生成器
  imputed_data = sess.run([G_sample], feed_dict = {X: X_mb, M: M_mb})[0]

  # todo 只填补缺失位置数据
  imputed_data = data_m * norm_data_x + (1-data_m) * imputed_data

  # Renormalization TODO 反归一化
  imputed_data = renormalization(imputed_data, norm_parameters)

  # Rounding TODO 对分类变量进行四舍五入的估算
  imputed_data = rounding(imputed_data, data_x)

  return imputed_data, rmse_list, theta_G
