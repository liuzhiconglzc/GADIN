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

'''Main function for UCI letter and spam datasets.
'''

# Necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
from sqlalchemy.orm.scoping import prop

from fillmethod.data_loader import data_loader, data_loader_new
from fillmethod.gain import gain, gain_fill
from fillmethod.utils import rmse_loss
# TODO PC-GAIN 的聚类算法, 自己加了一个均值填补法
from fillmethod.cluster import KM, SC, KMPP, AC, mean_fill
# TODO 分类器相关
import sklearn.svm as svm
from sklearn import preprocessing

# TODO 缺失森林填补法 ：版本问题解决不了
from impyute.imputation.cs import mice

# TODO KNN填补法
import pandas as pd
from sklearn.impute import KNNImputer

# TODO AE填补法
from fillmethod.AE import ae_impute, ae_impute_fill

# TODO GADIN填补法
from fillmethod.gadin import gadin, gadin_model
# TODO 图像展示
import matplotlib.pyplot as plt

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer todo 有问题
#
# from missingpy import MissForest
from fillmethod.data_loader import data_loader_fill
def main (args):
  '''Main function for UCI letter and spam datasets.
  
  Args:
    - data_name: letter or spam
    - miss_rate: probability of missing components
    - batch:size: batch size
    - hint_rate: hint rate
    - alpha: hyperparameter
    - iterations: iterations
    
  Returns:
    - imputed_data_x: imputed data
    - rmse: Root Mean Squared Error
  '''
  
  data_name = args.data_name
  miss_rate = args.miss_rate
  
  gain_parameters = {'batch_size': args.batch_size,
                     'hint_rate': args.hint_rate,
                     'alpha': args.alpha,
                     'iterations': args.iterations,
                     'noise_num': args.noise_num}
  
  # Load data and introduce missingness
  ori_data_x, miss_data_x, data_m = data_loader(data_name, miss_rate)

# ------------------TODO 以上是测试部分-------------------------------------------------------------------------------------
  # TODO 改动1和改动2，二选一，改动2不太稳定
  # TODO 改动2：整个数据集预填补后，直接聚类。*********************************** begain
  # ori_data_t = miss_data_x.copy()
  # KI = KNNImputer(n_neighbors=3, weights="uniform")
  # df_transformed = KI.fit_transform(ori_data_t)
  # data_c , data_class = KMPP(df_transformed, 3)
  # k = args.k
  # y =  data_class

  # TODO 改动2：整个数据集预填补后，直接聚类。*********************************** end


  # TODO 改动1：预训练 hmf ************************************ begain
  # 求出每一行，矩阵的和，最终得到一个，n*1的矩阵
  data_m_l= data_m.sum(axis=1)
  count = 0
  wz_index_list = []
  num_no = True
  num_i = 0
  account = args.account
  # 根据标记矩阵的行加和，取完整数据子集 TODO 完整样本并不太多，可考虑 放宽if条件，使用低缺失
  while num_no:
      for num in range(len(data_m_l)):
          # 如果完整样本少，可调节if的区间
          if data_m.shape[1] - num_i <= data_m_l[num] <= data_m.shape[1]:
              count = count + 1
              wz_index_list.append(num)

      print("完整样本数或低缺失：" + str(count))
      # todo 这里取大于 0.3样本总数，区间可调整，可设置参数 account
      if count > miss_data_x.shape[0] * account:
          num_no = False
      else:
          num_i = num_i + 1

  data_x_wz = np.zeros(shape=(count, data_m.shape[1]))
  # 均值预填补使用
  data_x_wz_m = np.zeros(shape=(count, data_m.shape[1]))
  print("最终：完整样本数或低缺失："+str(count))
  # wz_index_list = wz_index_list[:int(miss_data_x.shape[0] * account)]
  # print("最终使用：完整样本数或低缺失：" + str(len(wz_index_list)))
  # exit()
  # 循环获得完整样本集
  for li in range(len(wz_index_list)):
      data_x_wz[li] = miss_data_x[wz_index_list[li]]
      # 均值预填补使用
      data_x_wz_m[li] = data_m[wz_index_list[li]]

  # TODO KNN预填补 / 均值预填补  /AE 预填补  ******************* 部分样本预填补方法选择
  # KI = KNNImputer(n_neighbors=3, weights="uniform") # TODO KNN预填补
  # data_x_wz_t = KI.fit_transform(data_x_wz)
  # data_x_wz_t = mean_fill(data_x_wz, data_x_wz_m) #  # TODO 均值预填补
  # data_x_wz_t = ae_impute(data_x_wz, gain_parameters) #  # TODO AE预填补
  # data_x_wz_t = gain(data_x_wz, gain_parameters) #  # TODO GAIN预填补
  data_x_wz_t = gadin(data_x_wz, gain_parameters) #  # TODO GADIN预填补


  # TODO 这里K值需要写入参数，暂时固定用 3
  k = args.k
  # TODO 聚类方法,对结果有影响
  data_c , data_class = KM(data_x_wz_t, k)
  # TODO 分类器这里，可以选CART 或 SVM， 或不用分类器，预填补后，直接聚类
  # todo svm *********begain
  coder = preprocessing.OneHotEncoder()
  model = svm.SVC(kernel="linear", decision_function_shape="ovo")
  coder.fit(data_class.reshape(-1, 1))
  model.fit(data_x_wz_t, data_class)
  # todo 这里用0预填补
  ori_data_t = miss_data_x.copy()
  # 这里先用0代替缺失值，有时间测试均值 todo 与下面knn方法都是预填补方法，只用一个
  # ori_data_t[data_m == 0] = 0
  # df_transformed = ori_data_t
  # y = model.predict(ori_data_t)

  # todo 这里用KNN，预填补，样本过多，填补慢，可尝试均值填补
  # KI = KNNImputer(n_neighbors=3, weights="uniform") # TODO KNN预填补
  # df_transformed = KI.fit_transform(ori_data_t)
  # df_transformed = mean_fill(ori_data_t, data_m) # TODO 均值预填补
  # df_transformed = ae_impute(ori_data_t, gain_parameters) # TODO AE预填补
  # df_transformed = gain(ori_data_t, gain_parameters) # TODO GAIN预填补
  df_transformed = gadin(ori_data_t, gain_parameters)  # TODO GADIN预填补
  y = model.predict(df_transformed)
  # todo svm **********end

  # TODO 横向划分数据集
  # TODO 动态定义list
  # 记录每个数据子集下标[[子集1下标],[子集2下标],....]

  # *********************************************************************** TODO 改动1和改动2的分割线 TODO

  x_xb = []
  for i in range(k):
      x_xb.append([])

  # 每个子集下标集合
  for i in range(k):
      for j in range(len(y)):
          if y[j] == i:
              x_xb[i].append(j)

  # print(np.mat(x_xb[0]).T) todo list转矩阵并转置
  # 根据每个子集下标集合，划分x矩阵子集和，m矩阵子集，o矩阵子集
  # 先定义矩阵，用list包含
  x_x_list = []
  x_m_list = []
  x_o_list = []
  for xb in x_xb:
      x_x_list.append(np.zeros(shape=(len(xb), data_m.shape[1])))
      x_m_list.append(np.zeros(shape=(len(xb), data_m.shape[1])))
      x_o_list.append(np.zeros(shape=(len(xb), data_m.shape[1])))
  # 子集x,m,o矩阵填值
  for i in range(len(x_xb)):
      for j in range(len(x_xb[i])):
          x_x_list[i][j] = miss_data_x[x_xb[i][j]]
          x_m_list[i][j] = data_m[x_xb[i][j]]
          x_o_list[i][j] = ori_data_x[x_xb[i][j]]

  # 子集填补，计算误差值 TODO *********为测试方便，不用划分子集的方式，直接注释掉下面代码：分类效果对填补精度有影响
  # TODO 测试循环五次取平均值
  count_z = 0
  for it in range(2):
      for i in range(len(x_x_list)):
          # todo 使用GAIN
          # imputed_data_x = gain(x_x_list[i], gain_parameters)
          # todo 使用GADIN
          imputed_data_x = gadin(x_x_list[i], gain_parameters)
          # todo 使用AE
          # imputed_data_x = ae_impute(x_x_list[i], gain_parameters)

          # todo 使用MICE 这个太慢了,能不用就不用
          # imputed_data_x = mice(x_x_list[i])

          # Report the RMSE performance
          rmse = rmse_loss(x_o_list[i], imputed_data_x, x_m_list[i])

          # 数据子矩阵合并（横和纵都有合并）
          # print(np.mat(x_xb[0]).T) todo list转矩阵并转置
          # 合并行,列
          if i ==0:
              c = imputed_data_x
              c = np.concatenate((c, np.mat(x_xb[i]).T), axis=1)
          else:
              c = np.concatenate((c, np.concatenate((imputed_data_x, np.mat(x_xb[i]).T), axis=1)), axis=0)


          print()
          print('RMSE Performance: ' + str(np.round(rmse, 4)))

      # TODO 数据子集拼接回原来样本c,还需要按照下标排序，并且最后一列不参与计算
      # 矩阵排序 TODO 用for循环吧
      # x_i_z 多一列
      x_i_z = np.zeros(shape=(len(c), data_m.shape[1]+1))
      for i in range(len(c)):
          x_i_z[int(c[i, -1])] = c[i]

      print("合并后的总体误差：")
      rmse_z = rmse_loss(ori_data_x, x_i_z[:, :data_m.shape[1]], data_m)
      print(np.round(rmse_z, 4))
      count_z = count_z + rmse_z
  # TODO 缺失率对算法影响大，0.4以下 KNN好，以上本文好******************************
  print("DP 2次实验后的平均值：")
  print(np.round(count_z/2, 4))
  din_avg = np.round(count_z/2, 4)
  # print([ae_avg, in_avg, din_avg])

  return x_i_z, rmse_z

  exit()
  # TODO 改动1：预训练 hmf ************************************ end

  
  # TODO 改动1:之前 hmf ************************************ bgain
  # # Impute missing data
  # imputed_data_x = gain(miss_data_x, gain_parameters)
  #
  # # Report the RMSE performance
  # rmse = rmse_loss (ori_data_x, imputed_data_x, data_m)
  #
  # print()
  # print('RMSE Performance: ' + str(np.round(rmse, 4)))
  #
  # return imputed_data_x, rmse
  # TODO 改动1:之前 hmf ************************************ end

if __name__ == '__main__':  
  
  # Inputs for the main function
  # TODO spam 垃圾邮件 4601-57（48连续，） 标签2
  #  TODO letter 字母识别 20000-16
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data_name',
      choices=['breast', 'abalone', 'news', 'spam', 'letter', 'htru'],
      default='news',
      type=str)
  # todo 新增的 噪声位置个数, GADIN用到，
  parser.add_argument(
      '--noise_num',
      help='the number of noise location 取值 2 or 5 or others',
      default=5, # TODO ****注意：调整预填补样本数量*****
      type=int)
  # TODO 这里的缺失率指定；应该是：整体部分，而不是每一条占总条数的缺失
  # TODO 缺失率对算法影响大，0.4以下 KNN好，以上本文好*****************
  parser.add_argument(
      '--miss_rate',
      help='missing data probability',
      default=0.5,
      type=float)
  # TODO 尝试改变 样本数据对结果的影响：一般来说，小样本好些
  parser.add_argument(
      '--batch_size',
      help='the number of samples in mini-batch',
      default=68,
      type=int)
  parser.add_argument(
      '--hint_rate',
      help='hint probability',
      default=0.9,
      type=float)
  parser.add_argument(
      '--alpha',
      help='hyperparameter',
      default=100,
      type=float)
  # TODO 这里改为1000，不影响结果 hmf 训练次数多了不一定好
  parser.add_argument(
      '--iterations',
      help='number of training interations',
      default=10,
      type=int)

  # TODO 新增的 hmf ：如果样本过少，适当调节 batch_size or 聚类个数
  parser.add_argument(
      '--k',
      help='分类个数： number of Classification',
      default=2,
      type=int)
  # TODO 新增的 hmf ： 预填补样本个数，占总体比例
  parser.add_argument(
      '--account',
      help='预填补所占比例： number of Pre-xunlian',
      default=0.3,
      type=int)
  
  args = parser.parse_args() 

  # Calls main function  
  imputed_data, rmse = main(args)


# todo 训练模型，获得RMSE值
def run_model(df_x, it_number, mi_rate):

    miss_rate = mi_rate

    gain_parameters = {'batch_size': 24,
                       'hint_rate': 0.9,
                       'alpha': 100,
                       'iterations': it_number,
                       'noise_num': 2}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader_new(df_x, miss_rate)

    # ------------------TODO 以上是测试部分-------------------------------------------------------------------------------------
    # TODO 改动1和改动2，二选一，改动2不太稳定
    # TODO 改动2：整个数据集预填补后，直接聚类。*********************************** begain
    # ori_data_t = miss_data_x.copy()
    # KI = KNNImputer(n_neighbors=3, weights="uniform")
    # df_transformed = KI.fit_transform(ori_data_t)
    # data_c , data_class = KMPP(df_transformed, 3)
    # k = args.k
    # y =  data_class

    # TODO 改动2：整个数据集预填补后，直接聚类。*********************************** end

    # TODO 改动1：预训练 hmf ************************************ begain
    # 求出每一行，矩阵的和，最终得到一个，n*1的矩阵
    data_m_l = data_m.sum(axis=1)
    count = 0
    wz_index_list = []
    num_no = True
    num_i = 0
    # account = args.account
    account = 0.3
    # 根据标记矩阵的行加和，取完整数据子集 TODO 完整样本并不太多，可考虑 放宽if条件，使用低缺失
    while num_no:
        for num in range(len(data_m_l)):
            # 如果完整样本少，可调节if的区间
            if data_m.shape[1] - num_i <= data_m_l[num] <= data_m.shape[1]:
                count = count + 1
                wz_index_list.append(num)

        print("完整样本数或低缺失：" + str(count))
        # todo 这里取大于 0.3样本总数，区间可调整，可设置参数 account
        if count > miss_data_x.shape[0] * account:
            num_no = False
        else:
            num_i = num_i + 1

    data_x_wz = np.zeros(shape=(count, data_m.shape[1]))
    # 均值预填补使用
    data_x_wz_m = np.zeros(shape=(count, data_m.shape[1]))
    print("最终：完整样本数或低缺失：" + str(count))
    # wz_index_list = wz_index_list[:int(miss_data_x.shape[0] * account)]
    # print("最终使用：完整样本数或低缺失：" + str(len(wz_index_list)))
    # exit()
    # 循环获得完整样本集
    for li in range(len(wz_index_list)):
        data_x_wz[li] = miss_data_x[wz_index_list[li]]
        # 均值预填补使用
        data_x_wz_m[li] = data_m[wz_index_list[li]]

    # TODO KNN预填补 / 均值预填补  /AE 预填补  ******************* 部分样本预填补方法选择
    # KI = KNNImputer(n_neighbors=3, weights="uniform") # TODO KNN预填补
    # data_x_wz_t = KI.fit_transform(data_x_wz)
    # data_x_wz_t = mean_fill(data_x_wz, data_x_wz_m) #  # TODO 均值预填补
    # data_x_wz_t = ae_impute(data_x_wz, gain_parameters) #  # TODO AE预填补
    # data_x_wz_t = gain(data_x_wz, gain_parameters) #  # TODO GAIN预填补
    data_x_wz_t = gadin(data_x_wz, gain_parameters)  # # TODO GADIN预填补

    # TODO 这里K值需要写入参数，暂时固定用 3
    # k = args.k
    k= 2
    # TODO 聚类方法,对结果有影响
    data_c, data_class = KM(data_x_wz_t, k)
    # TODO 分类器这里，可以选CART 或 SVM， 或不用分类器，预填补后，直接聚类
    # todo svm *********begain
    coder = preprocessing.OneHotEncoder()
    model = svm.SVC(kernel="linear", decision_function_shape="ovo")
    coder.fit(data_class.reshape(-1, 1))
    model.fit(data_x_wz_t, data_class)
    # todo 这里用0预填补
    ori_data_t = miss_data_x.copy()
    # 这里先用0代替缺失值，有时间测试均值 todo 与下面knn方法都是预填补方法，只用一个
    # ori_data_t[data_m == 0] = 0
    # df_transformed = ori_data_t
    # y = model.predict(ori_data_t)

    # todo 这里用KNN，预填补，样本过多，填补慢，可尝试均值填补
    # KI = KNNImputer(n_neighbors=3, weights="uniform") # TODO KNN预填补
    # df_transformed = KI.fit_transform(ori_data_t)
    # df_transformed = mean_fill(ori_data_t, data_m) # TODO 均值预填补
    # df_transformed = ae_impute(ori_data_t, gain_parameters) # TODO AE预填补
    # df_transformed = gain(ori_data_t, gain_parameters) # TODO GAIN预填补
    df_transformed = gadin(ori_data_t, gain_parameters)  # TODO GADIN预填补
    y = model.predict(df_transformed)
    # todo svm **********end

    # TODO 横向划分数据集
    # TODO 动态定义list
    # 记录每个数据子集下标[[子集1下标],[子集2下标],....]

    # *********************************************************************** TODO 改动1和改动2的分割线 TODO

    x_xb = []
    for i in range(k):
        x_xb.append([])

    # 每个子集下标集合
    for i in range(k):
        for j in range(len(y)):
            if y[j] == i:
                x_xb[i].append(j)

    # print(np.mat(x_xb[0]).T) todo list转矩阵并转置
    # 根据每个子集下标集合，划分x矩阵子集和，m矩阵子集，o矩阵子集
    # 先定义矩阵，用list包含
    x_x_list = []
    x_m_list = []
    x_o_list = []
    for xb in x_xb:
        x_x_list.append(np.zeros(shape=(len(xb), data_m.shape[1])))
        x_m_list.append(np.zeros(shape=(len(xb), data_m.shape[1])))
        x_o_list.append(np.zeros(shape=(len(xb), data_m.shape[1])))
    # 子集x,m,o矩阵填值
    for i in range(len(x_xb)):
        for j in range(len(x_xb[i])):
            x_x_list[i][j] = miss_data_x[x_xb[i][j]]
            x_m_list[i][j] = data_m[x_xb[i][j]]
            x_o_list[i][j] = ori_data_x[x_xb[i][j]]

    # 子集填补，计算误差值 TODO *********为测试方便，不用划分子集的方式，直接注释掉下面代码：分类效果对填补精度有影响
    # TODO 测试循环五次取平均值,可调整
    count_z = 0
    for it in range(1):
        for i in range(len(x_x_list)):
            # todo 使用GAIN
            # imputed_data_x = gain(x_x_list[i], gain_parameters)
            # todo 使用GADIN
            imputed_data_x, rmse_list, theta_G = gadin_model(x_x_list[i], gain_parameters)
            # todo 使用AE
            # imputed_data_x = ae_impute(x_x_list[i], gain_parameters)

            # todo 使用MICE 这个太慢了,能不用就不用
            # imputed_data_x = mice(x_x_list[i])

            # Report the RMSE performance
            rmse = rmse_loss(x_o_list[i], imputed_data_x, x_m_list[i])

            # 数据子矩阵合并（横和纵都有合并）
            # print(np.mat(x_xb[0]).T) todo list转矩阵并转置
            # 合并行,列
            if i == 0:
                c = imputed_data_x
                c = np.concatenate((c, np.mat(x_xb[i]).T), axis=1)
            else:
                c = np.concatenate((c, np.concatenate((imputed_data_x, np.mat(x_xb[i]).T), axis=1)), axis=0)

            print()
            rmse = str(np.round(rmse, 4))
            print('RMSE Performance: ' + rmse)

        # TODO 数据子集拼接回原来样本c,还需要按照下标排序，并且最后一列不参与计算
        # 矩阵排序 TODO 用for循环吧
        # x_i_z 多一列
        x_i_z = np.zeros(shape=(len(c), data_m.shape[1] + 1))
        for i in range(len(c)):
            x_i_z[int(c[i, -1])] = c[i]

        print("合并后的总体误差：")
        rmse_z = rmse_loss(ori_data_x, x_i_z[:, :data_m.shape[1]], data_m)
        print(np.round(rmse_z, 4))
        count_z = count_z + rmse_z
    # TODO 缺失率对算法影响大，0.4以下 KNN好，以上本文好******************************
    print("DP 2次实验后的平均值：")
    print(np.round(count_z / 2, 4))
    din_avg = np.round(count_z / 2, 4)
    rmse_z = np.round(rmse_z, 4)
    # print([ae_avg, in_avg, din_avg])

    return x_i_z, rmse, rmse_list, theta_G


# todo 填补数据，获得RMSE值
def datafill(df_x, it_number):

    gain_parameters = {'batch_size': 24,
                       'hint_rate': 0.9,
                       'alpha': 100,
                       'iterations': it_number,
                       'noise_num': 2}

    # Load data and introduce missingness
    ori_data_x, miss_data_x, data_m = data_loader_fill(df_x)

    imputed_data_x, rmse_list,_ = gadin_model(miss_data_x, gain_parameters)
    # imputed_data_x, rmse_list = ae_impute_fill(miss_data_x, gain_parameters)
    # imputed_data_x, rmse_list = gain_fill(miss_data_x, gain_parameters)

    # Report the RMSE performance
    rmse_round = str(np.round(rmse_list[-1], 4))
    print('RMSE Performance: ' + rmse_round)

    norm_data_x = np.nan_to_num(miss_data_x, 0)
    imputed_data_x = data_m * norm_data_x + (1 - data_m) * imputed_data_x

    return imputed_data_x, rmse_round
