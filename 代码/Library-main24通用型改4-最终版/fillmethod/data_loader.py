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

'''Data loader for UCI letter, spam and MNIST datasets.
'''

# Necessary packages
import numpy as np
from fillmethod.utils import binary_sampler

# TODO 包的原因，不能测试mnist数据集
# from keras.datasets import mnist


def data_loader (data_name, miss_rate):
  '''Loads datasets and introduce missingness.
  
  Args:
    - data_name: letter, spam, or mnist
    - miss_rate: the probability of missing components
    
  Returns:
    data_x: original data
    miss_data_x: data with missing values
    data_m: indicator matrix for missing components
  '''
  
  # Load data
  if data_name in ['letter', 'spam', 'breast', 'news', 'abalone', 'balance', 'htru', 'cmb']:
    file_name = r'F:\pyproject\GAIN-2018-改动版\GAIN-master\data\\'+data_name+'.csv'
    data_x = np.loadtxt(file_name, delimiter=",", skiprows=1)
  elif data_name == 'mnist':
    # TODO 包的原因，不能测试mnist数据集
    print("版本原因")
    exit()
    # (data_x, _), _ = mnist.load_data()
    # data_x = np.reshape(np.asarray(data_x), [60000, 28*28]).astype(float)

  # Parameters
  no, dim = data_x.shape
  
  # Introduce missing data  TODO 随机缺失的制造
  data_m = binary_sampler(1-miss_rate, no, dim)
  miss_data_x = data_x.copy()
  # TODO 这种赋值方式很简单
  miss_data_x[data_m == 0] = np.nan

      
  return data_x, miss_data_x, data_m


def data_loader_new(data_x, miss_rate):

  # Parameters
  no, dim = data_x.shape

  # Introduce missing data  TODO 随机缺失的制造
  data_m = binary_sampler(1 - miss_rate, no, dim)
  miss_data_x = data_x.copy()
  # TODO 这种赋值方式很简单
  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m

def data_loader_fill(data_x):

  # Define mask matrix TODO 定义 M掩码矩阵
  data_m = 1 - np.isnan(data_x)
  miss_data_x = data_x.copy()
  # TODO 这种赋值方式很简单
  miss_data_x[data_m == 0] = np.nan

  return data_x, miss_data_x, data_m