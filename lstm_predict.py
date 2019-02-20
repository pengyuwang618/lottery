import pandas as pd
from numpy import array
from numpy import hstack
import numpy as np 
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn import preprocessing 


# split a multivariate sequence into samples
# 设计步长为20的时间序列
n_steps = 20
def create_dataset(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#读取福彩3D的中奖纪录
data = pd.read_csv('d:\\quantrading\\3D_record.csv')
data['date'] = pd.to_datetime(data['date'])
#选取其中1000期中奖纪录作为样本
data=data[4080:5080]

x=np.array(data['x'])
y=np.array(data['y'])
z=np.array(data['z'])    
#x,y,z三列数据通过格式化
in_seq1 = x.reshape((len(x), 1))
in_seq2 = y.reshape((len(y), 1))
in_seq3 = z.reshape((len(z), 1))

#合并三个单独的序列进入dataset
dataset = hstack((in_seq1, in_seq2, in_seq3))
#对序列数据进行标准化操作
scaler = preprocessing.MinMaxScaler() 
dataset = scaler.fit_transform(dataset.astype('float32')) 
# 对全体数据进行4:1比例的切分，80%数据用作训练样本，20%用做测试样本
train_size = int(len(dataset) * 0.8)
validation_size = len(dataset) - train_size
#获取训练数据集train和测试数据集合validation
train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]

# 将监督数据通过create_dataset()转换为序列数据
X_train, y_train = create_dataset(train,n_steps)
X_validation, y_validation = create_dataset(validation,n_steps)


# configure network
# on line training (batch_size=1)
n_batch = 1
# 训练次数设定为700
n_epoch = 700
# 神经单元数量为40
n_neurons = 40
# design network
# 假设彩票中奖纪录在时序上存在一定的函数映射关系，可以根据历史中奖纪录预测未来中奖纪录，因此模型设计为时间序列模型
# 设计为多变量(每一个变量对应一个中奖数字)单步预测输出的LSTM网络
# LSTM可以保持状态(state)，但是Stateless LSTM 只能保持一个批次内的状态, 为了记录长时序的状态，程序设计为stateful 
# 由于Keras的stateful 的state传递规则为本批次的第i个样本状态传递给下批次的第i个样本的初始状态，因此程序设计为on line training(batch_size=1)
model = Sequential()
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2]), stateful=True, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2]), stateful=True))
model.add(Dropout(0.2))
#Dense(3)的结果是预测模型输出下一期的三位福彩3D中奖数字
model.add(Dense(3))
#损失函数为均方差，优化算法是adam梯度下降
model.compile(loss='mean_squared_error', optimizer='adam')

# fit 操作：根据stateful LSTM的训练规则，shuffle手工设置为关闭，state在每次epoch结束后重置。
for i in range(n_epoch):
    model.fit(X_train, y_train, epochs=1, batch_size=n_batch, verbose=2, shuffle=False)
    model.reset_states()

# 预测1：采用样本训练数据预测，从直观观察结果，中奖率极高，这一结果体现了神经网络极强的非线性拟合能力。
print('training dataset predict')
for i in range(X_train.shape[0]):
    testX = X_train[i]    
    testX = testX.reshape(1, X_train.shape[1], X_train.shape[2])
    yhat = model.predict(testX, batch_size=n_batch)
    testy = scaler.inverse_transform([y_train[i]])
    testy = testy.round()
    yhat = scaler.inverse_transform(yhat)
    yhat = yhat.round()
    print('expected : ',testy,end=' ') 
    print('predict :',yhat.squeeze())
# 预测2：采用样本测试数据预测，从直观观察结果，一个中奖的也没有，这一结果体现了随机游走(random walk)的特点，一枚均匀的硬币的投掷正反面无法预测。
# 有监督学习的基本原理是通过构造由预测值与样本值差值的损失函数，基于优化算法，拟合出自变量和输出量之间的函数映射，对于随机游走数据，由于不存在映射函数，因此对于测试数据的结果，属于正常表现。
print('test dataset predict')
for i in range(X_validation.shape[0]):
    testX = X_validation[i]    
    testX = testX.reshape(1, X_validation.shape[1], X_validation.shape[2])
    yhat = model.predict(testX, batch_size=n_batch)
    testy = scaler.inverse_transform([y_validation[i]])
    testy = testy.round()
    yhat = scaler.inverse_transform(yhat)
    yhat = yhat.round()
    print('expected : ',testy,end=' ') 
    print('predict :',yhat.squeeze())
