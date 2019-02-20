from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn import preprocessing 
import pandas as pd
import numpy as np 

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
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

# 读取福彩3D的中奖纪录
data = pd.read_csv('d:\\quantrading\\3D_record.csv')
data['date'] = pd.to_datetime(data['date'])
# 选取其中1000期中奖纪录作为样本
data=data[4080:5080]

#x,y,z三列数据通过格式化
x=np.array(data['x'])
y=np.array(data['y'])
z=np.array(data['z'])    

in_seq1 = x.reshape((len(x), 1))
in_seq2 = y.reshape((len(y), 1))
in_seq3 = z.reshape((len(z), 1))

# 合并三个单独的序列进入dataset
dataset = hstack((in_seq1, in_seq2, in_seq3))
# 对全体数据进行4:1比例的切分，80%数据用作训练样本，20%用做测试样本
train_size = int(len(dataset) * 0.8)
validation_size = len(dataset) - train_size
#获取训练数据集train和测试数据集合validation
train, validation = dataset[0: train_size, :], dataset[train_size: len(dataset), :]
# 中奖序列数据步长设置为20
n_steps = 20
# 将数据进行拆分，生成训练数据集和测试数据集
X_train, y_train = split_sequences(train,n_steps)
X_validation, y_validation = split_sequences(validation,n_steps)

n_features = X_train.shape[2]
# 定义卷积神经网络模型
# 卷积类型有一维卷积(应用于NLP)、二维卷积(应用于图片处理)、三维卷积(应用于视频处理)
# 此处采用一维卷积Conv1D处理彩票中奖序列数据
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(n_steps, n_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
# Dense(n_features)的结果是预测模型输出下一期的三位福彩3D中奖数字
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
# 模型训练2400次
model.fit(X_train, y_train, epochs=2400, verbose=2)

# 预测1：采用样本训练数据预测，从直观观察结果，中奖率极高，这一结果体现了神经网络极强的非线性拟合能力。
print('predict training dataset')
for i in range(X_train.shape[0]):
    testX = X_train[i]    
    testX = testX.reshape(1, n_steps, n_features)
    yhat = model.predict(testX, verbose=0)
    yhat = yhat.round()
    testy = y_train[i]
    testy = testy.round()   
    print('training expected : ',testy,end=' ') 
    print('training predict :',yhat.squeeze())
# 预测2：采用样本测试数据预测，从直观观察结果，一个中奖的也没有，这一结果体现了随机游走(random walk)的特点，一枚均匀的硬币的投掷正反面无法预测。
print('predict test dataset')
for i in range(X_validation.shape[0]):
    testX = X_validation[i]    
    testX = testX.reshape(1, n_steps, n_features)
    yhat = model.predict(testX, verbose=0)
    yhat = yhat.round()
    testy = y_validation[i]
    testy = testy.round()
    print('expected : ',testy,end=' ') 
    print('predict :',yhat.squeeze())
