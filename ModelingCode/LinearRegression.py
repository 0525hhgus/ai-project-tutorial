from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

import numpy
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

# k겹 교차 검증 추가
'''
seed = 0
numpy.random.seed(seed)
tf.random.set_seed(3)
'''
df = pd.read_csv("pjdata_pre_cg2_nh.csv", names=['Tem', 'Wind', 'Hum', 'Pres',
                                                       'PM10', 'PM2.5', 'O3', 'NO2',
                                                'CO', 'SO2', 'Pat'], header=None)
# row = 649

x1 = df[['Tem', 'Wind', 'Hum', 'Pres', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2']]
y1 = df[['Pat']]
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, train_size=500, test_size=149)

model = Sequential()
#model.add(Dense(25, input_dim=9, activation='softmax'))
model.add(Dense(25, input_dim=10, activation='softplus'))
model.add(Dense(40, activation='softplus'))
model.add(Dense(5, activation='softplus'))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(x1_train, y1_train, epochs=300, batch_size=5)

#Y_predict = model.predict(x1_test).flatten()
Y_predict = model.predict(x1_test)
#print(Y_predict)
'''
for i in range(10):
    label = y1_test[i]
    prediction = Y_predict
    print("실제 : %.4f" % label)
    print("예상 : %.4f" % prediction)
'''

print("\n acc : %.4f" % (model.evaluate(x1, y1)[1]))
print("\n test acc : %.4f" % (model.evaluate(x1_test, y1_test)[1]))
#print(model.evaluate(x1, y1)[1])


# 실제 데이터와 예측 데이터 비교 그래프 (직선 : y=x 그래프)
print("선형 회귀 그래프")
plt.scatter(y1_test, Y_predict, alpha=0.4)
plt.plot(y1_test, y1_test, color='red')
plt.xlabel("Actual Data")
plt.ylabel("Predicted Data")
plt.title("LINEAR REGRESSION")
plt.show()

result = metrics.confusion_matrix(y1_test, Y_predict)
sns.heatmap(pd.DataFrame(result), annot=True, cmap='RdYlGn', fmt='g')
plt.xlabel("Actual Data")
plt.ylabel("Predicted Data")
plt.title("LOGISTIC REGRESSION")
plt.show()

