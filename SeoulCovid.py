# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler

raw_df = pd.read_csv("./covid_seoul.csv", encoding='cp949')

scaler = MinMaxScaler()

feature_cols = ['서울시 추가 확진', '1차접종 누계', '2차접종 누계', '3차접종 누계', '4차접종 누계',
              '동절기접종 누계', '평균기온(°C)', '평균 상대습도(%)', '평균 지면온도(°C)', '평균 30cm 지중온도(°C)',
              '5.0m 지중온도(°C)', '지하철 승차 인원', '일 미세먼지 농도(㎍/㎥)'] 
label_cols = ['서울시 추가 확진'] 

scaled_df = scaler.fit_transform(raw_df[feature_cols]) # numpy
scaled_df = pd.DataFrame(scaled_df, columns=feature_cols)

s1 = scaler.fit_transform(raw_df[label_cols])

label_df = pd.DataFrame(scaled_df, columns=label_cols)
feature_df = pd.DataFrame(scaled_df, columns=feature_cols)

label_np = label_df.to_numpy() # input type = Numpy
feature_np = feature_df.to_numpy() # input type = Numpy

def make_sequence_dataset(feature, label, window_size):
  feature_list = []
  label_list = []

  for i in range(len(feature) - window_size): # window size 만큼 묶은 feature - label 매칭.
    feature_list.append(feature[i:i+window_size])
    label_list.append(label[i+window_size])

  return np.array(feature_list), np.array(label_list)

window_size = 14 # 최대 잠복기:14 / 평균잠복기: 5

X, Y = make_sequence_dataset(feature_np, label_np, window_size)

print(X.shape, Y.shape) # 3차원 텐서

split = -50 # 마지막 50일 예측

x_train = X[0:split]
y_train = Y[0:split]

x_test = X[split:]
y_test = Y[split:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

X = tf.keras.layers.Input(shape = x_train[0].shape)

H = tf.keras.layers.LSTM(64, return_sequences=True)(X)
H = tf.keras.layers.LSTM(50)(H)

Y = tf.keras.layers.Dense(1)(H)

model = tf.keras.models.Model(X, Y)

model.summary()

adam = tf.keras.optimizers.Adam(learning_rate=0.001)
rmse = tf.keras.metrics.RootMeanSquaredError()

model.compile(loss='mse', optimizer = adam, metrics=rmse)
hist = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=90, batch_size=16)

# prediction
pred = model.predict(x_test)

ps = []
ys = []

ps = scaler.inverse_transform(pred)
ys = scaler.inverse_transform(y_test)

for i in range(len(ps)):
   if ps[i] < 0:
      ps[i] = 0

# 마지막 일 예측
print("prediction: %f" %ps[-1])
print("true: %f" %ys[[-1]])


# 학습 과정 저장
file = open("SeoulCovidHistory.txt", "a")

sum = 0
for i in hist.history['loss']:
    sum += i
avg_loss = sum / len(hist.history['loss'])
file.write("avg_loss: " + str(avg_loss) + "\n")

sum = 0
for i in hist.history['val_loss']:
    sum += i

avg_val_loss = sum / len(hist.history['val_loss'])
file.write("avg_val_loss: " + str(avg_val_loss) + "\n")

sum = 0
for i in hist.history['root_mean_squared_error']:
    sum += i

avg_rmse = sum / len(hist.history['root_mean_squared_error'])
file.write("avg_rmse: " + str(avg_rmse) + "\n")

sum = 0
for i in hist.history['val_root_mean_squared_error']:
    sum += i

avg_val_rmse = sum / len(hist.history['val_root_mean_squared_error'])
file.write("avg_val_rmse: " + str(avg_val_rmse) + "\n")

file.write("prediction: " + str(ps[-1]) + "\n")
file.write("true: " + str(ys[[-1]]) + '\n' + '\n')

file.close()

# Plot
fig = plt.figure(facecolor='white', figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(ys, label='True')
ax.plot(ps, label='Prediction')
ax.legend()
plt.show()