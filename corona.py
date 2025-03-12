# import
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# load dataset
data = pd.read_csv('./covid1.csv')
mid_prices = data['계(명)'].values

data.head()
len(mid_prices)

seq_len = 50
sequence_length = seq_len + 1

result = []
for index in range(len(mid_prices) - sequence_length):
    result.append(mid_prices[index : index + sequence_length])

# normalize Data (MinMaxScaler)
normalized_data =  []
for window in result:
    normalized_window = [(p - min(mid_prices)) / (max(mid_prices) - min(mid_prices)) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

# split train and test data
row = int(round(result.shape[0] * 0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

x_train.shape, x_test.shape

# building a model
# Input Layer
X = tf.keras.layers.Input(shape = [50, 1])

# Hidden Layer (LSTM)
# H = tf.keras.layers.LSTM(50, return_sequences=True)(X)
# H = tf.keras.layers.Dropout(0.2)(H)

# H = tf.keras.layers.LSTM(64)(H)
# H = tf.keras.layers.Dropout(0.2)(H)

H = tf.keras.layers.LSTM(128)(X)

# Output Layer
Y = tf.keras.layers.Dense(1)(H)

model = tf.keras.models.Model(X, Y)

model.compile(loss='mae', optimizer='adam')

model.summary()

# trianing
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=150) #xtrain 독립, ytrain 종속

# prediction
pred = model.predict(x_test)

ps = []
ys = []

for i in range(len(pred)):
  p = pred[i] * (max(mid_prices) - min(mid_prices)) + min(mid_prices)
  ps.append(p)

for i in range(len(y_test)):
  y = y_test[i] * (max(mid_prices) - min(mid_prices)) + min(mid_prices)
  ys.append(y)

print("prediction: %f" %p)
print("true: %f" %y)

fig = plt.figure(facecolor='white', figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(ys, label='True')
ax.plot(ps, label='Prediction')
ax.legend()
plt.show()