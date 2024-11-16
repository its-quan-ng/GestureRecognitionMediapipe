# train_lstm.py
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.callbacks import history
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split

# Load gesture datasets
datasets_dict = {
    'ONE': pd.read_csv('ONE.txt'),
    'TWO': pd.read_csv('TWO.txt'),
    'TWO_inverse': pd.read_csv('TWO_inverse.txt'),
    'THREE': pd.read_csv('THREE.txt'),
    'THREE_2': pd.read_csv('THREE_2.txt'),
    'THREE_3': pd.read_csv('THREE_3.txt'),
    'FOUR': pd.read_csv('FOUR.txt'),
    'PALM': pd.read_csv('PALM.txt'),
    'PALM_inverse': pd.read_csv('PALM.txt'),
    'STOP': pd.read_csv('STOP.txt'),
    'OK': pd.read_csv('OK.txt'),
    'CALL': pd.read_csv('CALL.txt'),
    'LIKE': pd.read_csv('LIKE.txt'),
    'DISLIKE': pd.read_csv('DISLIKE.txt'),
    'FIST': pd.read_csv('FIST.txt'),
    'MUTE': pd.read_csv('MUTE.txt'),
    'PEACE': pd.read_csv('PEACE.txt'),
    'PEACE_inverse': pd.read_csv('PEACE_inverse.txt'),
    'ROCK': pd.read_csv('ROCK.txt'),
    'GUN': pd.read_csv('GUN.txt'),
    'MINIHEART': pd.read_csv('MINIHEART.txt')
}

# Debug: Print shapes of all datasets
print("\nChecking dataset shapes:")
for name, df in datasets_dict.items():
    print(f"{name}: Shape={df.iloc[:, 1:].shape}, Columns={df.iloc[:, 1:].shape[1]}")


X = []
y = []
no_of_timestep = 10

# Create dataset pairs with numerical labels
datasets = [
    (datasets_dict['ONE'], 0),
    (datasets_dict['TWO'], 1),
    (datasets_dict['TWO_inverse'], 1),  # Same as TWO
    (datasets_dict['THREE'], 2),
    (datasets_dict['THREE_2'], 2),  # Same as THREE
    (datasets_dict['THREE_3'], 2),  # Same as THREE
    (datasets_dict['FOUR'], 3),
    (datasets_dict['PALM'], 4),
    (datasets_dict['PALM_inverse'], 4),
    (datasets_dict['STOP'], 5),
    (datasets_dict['OK'], 6),
    (datasets_dict['CALL'], 7),
    (datasets_dict['LIKE'], 8),
    (datasets_dict['DISLIKE'], 9),
    (datasets_dict['FIST'], 10),
    (datasets_dict['MUTE'], 11),
    (datasets_dict['PEACE'], 12),
    (datasets_dict['PEACE_inverse'], 12), # Same as PEACE
    (datasets_dict['ROCK'], 13),
    (datasets_dict['GUN'], 14),
    (datasets_dict['MINIHEART'], 15)
]


for dataset, label in datasets:
    data = dataset.iloc[:, 1:].values
    n_samples = len(data)
    for i in range(no_of_timestep, n_samples):
        X.append(data[i-no_of_timestep:i, :])
        y.append(label)

X, y = np.array(X), np.array(y)
print(X.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(LSTM(units=64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=64))
model.add(Dropout(0.2))
model.add(Dense(units=16, activation="softmax"))  # 11 classes (0-10)

model.compile(optimizer="adam", metrics=['accuracy'],
             loss="sparse_categorical_crossentropy")

model.fit(X_train, y_train, epochs=20, batch_size=32,
          validation_data=(X_test, y_test))
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Độ chính xác trên tập kiểm tra: {accuracy:.3f}")
print(f"Loss trên tập kiểm tra: {loss:.4f}")


model.save('hand_gesture_model.h5')