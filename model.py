from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
# Створюємо модель
model = Sequential()
# Додаємо шар з одним нейроном
model.add(Dense(1, input_shape=(1,)))
# Компилируем модель перед использованием
model.compile(optimizer='adam', loss='mean_squared_error')
# Огляд структури моделі
model.summary()

