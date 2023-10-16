import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Baca dataset dari file CSV
data = pd.read_csv('./breast-cancer.csv')

# Preprocessing data
X = data.iloc[:, 2:].values  # Mengambil fitur-fitur
y = data['diagnosis'].values

# Label encoding untuk target (M dan B)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model JST
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pelatihan model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluasi model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')
