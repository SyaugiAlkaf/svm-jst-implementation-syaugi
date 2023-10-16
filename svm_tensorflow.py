from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# Baca dataset dari file CSV
data = pd.read_csv('./breast-cancer.csv')

# Memisahkan fitur dan target
X = data.iloc[:, 2:].values  # Mengambil fitur-fitur dari kolom ke-2 hingga akhir
y = data.iloc[:, 1].values    # Kolom ke-1 adalah target (label diagnosis)

# Label encoding untuk target (M dan B)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Normalisasi fitur
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data menjadi data pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membangun model SVM
svm_model = SVC(kernel='linear', C=1.0)

# Pelatihan model
svm_model.fit(X_train, y_train)

# Prediksi data pengujian
y_pred = svm_model.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Laporan klasifikasi
print(classification_report(y_test, y_pred))
