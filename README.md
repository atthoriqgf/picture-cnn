# 🐶😺 Picture Classifier: Dog vs Cat using CNN & Streamlit

Proyek ini adalah aplikasi klasifikasi gambar yang mampu membedakan antara gambar **anjing** dan **kucing** menggunakan **Convolutional Neural Network (CNN)**. Model dilatih menggunakan dataset gambar dan di-deploy sebagai aplikasi web menggunakan **Streamlit**.

---

## 🔧 Fitur

- Upload gambar:
  - JPG
  - JPEG (Not tested)
  - PNG (Not tested)
- Prediksi apakah gambar tersebut adalah **kucing** atau **anjing**
- Tampilan interaktif berbasis Streamlit

---

## 📂 Struktur Folder
```yaml
picture-cnn/
│
├── dataset/ # Berisi data training dan testing
│ ├── train/
│ └── test/
│
├── train_model.py # Script untuk melatih dan menyimpan model CNN
├── app.py # Aplikasi web Streamlit
├── .gitignore
├── requirements.txt # Daftar dependensi Python
└── README.md # Dokumentasi proyek
```

___

## 🤔 How to install this?
### 1. Buat dan Aktifkan Virtual Environment
#### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Depedensi
```bash
pip install -r requirements.txt
```

### 3. Latih Model (Jika belum ada `model_cnn.h5`)
```bash
python train_model.py
```

### 4. Jalankan aplikasi Streamlit
```bash
streamlit run app.py
```

___

## 💻 Teknologi yang digunakan
- Python 3.10 (penting karena untuk support TensorFlow / Keras)
- TensorFlow / Keras
- Streamlit
- Streamlit Cloud
- NumPy
- Pillow (PIL)
