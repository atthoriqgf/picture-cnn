import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model
model_url = "https://huggingface.co/atthoriqgf112/picture-cnn/resolve/main/model_cnn.h5"
model_path = "model_cnn.h5"

# Unduh file jika belum ada
if not os.path.exists(model_path):
    with requests.get(model_url, stream=True) as r:
        r.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# Memuat model CNN yang sudah dilatih
model = load_model(model_path)

st.title("Klasifikasi Gambar Kucing vs Anjing 😺🐶")
st.write("Aplikasi ini akan menerima gambar dan menampilkan hasil prediksi apakah ini kucing atau anjing.")

# Input file gambar
uploaded_file = st.file_uploader("Silakan upload gambar", type=["jpg", "jpeg", "png"])

# Jika ada gambar yang diupload
if uploaded_file is not None:
    # Tampilkan gambar
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang di-upload", use_column_width=True)
    
    # Preprocess gambar
    img = img.resize((150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)

    # Interpretasi
    if prediction[0] > 0.5:
        st.subheader("Prediksi: Anjing 🐶")
    else:
        st.subheader("Prediksi: Kucing 😺")

    # Tombol prediksi (sementara masih dummy)
    # if st.button("Prediksi"):
    #     st.success("Hasil prediksi: Ini adalah gambar **Kucing** 🐱 dengan akurasi 95%")
