import streamlit as st
from PIL import Image

st.title("🖼️ Upload Gambar untuk Prediksi AI")
st.write("Aplikasi ini akan menerima gambar dan menampilkan hasil prediksi (dummy).")

# Input file gambar
uploaded_file = st.file_uploader("Silakan upload gambar", type=["jpg", "jpeg", "png"])

# Jika ada gambar yang diupload
if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file)
    st.image(image, caption="Gambar yang di-upload", use_column_width=True)

    # Tombol prediksi (sementara masih dummy)
    if st.button("Prediksi"):
        st.success("Hasil prediksi: Ini adalah gambar **Kucing** 🐱 dengan akurasi 95%")
