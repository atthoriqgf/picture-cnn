from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Memuat model CNN yang sudah dilatih
model = load_model('model_cnn.h5')

# Path ke gambar yang ingin diuji
# img_path = 'dataset/test/cat/cat.4004.jpg'
img_path = 'dataset/test/dog/dog.4001.jpg'

# Memuat dan memproses gambar
img = image.load_img(img_path, target_size=(150, 150)) # Sesuaikan dengna ukuran input model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) #Membuat batch dimensi

# Prediksi
prediction = model.predict(img_array)

# Interpretasi hasil
if prediction[0] > 0.5:
    print("Prediksi: Gambar ini adalah Anjing 🐶")
else:
    print("Prediksi: Gambar ini adalah Kucing 😺")