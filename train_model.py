import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Konfigurasi path dataset
base_dir = 'dataset' # folder utama
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

#Preprocessing dan Augmentasi
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

print("Label mapping:", train_generator.class_indices)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# ------------------------------
# Arsitektur Model CNN
# ------------------------------
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),  # regularisasi
    layers.Dense(1, activation='sigmoid')  # untuk binary classification
])

# ------------------------------
# Kompilasi dan Training
# ------------------------------
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# ------------------------------
# Mulai Training
# ------------------------------
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# ------------------------------
# Simpan Model
# ------------------------------
model.save("model_cnn.h5")
print("✅ Model berhasil disimpan sebagai model_cnn.h5")

