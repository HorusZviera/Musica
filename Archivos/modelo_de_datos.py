import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Configurar el número de hilos
num_threads = os.cpu_count()
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

# Cargar datos procesados
data_file = '../Datos/speech_commands_data_procesado.npz'
data = np.load(data_file, allow_pickle=True)
features = data['features']
labels = data['labels']
classes = data['classes'] if 'classes' in data else np.unique(labels)

# Normalizar las características
features = features / np.max(np.abs(features), axis=0)  # Normalización entre -1 y 1

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, random_state=42)

# Convertir etiquetas a formato categórico
y_train = to_categorical(y_train, num_classes=len(classes))
y_val = to_categorical(y_val, num_classes=len(classes))

# Definir el modelo
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')  # Salida con número de clases
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
epochs = 100  # Aumentar el número de épocas
batch_size = 32  # Reducir el tamaño del lote
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))

# Guardar el modelo entrenado
model.save('../Datos/modelo_comandos_de_voz_mejorado.h5')
print("Modelo guardado como 'modelo_comandos_de_voz_mejorado.h5' en la carpeta Datos.")
