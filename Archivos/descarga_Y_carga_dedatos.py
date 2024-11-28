import librosa                                      # Libreria para procesamiento de audio
import librosa.display                              # Modulo para visualizacion de datos de audio
import matplotlib.pyplot as plt                     # Libreria para visualizacion de datos
import tarfile                                      # Modulo para descomprimir archivos .tar.gz
import os                                           # Modulo para manipulacion de archivos y directorios
import urllib.request                               # Modulo para descargar archivos de la web
import numpy as np                                  # Libreria para manipulacion de arreglos
from sklearn.preprocessing import LabelEncoder      # Codificacion de etiquetas
import seaborn as sns                               # Libreria para visualizacion de datos estadisticos






# 1. Cargar la base de datos con la informacion para entrenar el modelo de Machine Learning: Google Speech Commands Dataset (2.3gb) ------------------------------------------------
url = 'https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
data_dir = '../Datos/speech_commands'  # Directorio donde se guardara el conjunto de datos
output_file = '../Datos/speech_commands_data_procesado.npz'


# Verificar si el directorio de datos existe y contiene carpetas
if os.path.exists(data_dir) and any(os.path.isdir(os.path.join(data_dir, d)) for d in os.listdir(data_dir)):
    print(f"El directorio {data_dir} contiene carpetas.")
else:
    print(f"El directorio {data_dir} no contiene carpetas o no existe.")

# Preguntar al usuario si desea procesar los datos
process_data = input("¿Desea procesar los datos? Ingrese 1 para si, cualquier otra tecla para no: ")

if process_data == '1':

    # Descargar el conjunto de datos
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        filename = os.path.join(data_dir, 'speech_commands_v0.02.tar.gz')
        print("Descargando Carpeta con datos de prueba...")
        urllib.request.urlretrieve(url, filename)
        
        # Descomprimir el archivo
        print("Descomprimiendo carpeta...")
        with tarfile.open(filename, 'r:gz') as tar:
            tar.extractall(path=data_dir)
        
        # Eliminar el archivo comprimido
        os.remove(filename)

    print("Conjunto de datos descargado y extraido en:", data_dir)

    # 2. Cargar los datos de entrenamiento, validacion, codificar las etiquetas y guardar en .npz

    # Inicializar listas para almacenar las caracteristicas de audio y etiquetas
    commands = os.listdir(data_dir)
    features = []
    labels = []

    # Funcion para extraer MFCCs de un archivo de audio
    def extract_mfcc(file_path, sample_rate=22050, n_mfcc=13):
        audio, sr = librosa.load(file_path, sr=sample_rate)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        mfccs_mean = np.mean(mfccs, axis=1)  # Promediar a lo largo del tiempo
        return mfccs_mean

    # Procesar cada carpeta de comandos
    for command in commands:
        command_dir = os.path.join(data_dir, command)
        if os.path.isdir(command_dir):  # Solo procesar carpetas
            print(f'Procesando comando: {command}')
            for filename in os.listdir(command_dir):
                if filename.endswith('.wav'):
                    file_path = os.path.join(command_dir, filename)
                    
                    # Extraer caracteristicas y anadir a las listas
                    mfccs = extract_mfcc(file_path)
                    features.append(mfccs)
                    labels.append(command)

    # Convertir listas a arrays de NumPy
    features = np.array(features)
    labels = np.array(labels)

    print("Preprocesamiento completado.")
    print("Caracteristicas de audio:", features.shape)
    print("Etiquetas:", labels.shape)

    # Convertir etiquetas a valores numericos para facilitar el entrenamiento
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    classes = label_encoder.classes_  # Obtener el mapeo de clases

    print("Etiquetas codificadas:", labels_encoded.shape)

    # Si existe un archivo antiguo, eliminarlo para evitar conflictos
    if os.path.exists(output_file):
        os.remove(output_file)

    # Guardar los arrays en un archivo .npz para evitar tener que procesar los datos nuevamente
    np.savez(output_file, features=features, labels=labels_encoded, classes=classes)
    print(f"Datos guardados en {output_file}")

    # Verificar el contenido del archivo .npz
    data = np.load(output_file, allow_pickle=True)
    print("Contenido del archivo guardado:", list(data.keys()))  # Confirmacion de claves en el archivo

else:
    # Mostrar el contenido del archivo .npz si no se procesan los datos
    try:
        data = np.load(output_file, allow_pickle=True)
        features = data['features']
        labels_encoded = data['labels']

        print("Datos cargados desde archivo .npz")
        print("Caracteristicas de audio:", features.shape)
        print("Etiquetas codificadas:", labels_encoded.shape)

        # Obtener las etiquetas unicas
        unique_labels = np.unique(labels_encoded)
        print("Etiquetas unicas:", unique_labels)

        # Mostrar algunos ejemplos de caracteristicas y etiquetas
        print("\nEjemplos de caracteristicas y etiquetas:")
        for i in range(5):
            print(f"Ejemplo {i + 1}:")
            print("Caracteristicas:", features[i])
            print("Etiqueta codificada:", labels_encoded[i])

        # Visualizar la distribucion de las etiquetas
        plt.figure(figsize=(10, 6))
        sns.barplot(x=np.arange(len(np.bincount(labels_encoded))), y=np.bincount(labels_encoded))
        plt.title('Distribucion de Comandos de Voz')
        plt.xlabel('Comando')
        plt.ylabel('Cantidad de Muestras')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print("El archivo 'speech_commands_data_procesado.npz' no se encuentra. Asegurese de que los datos han sido procesados y guardados.")
