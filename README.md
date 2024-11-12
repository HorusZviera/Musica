
# Proyecto de Reconocimiento de Comandos de Voz

Este proyecto está diseñado para reconocer comandos de voz utilizando aprendizaje automático con procesamiento de señales de audio y aprendizaje profundo. Utiliza el conjunto de datos Google Speech Commands para entrenar y validar un modelo que reconoce comandos de voz.

## Estructura del Proyecto

El proyecto consta de los siguientes archivos que estan en la carpeta Archivos:

1. **descarga_Y_carga_dedatos.py**: Este script maneja la descarga, extracción y preprocesamiento del conjunto de datos Google Speech Commands. Extrae características MFCC de los archivos de audio y guarda los datos procesados en un archivo `.npz`.
2. **modelo_de_datos.py**: Este script define, compila y entrena un modelo de aprendizaje profundo con los datos preprocesados. El modelo entrenado se guarda en un archivo `.h5`.
3. **ver_modelo.py**: Este script carga un modelo guardado desde un archivo `.h5` y muestra un resumen de la arquitectura del modelo.

El proyecto también incluye el directorio **speech_commands**, que contiene los datos de audio necesarios para el entrenamiento.



## Descripción de Archivos

### 1. descarga_Y_carga_dedatos.py

Este script se utiliza para descargar y preprocesar el conjunto de datos:

- Descarga el conjunto de datos Google Speech Commands desde la URL proporcionada.
- Extrae el conjunto de datos y procesa cada archivo de audio para extraer características MFCC (coeficientes cepstrales de frecuencia Mel).
- Codifica las etiquetas usando `LabelEncoder` para facilitar el procesamiento por el modelo.
- Guarda los datos preprocesados como `speech_commands_data_procesado.npz` para su uso posterior en el entrenamiento del modelo.

Para ejecutar este script, usa:
```bash
python descarga_Y_carga_dedatos.py
```
Se te pedirá que confirmes si deseas procesar los datos. Ingresa `1` para procesar y guardar los datos.

### 2. modelo_de_datos.py

Este script se encarga de entrenar el modelo de aprendizaje automático.

- Carga los datos preprocesados desde `speech_commands_data_procesado.npz`.
- Normaliza las características y divide los datos en conjuntos de entrenamiento y validación.
- Define un modelo de red neuronal con múltiples capas `Dense`, `BatchNormalization` y `Dropout` para mejorar la generalización.
- Compila y entrena el modelo usando la pérdida de entropía cruzada categórica y el optimizador Adam.
- Guarda el modelo entrenado como `modelo_comandos_de_voz_mejorado.h5`.

Para ejecutar este script, usa:
```bash
python modelo_de_datos.py
```

### 3. ver_modelo.py

Este script se utiliza para ver la arquitectura de un modelo entrenado.

- Carga el modelo desde un archivo `.h5`.
- Muestra un resumen de la arquitectura del modelo, mostrando la configuración de cada capa y el total de parámetros.

Para usar este script, ejecuta:
```bash
python ver_modelo.py
```

La ruta del archivo de modelo se puede modificar en el script para cargar cualquier archivo de modelo guardado.

## Requisitos

Este proyecto requiere los siguientes paquetes de Python:

- `numpy`
- `tensorflow`
- `librosa`
- `matplotlib`
- `scikit-learn`
- `seaborn`

Instala las dependencias con:
```bash
pip install numpy tensorflow librosa matplotlib scikit-learn seaborn
```

## Conjunto de Datos

El conjunto de datos utilizado en este proyecto es el [Google Speech Commands Dataset](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz). Este conjunto de datos se descargará y procesará automáticamente mediante el script `descarga_Y_carga_dedatos.py`.

## Uso

1. Ejecuta `descarga_Y_carga_dedatos.py` para descargar, extraer y preprocesar el conjunto de datos.
2. Ejecuta `modelo_de_datos.py` para entrenar el modelo.
3. Ejecuta `ver_modelo.py` para mostrar el resumen del modelo.

## Detalles del Modelo

- El modelo es una red neuronal completamente conectada con cinco capas.
- Cada capa usa activación ReLU, normalización por lotes y dropout para mejorar la estabilidad del entrenamiento y la generalización.
- La capa de salida usa activación softmax para clasificación multiclase.

## Notas Adicionales

- El número de épocas y el tamaño del lote en `modelo_de_datos.py` se pueden ajustar para experimentación, pero por defecto tiene 100 épocas y un tamaño de lote de 32.
- El archivo de datos preprocesados `speech_commands_data_procesado.npz` permite un reentrenamiento más rápido del modelo sin necesidad de volver a procesar los datos brutos.

## Link video explicativo
- [Video Explicativo](https://www.youtube.com/watch?v=_tDSlr7QJLQ)
    
## Enlaces Útiles

- [Documentación de TensorFlow](https://www.tensorflow.org/learn)
- [Documentación de Librosa](https://librosa.org/doc/latest/index.html)
- [Documentación de Scikit-learn](https://scikit-learn.org/stable/user_guide.html)
