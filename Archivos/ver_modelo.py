import tensorflow as tf

def ver_modelo(ruta_archivo):
    # Cargar el modelo desde el archivo .h5
    modelo = tf.keras.models.load_model(ruta_archivo)
    
    # Mostrar un resumen del modelo
    modelo.summary()

# Ejemplo de uso
ruta_archivo = '../Datos/modelo_comandos_de_voz.h5'
ver_modelo(ruta_archivo)