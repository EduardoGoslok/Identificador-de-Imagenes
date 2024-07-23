import tensorflow as tf  #Tiene modelos de Machine Learning
from keras import layers,models  
import os # sirve para poner las rutas
import numpy as np # libreria del data science
import cv2 # para usar imagenes
import random 

width = 300  #Ancho predeterminado de las imagenes
height = 300  #Largo predeterminado de las imagenes
ruta_train = 'cats_and_dogs/train/'
ruta_predict = '   '  #Ruta de la imagen a identificar 

#Datos de entrenamiento y validacion por separado

train_x = []
train_y = []

labels = os.listdir(ruta_train)

for i in os.listdir(ruta_train):
    for j in os.listdir(ruta_train+i):
        img = cv2.imread(ruta_train+i+'/'+j)
        resized_image = cv2.resize(img, (width,height))

        train_x.append(resized_image)

        for x,y in enumerate(labels):
            if y == i:
                array = np.zeros(len(labels))
                array[x]=1
                train_y.append(array)

x_data = np.array(train_x)
y_data = np.array(train_y)

model = tf.keras.Sequential([
    layers.Conv2D(32, 3,3, input_shape=(width, height, 3)),  #Detecta bordes, texturas, etc. en la imagen de entrada.
    layers.Activation('relu'),  #Permite que aprenda relaciones mas complejas de los datos 
    layers.MaxPooling2D(pool_size=(2,2)), #Reduce dimensiones de la imagen, conservando características importantes
    layers.Conv2D(32, 3,3),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Conv2D(64, 3,3),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(64),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    layers.Dense(2),
    layers.Activation('sigmoid') #rango [0, 1], representando la probabilidad.
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 100

model.fit(x_data, y_data, epochs = epochs)

models.save_model(model, 'mimodel.keras')

model= models.load_model('mimodel.keras')


my_image = cv2.imread(ruta_predict) #Función de OpenCV para leer imágenes
my_image = cv2.resize(my_image, (width, height)) #Ajusta el tamaño a las dimenciones especificadas

result = model.predict(np.array([my_image]))[0]

porcentaje = max(result)*10

grupo = labels[result.argmax()]

print(grupo, round(porcentaje))

cv2.imshow(ruta_predict)
cv2.waitKey(0)