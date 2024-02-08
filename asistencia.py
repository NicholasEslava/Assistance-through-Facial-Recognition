import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime


# creas base de datos
ruta = 'empleados'
mis_imagenes = []
nombres_empleado = []
lista_empleados = os.listdir(ruta)

for nombre in lista_empleados:
    imagen_actual = cv2.imread(f'{ruta}/{nombre}')
    mis_imagenes.append(imagen_actual)
    nombres_empleado.append(os.path.splitext(nombre)[0])

print(lista_empleados)


# codificar imagenes
def codificar(imagenes):

    # crear una lista nueva
    lista_codificada = []

    # pasar imagenes a rgb
    for imagen in imagenes:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

        # codificar
        codificado = fr.face_encodings(imagen)[0]

        # agregar a la lista
        lista_codificada.append(codificado)

    # devolver lista codificada
    return lista_codificada


# registrar ingresos 
def registrar_ingresos(persona):
    f = open('registro.csv', 'r+')
    lista_datos = f.readline()
    nombres_registro = []
    for linea in lista_datos:
        ingreso = linea.split(',')
        nombres_registro.append(ingreso[0])
    
    if persona not in nombres_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime('%H:%M')
        f.writelines(f'\n{persona}, {string_ahora}')


lista_empleados_codificada = codificar(mis_imagenes)

# tomar una imagen de camara web
captura = cv2.VideoCapture(1)

# leer la imagen de la camara
while True:
 
    exito, imagen = captura.read()
    
    cv2.imshow("Capturando imagen",imagen)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('Q'):
        break


if not exito:
    print('No se ha podido tomar la captura')
else:

    # reconocer rostro en la captura
    rostro_captura = fr.face_locations(imagen)

    # codificar rostro capturado 
    rostro_captura_codificada = fr.face_encodings(imagen, rostro_captura)

    # buscar coicidencias
    for caracodi, caraubi in zip(rostro_captura_codificada, rostro_captura):
        coicidencia = fr.compare_faces(lista_empleados_codificada, caracodi)
        distancias = fr.face_distance(lista_empleados_codificada, caracodi)

        print(distancias)

        indice_coicidencia = numpy.argmin(distancias)

        # mostrar coicidencia 
        if distancias[indice_coicidencia] > 0.6:

            cv2.putText(imagen, f'MATCH NOT FOUND {distancias[indice_coicidencia].round(2)}', (50, 50),cv2.FONT_HERSHEY_COMPLEX,
            1, (255, 0, 0), 2)
            y1, x2, y2, x1 = caraubi

            # mostra NO coicidencia
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.rectangle(imagen, (x1, y2 - 35),  (x2, y2), (255, 0, 0), cv2.FILLED)
            cv2.putText(imagen, 'UNKNOWN', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # mostrar la imagen
            cv2.imshow('IMAGEN WEB', imagen)

            # mantener ventana abierta
            cv2.waitKey(0)

        else:

            # coicidencia
            nombre = nombres_empleado[indice_coicidencia]
            y1, x2, y2, x1 = caraubi
            cv2.rectangle(imagen, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(imagen, (x1, y2 - 35),  (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(imagen, nombre, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # registrar el ingreso 
            registrar_ingresos(nombre)

            # mostrar la imagen
            cv2.imshow('IMAGEN WEB', imagen)

            # mantener ventana abierta
            cv2.waitKey(0)



