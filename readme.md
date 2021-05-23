# Monografía – Especialización en Analítica y ciencia de datos

## Presentado por:

- Danilo Diaz Valencia

- Santiago Jaramillo 

- Julian Eusse Jaramillo

## Identificación de Etiquetas en Líneas de Manufactura

Este repo es un guía para la ejecución demo del proyecto Identificación de etiquetas en líneas de manufactura.

- Detección de Hoja
- Reorientación de Hoja
- Detección de Palabra 

<p align="center">
  <img src="Docs/project_flow.JPG">
</p>

## Tabla de contenido

1. Requerimientos
2. Experimentos (Notebooks)
3. Recolección y Etiquetado de Datos
4. Separado de Datos
5. Generación de tf record
6. Entrenamiento de Modelos
7. Exportado de Modelos
8. Prueba de modelo terminado
9. Docker Image

## Requerimientos

## Instalar TensorFlow

El primer paso es instalar TensorFlow-. Hay muchos videos geniales en YouTube que brindan más detalles sobre cómo hacer esto y recomiendo echar un vistazo a este [video](https://www.youtube.com/watch?v=oqd54apcgGE) y [guía](https://github.com/armaanpriyadarshan/Training-a-Custom-TensorFlow-2.X-Object-Detector) visualización de cómo hacerlo. Los requisitos para TensorFlow-GPU son Anaconda, CUDA y cuDNN. Los dos últimos, CUDA y cuDNN  , son necesarios para utilizar la memoria gráfica de la GPU y cambiar la carga de trabajo aunque también se puede usar Tensorflow normal pero esto retrasar entrenamiento y procesos . Mientras tanto, Anaconda es lo que usaremos para configurar un entorno virtual donde instalaremos los paquetes necesarios.
```
conda create -n tensorflow pip python=3.8

```
Instalar tensorflow
```
pip install tensorflow
```
si esta usando tensorflow GPU use
```
pip install tensorflow-gpu
```

## Instalar TensorFlow-Api

[TensorFlow models repository](https://github.com/tensorflow/models)

```
git clone https://github.com/tensorflow/models.git
```