# Mantenimiento Predictivo en Entornos IoT con PCA y Redes LSTM

Este repositorio contiene una solución integral de Machine Learning aplicada al mantenimiento predictivo industrial, utilizando el popular dataset **C-MAPSS de la NASA** para la simulación de degradación de turbinas de avión.

## 📋 Arquitectura del Proyecto

El flujo de trabajo implementado resuelve un problema dinámico de series temporales mediante las siguientes etapas de Ciencia de Datos:

1. **Ingeniería de Características**: Cálculo del *Time-To-Failure* (TTF) y binarización del estado crítico de la máquina (ventana límite de 30 ciclos operativos).
2. **Reducción de Dimensionalidad (PCA)**: Procesamiento y descorrelación de los 21 sensores analógicos originales del IoT, transformándolos en **8 Componentes Principales** ortogonales. Se implementaron controles estrictos para mitigar el *Data Leakage* al heredar los parámetros de entrenamiento al test.
3. **Estructuración de Tensores (3D)**: Modelado de ventanas de tiempo deslizantes (*sliding windows*) de tamaño $t=50$ para preservar el contexto histórico.
4. **Deep Learning**: Red Neuronal Recurrente basada en arquitectura **LSTM** (Long Short-Term Memory) con capas de regularización Dropout y criterios de parada temprana (*Early Stopping*).

## 🚀 Requisitos para Ejecución

Para correr el script de forma local, instala las dependencias necesarias:

```bash
pip install pandas numpy matplotlib scikit-learn tensorflow