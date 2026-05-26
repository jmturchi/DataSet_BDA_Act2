# ==============================================================================
# PROYECTO: MANTENIMIENTO PREDICTIVO EN ENTORNOS IoT (DATASET C-MAPSS NASA)
# ALUMNO: MATÍAS TURCHI
# ==============================================================================

# ------------------------------------------------------------------------------
# BLOQUE 1: IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN INICIAL
# ------------------------------------------------------------------------------
import pandas as pd  # Librería para manipulación y análisis de datos estructurados (DataFrames)
import numpy as np  # Librería para operaciones matemáticas y manejo de vectores/matrices
import matplotlib.pyplot as plt  # Librería base para la generación de gráficos y plots
from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Herramientas para normalización y escalado de variables
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score, ConfusionMatrixDisplay  # Métricas de evaluación
from keras.models import Sequential  # Contenedor lineal de Keras para apilar capas de la red neuronal
from keras.layers import Input, LSTM, Dropout, Dense  # Capas específicas para construir la arquitectura de la red
from keras.callbacks import EarlyStopping  # Callback para detener el entrenamiento si el modelo deja de mejorar

# Configuración de estilo para los gráficos de matplotlib
plt.style.use('ggplot')  # Aplica el estilo visual 'ggplot' (fondos grises con cuadrículas blancas)

"""
================================================================================
💡 CONEXIÓN CON CIENCIA DE DATOS: ENTORNO Y HERRAMIENTAS
Dividimos las herramientas en tres pilares fundamentales de la Ciencia de Datos:
1. Wrangling (Pandas/NumPy): Permiten estructurar los datos crudos del IoT.
2. Análisis Exploratorio y Validación (Matplotlib/Sklearn): Herramientas 
   para auditar la calidad estadística de las transformaciones y del modelo.
3. Deep Learning (Keras): Framework especializado en procesar tensores 
   multidimensionales a gran escala, óptimo para algoritmos como LSTM.
================================================================================
"""

# ------------------------------------------------------------------------------
# BLOQUE 2: CARGA DE DATOS Y INGENIERÍA DE CARACTERÍSTICAS (RUL, TTF, LABELS)
# ------------------------------------------------------------------------------
col_names = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']

print("📥 Cargando conjuntos de datos desde los repositorios...")
dataset_train = pd.read_csv('https://raw.githubusercontent.com/jmturchi/DataSet_BDA_Act2/refs/heads/main/PM_train', sep=' ', header=None).drop([26, 27], axis=1)
dataset_train.columns = col_names  

dataset_test = pd.read_csv('https://raw.githubusercontent.com/jmturchi/DataSet_BDA_Act2/refs/heads/main/PM_test', sep=' ', header=None).drop([26, 27], axis=1)
dataset_test.columns = col_names  

pm_truth = pd.read_csv('https://raw.githubusercontent.com/jmturchi/DataSet_BDA_Act2/refs/heads/main/PM_truth', sep=' ', header=None).drop([1], axis=1)
pm_truth.columns = ['more']  
pm_truth['id'] = pm_truth.index + 1  

rul = pd.DataFrame(dataset_test.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']  

pm_truth['rtf'] = pm_truth['more'] + rul['max']
pm_truth.drop('more', axis=1, inplace=True)  

dataset_test = dataset_test.merge(pm_truth, on=['id'], how='left')
dataset_test['ttf'] = dataset_test['rtf'] - dataset_test['cycle']
dataset_test.drop('rtf', axis=1, inplace=True)  

dataset_train['ttf'] = dataset_train.groupby(['id'])['cycle'].transform(max) - dataset_train['cycle']

df_train = dataset_train.copy()
df_test = dataset_test.copy()

period = 30
df_train['label_bc'] = df_train['ttf'].apply(lambda x: 1 if x <= period else 0)
df_test['label_bc'] = df_test['ttf'].apply(lambda x: 1 if x <= period else 0)

"""
================================================================================
💡 CONEXIÓN CON CIENCIA DE DATOS: INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)
El dataset original entrega datos crudos de series temporales. Mediante la Ingeniería de 
Características construimos la variable rezagada TTF (Time-to-Failure). 
Transformar un problema continuo a uno de clasificación binaria (¿va a fallar en los 
próximos 30 ciclos?) permite convertir un problema complejo de regresión en un 
sistema robusto de alerta temprana (Early Warning System).
================================================================================
"""

# ------------------------------------------------------------------------------
# BLOQUE 3: PREPROCESAMIENTO Y PROCESO CORRECTO DE PCA (TRAIN VS TEST)
# ------------------------------------------------------------------------------
print("⚙️ Ejecutando Análisis de Componentes Principales (PCA)...")
dataset_train_pca = df_train.drop(['id', 'cycle', 'setting1', 'setting2', 'setting3', 'ttf', 'label_bc'], axis=1)
dataset_train_no_pca = df_train.drop(['setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'], axis=1)

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(dataset_train_pca)

cov_x = np.cov(x_train_scaled.T)
eig_vals, eig_vecs = np.linalg.eig(cov_x)

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
eig_pairs.sort(key=lambda x: x[0], reverse=True)

matrix_w = np.hstack([eig_pairs[i][1].reshape(21, 1) for i in range(8)])
Y_train = x_train_scaled.dot(matrix_w)

pca_train_df = pd.DataFrame(data=Y_train, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6', 'PC 7', 'PC 8'])
dataset_train_no_pca = dataset_train_no_pca.join(pca_train_df)

# --- PROCESAMIENTO CORRECTO DEL TEST (PREVENCIÓN DE DATA LEAKAGE) ---
dataset_test_pca = df_test.drop(['id', 'cycle', 'setting1', 'setting2', 'setting3', 'ttf', 'label_bc'], axis=1)
dataset_test_no_pca = df_test.drop(['setting3', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'], axis=1)

# Heredamos la transformación de Train sin hacer un nuevo fit
x_test_scaled = scaler.transform(dataset_test_pca)
Y_test = x_test_scaled.dot(matrix_w)

pca_test_df = pd.DataFrame(data=Y_test, columns=['PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6', 'PC 7', 'PC 8'])
dataset_test_no_pca = dataset_test_no_pca.join(pca_test_df)

"""
================================================================================
💡 CONEXIÓN CON CIENCIA DE DATOS: REDUCCIÓN DE DIMENSIONALIDAD Y DATA LEAKAGE
El PCA busca proyectar los datos hacia subespacios de máxima varianza. Para evitar 
la "Fuga de Datos" (Data Leakage) el conjunto de test (que simula el futuro en producción) 
se debe adaptar obligatoriamente a las reglas estadísticas del pasado (Train), 
heredando la media, desviación estándar y la matriz de proyección matemática $W$.
================================================================================
"""

# ------------------------------------------------------------------------------
# BLOQUE 4: ESCALADO FINAL (MIN-MAX SCALER)
# ------------------------------------------------------------------------------
features_col_name = ['setting1', 'setting2', 'PC 1', 'PC 2', 'PC 3', 'PC 4', 'PC 5', 'PC 6', 'PC 7', 'PC 8']
min_max_scaler = MinMaxScaler()

dataset_train_no_pca[features_col_name] = min_max_scaler.fit_transform(dataset_train_no_pca[features_col_name])
dataset_test_no_pca[features_col_name] = min_max_scaler.transform(dataset_test_no_pca[features_col_name])

# ------------------------------------------------------------------------------
# BLOQUE 5: PREPARACIÓN DE TENSORES EN SECUENCIAS TEMPORALES (3D)
# ------------------------------------------------------------------------------
def gen_sequence(id_df, seq_length, seq_cols):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    lstm_array = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        lstm_array.append(data_array[start:stop, :])
    return np.array(lstm_array)

def gen_label(id_df, seq_length, seq_cols, label):
    df_zeros = pd.DataFrame(np.zeros((seq_length - 1, id_df.shape[1])), columns=id_df.columns)
    id_df = pd.concat([df_zeros, id_df], ignore_index=True)
    num_elements = id_df.shape[0]
    y_label = []
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        y_label.append(id_df[label][stop])
    return np.array(y_label)

seq_length = 50  
seq_cols = features_col_name  

print("📦 Estructurando tensores temporales 3D para la red LSTM...")
X_train = np.concatenate(list(gen_sequence(dataset_train_no_pca[dataset_train_no_pca['id'] == id], seq_length, seq_cols) for id in dataset_train_no_pca['id'].unique()))
y_train = np.concatenate(list(gen_label(dataset_train_no_pca[dataset_train_no_pca['id'] == id], seq_length, seq_cols, 'label_bc') for id in dataset_train_no_pca['id'].unique()))

X_test = np.concatenate(list(gen_sequence(dataset_test_no_pca[dataset_test_no_pca['id'] == id], seq_length, seq_cols) for id in dataset_test_no_pca['id'].unique()))
y_test = np.concatenate(list(gen_label(dataset_test_no_pca[dataset_test_no_pca['id'] == id], seq_length, seq_cols, 'label_bc') for id in dataset_test_no_pca['id'].unique()))

"""
================================================================================
💡 CONEXIÓN CON CIENCIA DE DATOS: TENSORES TEMPORALES TRIDIMENSIONALES
Para que una red LSTM funcione, debemos reestructurar la información bidimensional en 
un Tensor Tridimensional con la forma: [Muestras, Pasos de Tiempo, Características].
Este proceso de ventana deslizante ("sliding window") preserva la cronología exacta 
de los ciclos del motor, permitiendo al algoritmo analizar tendencias temporales.
================================================================================
"""

# ------------------------------------------------------------------------------
# BLOQUE 6: CONSTRUCCIÓN Y ENTRENAMIENTO DE LA RED NEURONAL RECURRENTE (LSTM)
# ------------------------------------------------------------------------------
nb_features = X_train.shape[2]  
timestamp = seq_length  

model = Sequential()  
model.add(Input(shape=(timestamp, nb_features)))  
model.add(LSTM(units=100, return_sequences=True))  
model.add(Dropout(0.2))  
model.add(LSTM(units=50, return_sequences=False))  
model.add(Dropout(0.2))  
model.add(Dense(units=1, activation='sigmoid'))  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("🚀 Iniciando entrenamiento de la Red Neuronal LSTM...")
model.fit(X_train, y_train, epochs=10, batch_size=200, validation_split=0.05, verbose=1,
          callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')])

# ------------------------------------------------------------------------------
# BLOQUE 7: EVALUACIÓN DEL MODELO Y MÉTRICAS DE RENDIMIENTO
# ------------------------------------------------------------------------------
print("🧠 Realizando inferencia sobre el conjunto de validación...")
y_pred_prob = model.predict(X_test)
y_pred = np.where(y_pred_prob > 0.5, 1, 0)

print('\n======================================================')
print('📈 REPORTES DE EVALUACIÓN FINAL DEL MODELO')
print('======================================================')
print(f'Exactitud (Accuracy): {accuracy_score(y_test, y_pred):.4f}')
print(f'F1-score (Ponderado): {f1_score(y_test, y_pred, average="weighted"):.4f}')
print(f'Exactitud Equilibrada (Balanced Accuracy): {balanced_accuracy_score(y_test, y_pred):.4f}')
print('======================================================')

# --- GENERACIÓN DE LA MATRIZ DE CONFUSIÓN ---
cm = confusion_matrix(y_test, y_pred)  
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal (0)', 'Fallo Inminente (1)'])  
disp.plot(cmap='Blues', values_format='d')  
plt.title("Matriz de Confusión Final - Red LSTM")  
plt.savefig('matriz_confusion.png', dpi=300) # Guardamos el gráfico automáticamente para GitHub
print("💾 Matriz de confusión guardada como 'matriz_confusion.png'")
plt.show()