#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Define la ruta a la carpeta donde están tus archivos de datos.
# Asegúrate de especificar los datos por la ruta real en tu sistema.
data_path = r"PATH"

import wfdb
import pandas as pd
import os

# 1. Inspeccionar el archivo de datos clínicos (AdditionalData.csv)
print("--- Inspección del archivo AdditionalData.csv ---")
try:
    df_clinical = pd.read_csv(os.path.join(data_path, 'AdditionalData.csv'))
    print(f"Número de filas y columnas: {df_clinical.shape}")
    print("\nPrimeras 5 filas:")
    print(df_clinical.head())
    print("\nInformación de las columnas (tipos de datos y valores no nulos):")
    df_clinical.info()
except FileNotFoundError:
    print(f"Error: No se encontró el archivo 'AdditionalData.csv' en la ruta {data_path}")

# 2. Inspeccionar un archivo de registro WFDB (.dat y .hea)
# Usaremos el ID de ejemplo. Puedes cambiarlo por cualquiera nombre de los 98 registros anotados.
record_name = '001'
print("\n--- Inspección de un registro de ECG WFDB (.hea y .dat) ---")
try:
    record = wfdb.rdrecord(os.path.join(data_path, record_name))
    print(f"Nombre del registro: {record.record_name}")
    print(f"Número de canales: {record.n_sig}")
    print(f"Frecuencia de muestreo (Hz): {record.fs}")
    print(f"Duración total del registro: {record.sig_len / record.fs} segundos")
    print(f"Canales de la señal: {record.sig_name}")
    print("\nPrimeros 100 puntos de la señal del primer canal:")
    # Imprime una pequeña porción de la señal para evitar sobrecargar la salida.
    print(record.p_signal[:100, 0])
except FileNotFoundError:
    print(f"Error: No se encontró el registro WFDB '{record_name}' en la ruta {data_path}")

# 3. Inspeccionar un archivo de anotaciones de ritmo (.atr)
print("\n--- Inspección del archivo de anotaciones de ritmo (.atr) ---")
try:
    annotation = wfdb.rdann(os.path.join(data_path, record_name), 'atr')
    print(f"Número total de anotaciones: {len(annotation.sample)}")
    print("\nPrimeras 10 anotaciones:")
    # 'sample' es la posición del beat, 'symbol' es el tipo de anotación.
    # 'aux_note' contiene la anotación de ritmo (e.g., '(AFIB').
    df_annotations = pd.DataFrame({
        'sample': annotation.sample[:10],
        'symbol': annotation.symbol[:10],
        'aux_note': annotation.aux_note[:10]
    })
    print(df_annotations)
    
    # Imprime un resumen de los tipos de ritmo encontrados.
    print("\nConteo de tipos de ritmo en este registro:")
    # Aseguramos de que el símbolo sea la clave para el conteo
    rhythm_counts = pd.Series(annotation.aux_note).value_counts()
    print(rhythm_counts)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de anotación '.atr' para el registro '{record_name}'")
except Exception as e:
    print(f"Ocurrió un error al leer las anotaciones: {e}")

# 4. Inspeccionar un archivo de picos R (.qrs)
print("\n--- Inspección del archivo de picos R (.qrs) ---")
try:
    r_peaks = wfdb.rdann(os.path.join(data_path, record_name), 'qrs')
    print(f"Número de picos R detectados: {len(r_peaks.sample)}")
    print("\nPrimeros 10 picos R (posiciones en la señal):")
    print(r_peaks.sample[:10])
except FileNotFoundError:
    print(f"Error: No se encontró el archivo de picos R '.qrs' para el registro '{record_name}'")
except Exception as e:
    print(f"Ocurrió un error al leer los picos R: {e}")

#Optimizacion del tipo de archivo a manejar
import wfdb
import pandas as pd
import numpy as np
import os

import pyarrow

# Carga la información clínica para saber qué registros están anotados
df_clinical = pd.read_csv(os.path.join(data_path, 'AdditionalData.csv'))
annotated_records = df_clinical[df_clinical['Annotated']]['Data_ID'].tolist()

# Lista para almacenar las características de todos los registros
all_features = []

for record_id in annotated_records:
    try:
        record_name = str(record_id).zfill(3) # Formato de 3 dígitos, ej. '001'
        print(f"Procesando el registro: {record_name}")

        # 1. Leer las anotaciones de picos R y de ritmo
        r_peaks = wfdb.rdann(os.path.join(data_path, record_name), 'qrs')
        rhythm_ann = wfdb.rdann(os.path.join(data_path, record_name), 'atr')

        # 2. Asignar la etiqueta de ritmo a cada pico R
        # Esta es la parte más crítica y compleja.
        # Creamos un DataFrame con los picos R y su posición temporal
        rhythm_intervals = pd.DataFrame({
            'start_index': rhythm_ann.sample,
            'rhythm_label': rhythm_ann.aux_note
        })
        
        # # Añadir un 'end_index' para cada intervalo.
        rhythm_intervals['end_index'] = rhythm_intervals['start_index'].shift(-1).fillna(np.inf)
        
        # Crea un DataFrame con las posiciones del pico R.
        df_rpeaks = pd.DataFrame({'sample_index': r_peaks.sample})
        
        # Inicializa una columna para etiquetas de ritmo
        df_rpeaks['rhythm_label'] = ""
        
        # Itera a través de los intervalos rítmicos y asignas la etiqueta a los picos R.
        for idx, row in rhythm_intervals.iterrows():
            start = row['start_index']
            end = row['end_index']
            label = row['rhythm_label']
            
            # Comprueba si el índice de muestra del pico R se encuentra dentro del intervalo del ritmo actual.
            mask = (df_rpeaks['sample_index'] >= start) & (df_rpeaks['sample_index'] < end)
            df_rpeaks.loc[mask, 'rhythm_label'] = label
        
        # Obtenemos la etiqueta de ritmo para cada intervalo
        # y la unimos al DataFrame de picos R
        # (Esto es una simplificación, la lógica real es más compleja)
        labels = np.zeros(len(df_rpeaks), dtype='object')
        for i in range(len(rhythm_ann.sample)):
            start_index = rhythm_ann.sample[i]
            end_index = rhythm_ann.sample[i+1] if i+1 < len(rhythm_ann.sample) else np.inf
            current_label = rhythm_ann.aux_note[i]
            
            # Asigna esta etiqueta a todos los picos R dentro del intervalo
            labels[(df_rpeaks['sample_index'] >= start_index) & (df_rpeaks['sample_index'] < end_index)] = current_label

        df_rpeaks['rhythm_label'] = labels

        # 3. Calcular los intervalos RR
        df_rpeaks['rr_interval'] = df_rpeaks['sample_index'].diff().fillna(0) / 200.0  # en segundos

        # 4. Extracción de características
        # Puedes añadir características más complejas aquí si tus archivos tiene diferentes indicaciones u ociones
        df_rpeaks['rr_std_5_beats'] = df_rpeaks['rr_interval'].rolling(window=5, min_periods=1).std()
        
        # 5. Añadir la ID del registro
        df_rpeaks['Data_ID'] = record_id
        
        # 6. Guardar las características extraídas
        all_features.append(df_rpeaks)

    except FileNotFoundError:
        print(f"Advertencia: Archivos no encontrados para el registro {record_name}. Saltando...")
        continue
    except Exception as e:
        print(f"Error al procesar el registro {record_name}: {e}")

# Unir todas las características en un solo DataFrame
if all_features:
    df_all_features = pd.concat(all_features, ignore_index=True)
    # Ahora puedes guardar este DataFrame en un archivo Parquet
    print("\nCaracterísticas extraídas para todos los registros anotados:")
    print(df_all_features.head())
    output_path = os.path.join('data', 'processed', 'holter_features.parquet')
    df_all_features.to_parquet(output_path, engine='pyarrow') # Se requiere 'pyarrow' o 'fastparquet'
    print(f"Datos procesados guardados en: {output_path}")

#Lectura del nuevo formato obtenido, esto reduce bien el tiempo de compilacion 
import wfdb
import pandas as pd
import numpy as np
import os

# Cargar información clínica
df_clinical = pd.read_csv(os.path.join(data_path, 'AdditionalData.csv'))
annotated_records = df_clinical[df_clinical['Annotated']]['Data_ID'].tolist()

# Lista para almacenar características de todos los registros
all_features = []

# Recorre secuencialmente los registros anotados.
for record_id in annotated_records:
    try:
        record_name = str(record_id).zfill(3)
        print(f"Procesando el registro: {record_name}")

        # 1. Lee anotaciones sobre picos R y ritmo
        r_peaks = wfdb.rdann(os.path.join(data_path, record_name), 'qrs')
        rhythm_ann = wfdb.rdann(os.path.join(data_path, record_name), 'atr')

        # 2. Asignaa etiquetas de ritmo (lógica robusta)
        df_rpeaks = pd.DataFrame({'sample_index': r_peaks.sample})
        labels = []
        label_index = 0
        current_rhythm = rhythm_ann.aux_note[0]
        
        for r_peak_sample in df_rpeaks['sample_index']:
            if label_index + 1 < len(rhythm_ann.sample) and r_peak_sample >= rhythm_ann.sample[label_index + 1]:
                label_index += 1
                current_rhythm = rhythm_ann.aux_note[label_index]
            labels.append(current_rhythm)
        
        df_rpeaks['rhythm_label'] = labels

        # 3. Calcula los intervalos RR
        df_rpeaks['rr_interval'] = df_rpeaks['sample_index'].diff().fillna(0) / 200.0

        # 4. Extrae las características
        df_rpeaks['rr_std_5_beats'] = df_rpeaks['rr_interval'].rolling(window=5, min_periods=1).std()
        
        # 5. Agrega el ID del registro
        df_rpeaks['Data_ID'] = record_id
        
        all_features.append(df_rpeaks)
        
        print(f"Registro {record_name} procesado. Total de latidos: {len(df_rpeaks)}")
        
    except FileNotFoundError:
        print(f"Advertencia: Archivos no encontrados para el registro {record_name}. Saltando...")
        continue
    except Exception as e:
        print(f"Error al procesar el registro {record_name}: {e}")
        continue

# 6. Concatenas todas las características y guardar en un Parquet.
if all_features:
    df_all_features = pd.concat(all_features, ignore_index=True)
    
    output_dir = r"PATH"
    # Crea el directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'holter_features.parquet')
    df_all_features.to_parquet(output_path, engine='pyarrow')
    
    print("\nProcesamiento completado.")
    print(f"Características extraídas y guardadas en: {output_path}")
    print(f"Dimensiones finales del DataFrame: {df_all_features.shape}")
else:
    print("No se pudieron extraer características de ningún registro.")


#Ahora de usar el Partquet optimizado
# Define la ruta del archivo Parquet que acabas de crear
parquet_path = r"PATH"

import pandas as pd
import numpy as np
import os

# Carga el DataFrame
try:
    df_features = pd.read_parquet(parquet_path)
    print("DataFrame de características cargado exitosamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en la ruta: {parquet_path}")
    exit() # Termina el script si no encuentra el archivo

# 2. Carga la información clínica del archivo CSV
df_clinical = pd.read_csv(os.path.join(data_path, 'AdditionalData.csv'))

# 3. Unir los dos DataFrames
# La unión se realiza usando 'Data_ID' como clave, uniendo cada latido con los datos clínicos del paciente.
df_final = pd.merge(df_features, df_clinical, on='Data_ID', how='left')

# 4. Convertir la etiqueta de ritmo 'rhythm_label' a una variable binaria 'is_AFIB'
# Se asume que '(AFIB' es la única anotación relevante pero puede variar segun la base de datos consultada.
df_final['is_AFIB'] = df_final['rhythm_label'].apply(lambda x: 1 if x == '(AFIB' else 0)

# --- Verificación Final ---
print("\nDataFrame final unificado y con etiquetas binarias:")
print(df_final.head())
print(f"Dimensiones del DataFrame final: {df_final.shape}")

# Conteo de la variable objetivo 'is_AFIB'
print("\nConteo de etiquetas binarias 'is_AFIB':")
print(df_final['is_AFIB'].value_counts())

# Porcentaje de cada clase
print("\nPorcentaje de la variable objetivo 'is_AFIB':")
print(df_final['is_AFIB'].value_counts(normalize=True) * 100)


#Limpieza y formato de los datos
# Filtra los latidos con y sin AFIB
df_afib = df_final[df_final['is_AFIB'] == 1]
df_normal = df_final[df_final['is_AFIB'] == 0]

print(f"\nNúmero de latidos con AFIB: {len(df_afib)}")
print(f"Número de latidos normales: {len(df_normal)}")

# Toma una muestra aleatoria de latidos normales, igual al número de latidos con AFIB
num_normal_samples = len(df_afib)
df_normal_sample = df_normal.sample(n=len(df_afib), random_state=50)

# Combina los DataFrames para crear el conjunto de datos de entrenamiento equilibrado
df_balanced = pd.concat([df_afib, df_normal_sample])

# Muestra el conteo de la variable objetivo en el nuevo DataFrame
print("\nConteo del DataFrame equilibrado:")
print(df_balanced['is_AFIB'].value_counts())

# Opcional: Mezcla el DataFrame para asegurar que los datos no están ordenados por clase
df_balanced = df_balanced.sample(frac=1, random_state=50).reset_index(drop=True)

print("\nDataFrame equilibrado listo para el modelado. Primeras 5 filas:")
print(df_balanced.head())


#Preconfiguracion del modelaje oara ML
from sklearn.model_selection import train_test_split

# Defines las características (X) y la variable objetivo (y).
# Comienza con un conjunto básico de características.
features_to_use = [
    'rr_interval', 
    'rr_std_5_beats', 
    'Age_at_Holter', 
    'Sex',
    'HTN',
    'CHF'
]

# Crear la matriz de características (X) y el vector objetivo (y)
X = df_balanced[features_to_use]
y = df_balanced['is_AFIB']

# Es posible que haya valores NaN del cálculo inicial rr_std_5_beats.
# Los rellenaremos con la media o con 0 para preparar los datos para el modelado.
X = X.fillna(X.mean(numeric_only=True))
# La columna «Sexo» es de tipo «objeto», por lo que debemos identificarla con un tipo numérico.
X['Sex'] = X['Sex'].map({'M': 0, 'F': 1})
X['HTN'] = X['HTN'].astype(int)
X['CHF'] = X['CHF'].astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
# Usamos `stratify=y` para garantizar que la proporción de latidos AFIB y normales
# sea la misma en los conjuntos de entrenamiento y prueba.
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, # 30% of the data for testing
    random_state=50,
    stratify=y
)

print("\nDimensiones de los conjuntos de datos:")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


#Podemos agregar mas detalles y caracteriscas para analizar, aqui podemos sugerir elementos de estadistica
# Carga de las librerías necesarias
from scipy.stats import kurtosis, skew
import numpy as np

# Calcula el ritmo cardíaco instantáneo (BPM)
# Se usa np.where para manejar el caso de rr_interval = 0 y evitar divisiones por cero.
df_balanced['BPM'] = np.where(df_balanced['rr_interval'] != 0, 60 / df_balanced['rr_interval'], 0)

# Agrega las características estadísticas
# Vamos a calcular estas estadísticas por cada paciente (Data_ID) para enriquecer el conjunto de datos.
# Aunque el modelo es por latido, las estadísticas del paciente completo pueden ser predictivas.
df_stats = df_balanced.groupby('Data_ID').agg(
    rr_mean=('rr_interval', 'mean'),
    rr_median=('rr_interval', 'median'),
    rr_kurtosis=('rr_interval', lambda x: kurtosis(x, nan_policy='omit')),
    rr_skew=('rr_interval', lambda x: skew(x, nan_policy='omit'))
).reset_index()

# Unir estas estadísticas al DataFrame original.
# Esta unión agregará estas nuevas columnas a cada latido, según el paciente al que pertenece.
df_balanced = pd.merge(df_balanced, df_stats, on='Data_ID', how='left')

# Imprime las nuevas columnas para verificar
print("\nDataFrame con las nuevas características:")
print(df_balanced[['BPM', 'rr_mean', 'rr_median', 'rr_kurtosis', 'rr_skew']].head())


#Para aplicar SMOTE debemos seccionar nuestro DataFrame
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

# Define la lista completa de características a utilizar
features_to_use = [
    'rr_interval', 'rr_std_5_beats', 'Age_at_Holter', 'Sex', 'HTN', 'CHF', 
    'BPM', 'rr_mean', 'rr_median', 'rr_kurtosis', 'rr_skew'
]

# Convierte las columnas a tipo numérico
df_balanced['Sex'] = df_balanced['Sex'].map({'M': 0, 'F': 1})
df_balanced['HTN'] = df_balanced['HTN'].astype(int)
df_balanced['CHF'] = df_balanced['CHF'].astype(int)

# Rellena los valores NaN que puedan haber quedado
df_balanced = df_balanced.fillna(df_balanced.mean(numeric_only=True))

# Crea la matriz de características (X) y el vector objetivo (y)
X = df_balanced[features_to_use]
y = df_balanced['is_AFIB']

# Divide los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=50,
    stratify=y
)

# Aplica SMOTE al conjunto de entrenamiento
smote = SMOTE(random_state=50)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("\nDimensiones de los conjuntos de datos después de SMOTE:")
print(f"X_train_resampled shape: {X_train_resampled.shape}")
print(f"y_train_resampled shape: {y_train_resampled.shape}")


#Si queremos sacar el mejor escenario, considera agregar esta parte para obtener los hiperparametros, esto ayuda tambien para el proceso de la compilacion
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
from sklearn.metrics import f1_score, make_scorer

# Define la cuadrícula de hiperparámetros a buscar
param_grid = {
    'num_leaves': [20, 31, 40],
    'learning_rate': [0.05, 0.1, 0.15],
    'n_estimators': [100, 200, 300],
    'reg_alpha': [0.0, 0.1, 0.5] # Regularización L1
}

# Usa `f1_score` como métrica de evaluación para GridSearchCV, 
# ya que es la más adecuada para problemas de desbalance de clases.
scorer = make_scorer(f1_score)

# Inicializa GridSearchCV
grid_search = GridSearchCV(
    estimator=lgb.LGBMClassifier(random_state=50), 
    param_grid=param_grid, 
    scoring=scorer, 
    cv=5, # Validación cruzada de 5 pliegues
    verbose=1, 
    n_jobs=-1 # Usa todos los núcleos disponibles
)

# Entrena el modelo y busca la mejor combinación de hiperparámetros
grid_search.fit(X_train_resampled, y_train_resampled)

# Muestra los mejores hiperparámetros y el mejor puntaje
print("\nMejores hiperparámetros encontrados:")
print(grid_search.best_params_)
print(f"Mejor F1-Score obtenido en el conjunto de entrenamiento: {grid_search.best_score_:.4f}")

# Utiliza el mejor modelo para predecir en el conjunto de prueba original
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)


#Ya podemos ver los primeros resultados limpios que obtienes de esta configuracion
# Evalúa el modelo optimizado
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

print("\n--- Evaluación del Modelo Optimizado ---")
print(f"Precisión General: {accuracy_score(y_test, y_pred_tuned):.4f}")
print(f"Precisión de la Clase AFIB (1): {precision_score(y_test, y_pred_tuned):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred_tuned))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Los datos de la matriz de confusión que obtuviste
cm = np.array([[196, 43], [45, 193]])

# Crear la visualización
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicción Normal (0)', 'Predicción AFIB (1)'],
            yticklabels=['Real Normal (0)', 'Real AFIB (1)'])
plt.title('Matriz de Confusión del Modelo Optimizado')
plt.ylabel('Etiqueta Real')
plt.xlabel('Etiqueta Predicha')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Asume que 'best_model' es tu modelo LightGBM optimizado de los pasos anteriores.
# Si no lo tienes, vuelve a ejecutar el código de GridSearchCV.
feature_importances = best_model.feature_importances_
feature_names = X.columns # X es tu DataFrame de características

# Crea un DataFrame para facilitar la visualización
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importances})
importance_df = importance_df.sort_values(by='importance', ascending=False)

# Crea la visualización
plt.figure(figsize=(10, 6))
sns.barplot(
    x='importance', 
    y='feature', 
    data=importance_df, 
    palette='viridis',
    hue='feature', # Añade esta línea
    legend=False   # Añade esta línea
)
plt.title('Importancia de las Características del Modelo LightGBM')
plt.xlabel('Importancia (Medida por la ganancia dividida)')
plt.ylabel('Característica')
plt.show()

print("\nImportancia de las características (ordenado de mayor a menor):")
print(importance_df)

