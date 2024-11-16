import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, make_scorer
from sklearn.inspection import PartialDependenceDisplay
from scipy.stats import probplot
from scipy import stats
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import json
import os

# Configuración de gráficos
sns.set(style="whitegrid")

# Rutas de los datasets
DUMMIE_MAT_DATASET = 'Datasets_modificados/matematicas/mat-dummie.csv'
DUMMIE_POR_DATASET = 'Datasets_modificados/portugues/por-dummie.csv'
CLEANED_MAT_DATASET = 'Datasets_modificados/matematicas/mat-cleanned.csv'
CLEANED_POR_DATASET = 'Datasets_modificados/portugues/por-cleanned.csv'
GROUPED_AND_CLEANED_MAT_DATASET = 'Datasets_modificados/matematicas/mat-grouped-and-cleanned.csv'
GROUPED_AND_CLEANED_POR_DATASET = 'Datasets_modificados/portugues/por-grouped-and-cleanned.csv'

def seleccionar_dataset():
    print("\nElige el dataset con el que quieras trabajar:")
    print("1. Matematicas")
    print("2. Portugues")
    print("3. Salir")

    choice = input("Selecciona una opción (1/2/3): ")
    
    dataset_paths = {
        'original': DUMMIE_MAT_DATASET if choice == '1' else DUMMIE_POR_DATASET,
        'cleaned': CLEANED_MAT_DATASET if choice == '1' else CLEANED_POR_DATASET,
        'grouped_and_cleanned': GROUPED_AND_CLEANED_MAT_DATASET if choice == '1' else GROUPED_AND_CLEANED_POR_DATASET
    }
    
    if choice == '3':
        print("Saliendo...")
        exit()
    if choice != '1' and choice!='2':
        print("Opción no válida.")
        exit()
    else:
        return dataset_paths

def cargar_y_preparar_datos(dataset_path):
    df = pd.read_csv(dataset_path, delimiter=';')
    y = df['G3']
    X = df.drop(columns=['G3'])
    return X, y
    

from sklearn.model_selection import learning_curve
import numpy as np
import matplotlib.pyplot as plt

def mostrar_curva_de_aprendizaje(model, X, y, version_dataset, cv=5):
    # Obtenemos los tamaños de entrenamiento, los puntajes de entrenamiento y los de validación
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_squared_error'
    )
    
    # Convertimos los errores negativos MSE a RMSE
    train_rmse = np.sqrt(-train_scores.mean(axis=1))
    val_rmse = np.sqrt(-val_scores.mean(axis=1))
    
    # Graficamos la curva de aprendizaje
    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_rmse, 'o-', label="Entrenamiento")
    plt.plot(train_sizes, val_rmse, 'o-', label="Validación")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("RMSE")
    plt.legend()
    plt.title(f'Dataset {version_dataset.capitalize()} - Curva de Aprendizaje')
    plt.show()

    

def graficar_comparacion(all_y_test_and_pred, model_type):
    
    # Gráfica de Predicciones vs Reales
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (dataset_name, (all_y_pred, all_y_test)) in enumerate(all_y_test_and_pred.items()):
        
        slope, intercept, r, p, std_err = stats.linregress(all_y_pred, all_y_test)
        line = slope * all_y_test + intercept
        axs[idx].scatter(all_y_test, all_y_pred, alpha=0.5, label="Predicciones")
        axs[idx].plot(all_y_test, line, 'r--', lw=2, label=f'Regresión (slope={slope:.2f}, intercept={intercept:.2f})')
        axs[idx].set_title(f'Dataset {dataset_name.capitalize()}') # - Predicciones vs Valores Reales (Consolidado de Folds)')
        axs[idx].set_xlabel('Valores Reales')
        axs[idx].set_ylabel('Predicciones')
        axs[idx].legend()
    plt.suptitle(f'Predicciones vs Valores Reales')
    plt.show()
    
    # Histograma de Residuales
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    for idx, (dataset_name, (all_y_pred, all_y_test)) in enumerate(all_y_test_and_pred.items()):
        residuales = all_y_test - all_y_pred
        axs[idx].hist(residuales, bins=30, edgecolor='k', alpha=0.7)
        axs[idx].set_xlabel('Error Residual')
        axs[idx].set_ylabel('Frecuencia')
        axs[idx].set_title(f'{dataset_name.capitalize()} - Histograma de Residuales')
    plt.suptitle(f'Comparación de Histogramas de Residuales')
    plt.show()
    
        


def cross_validation_metrics(model, X, y, version_dataset):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores, rmse_scores, mae_scores, r2_scores = [], [], [], []
    X = X.values 
    y = y.values 
    
    # Listas para almacenar todas las predicciones y valores reales de todos los folds
    all_y_test, all_y_pred = [], []

    print(f"DATASET\n{'Fold':<10} {'MSE':<10} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    
    for fold, (train_index, test_index) in enumerate(kf.split(X), start=1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Entrenar el modelo
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Almacenar los valores reales y predicciones de este fold
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

        # Calcular las métricas (métricas negativas por ser 'neg_mean_squared_error')
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Guardar resultados
        mse_scores.append(mse)
        rmse_scores.append(rmse)
        mae_scores.append(mae)
        r2_scores.append(r2)

        # Imprimir el resultado de cada métrica para el fold actual (con el signo corregido)
        print(f"Fold {fold:<4}: {mse:<10.4f} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")
        


    # Promediar las métricas y devolver el diccionario de resultados
    metrics_dict = {
        'MSE': [
            np.mean(mse_scores),
            np.std(mse_scores)
        ],
        'RMSE': [
            np.mean(rmse_scores),
            np.std(rmse_scores)
        ],
        'MAE': [
            np.mean(mae_scores),
            np.std(mae_scores)
        ],
        'R²': [
            np.mean(r2_scores),
            np.std(r2_scores)
        ]
    }
    
    # Imprimir resultados totales con la media ± desviación estándar
    print("\n" + " " * 8 + f"MSE (mean ± std)       RMSE (mean ± std)        MAE (mean ± std)       R² (mean ± std)")
    print(f"TOTAL  {metrics_dict['MSE'][0]:<10.4f} ± {metrics_dict['MSE'][1]:<10.4f} "
          f"{metrics_dict['RMSE'][0]:<10.4f} ± {metrics_dict['RMSE'][1]:<10.4f} "
          f"{metrics_dict['MAE'][0]:<10.4f} ± {metrics_dict['MAE'][1]:<10.4f} "
          f"{metrics_dict['R²'][0]:<10.4f} ± {metrics_dict['R²'][1]:<10.4f}")
    
    
    # Convertir listas a arrays de NumPy para facilitar cálculos posteriores
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    
    graficar_comparacion(all_y_test, all_y_pred, version_dataset)
    return model, metrics_dict


def graficar_comparacion(all_y_test, all_y_pred, version_dataset):
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Gráfica de Predicciones vs Reales
    slope, intercept, r, p, std_err = stats.linregress(all_y_pred, all_y_test)
    line = slope * all_y_test + intercept
    
    axes[0].scatter(all_y_test, all_y_pred, alpha=0.5, label="Predicciones")
    axes[0].plot(all_y_test, line, 'r--', lw=2, label=f'Regresión (slope={slope:.2f}, intercept={intercept:.2f})')
    axes[0].set_title(f'Dataset {version_dataset.capitalize()} - Predicciones vs Valores Reales (Consolidado de Folds)')
    axes[0].set_xlabel('Valores Reales')
    axes[0].set_ylabel('Predicciones')

    # Histograma de Residuales
    residuales = all_y_test - all_y_pred
    axes[1].hist(residuales, bins=30, edgecolor='k', alpha=0.7)
    axes[1].set_xlabel('Error Residual')
    axes[1].set_ylabel('Frecuencia')
    axes[1].set_title(f'Dataset {version_dataset.capitalize()} - Histograma de Residuales (Consolidado de Folds)')
    
    plt.show()
    
def calcular_coeficientes(X, modelo, tipo_regresion):
    if tipo_regresion=='lineal':
        coeficientes = modelo.coef_
    else:
        coeficientes = modelo.coef_[0]
    variables = X.columns
    coef_df = pd.DataFrame({'Variable': variables, 'Coeficiente': coeficientes})
    coef_df = coef_df.sort_values(by='Coeficiente', ascending=True).reset_index(drop=True)
    return coef_df


def generar_reporte_json(dataset_path, model_type ,coef_df):
    
    reporte = {
        "dataset_path": dataset_path,
        "modelo": model_type,
        "variables": [
            {"variable": row["Variable"], "coeficiente": row["Coeficiente"]}
            for _, row in coef_df.iterrows()
        ]
    }
    
    reporte_json = json.dumps(reporte, indent=4)
    carpeta = dataset_path.split("/")[0]+'/'+dataset_path.split("/")[1]+'/'
    os.makedirs(carpeta, exist_ok=True)

    ruta_archivo = os.path.join(carpeta, f"{os.path.basename(dataset_path).replace('.csv', '')}_"+model_type+"_reporte.json")

    with open(ruta_archivo, 'w') as archivo:
        archivo.write(reporte_json)
    
    print(f"Se ha generado un reporte JSON guardado en {ruta_archivo}")
    

def main():
    
    resultados = {}
    y_tests_preds = {}
    coeficientes = {}
    interceptados = {}
    metricas = {}
    
    print("\nElige el tipo de regresión que deseas usar:")
    print("1. Regresión Lineal")
    print("2. Regresión Logística")

    model_choice = input("Selecciona una opción (1/2): ")
    dataset_paths = seleccionar_dataset()
    for key, dataset_path in dataset_paths.items():
        X, y = cargar_y_preparar_datos(dataset_path)

        if model_choice == '1':
            model = LinearRegression()
            model, scores = cross_validation_metrics(model, X, y, key)
            coef_df = calcular_coeficientes(X, model, 'lineal')
            generar_reporte_json(dataset_path, 'lineal', coef_df)

            
        elif model_choice == '2':
            model = LogisticRegression(max_iter=5000)
            model, scores = cross_validation_metrics(model, X, y, key)
            coef_df = calcular_coeficientes(X, model, 'logistica')
            generar_reporte_json(dataset_path, 'logistica', coef_df)
        else:
            print("Opción no válida.")
    
            

if __name__ == "__main__":
    main()

