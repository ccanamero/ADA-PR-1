import os
import pandas as pd
import json
from matplotlib import pyplot as plt
import seaborn as sb
from ydata_profiling import ProfileReport

# Rutas de los datasets
ORIGINAL_MAT_DATASET = 'student/student-mat.csv'
ORIGINAL_POR_DATASET = 'student/student-por.csv'

# Set the ggplot style (opcional)
plt.style.use("ggplot")

def cargar_columnas_a_eliminar():
    with open('variables-to-be-cleanned.json', 'r') as file:
        data = json.load(file)
    return data

def cargar_columnas_a_agrupar():
    with open('variables-to-be-grouped.json', 'r') as file:
        data = json.load(file)
    return data

def cargar_y_preparar_dataset(dataset_path: str, tipo_dataset:str):
    # Cargar el dataset original
    df = pd.read_csv(dataset_path, delimiter=';')
    
    # Convertir variables categóricas a variables dummy (one-hot encoding)
    df_dummies = pd.get_dummies(df, drop_first=False)
    
    cleanned_dataset = ''
    if tipo_dataset=='matematicas':
        dummie_dataset = 'mat-dummie.csv'
    else:
        dummie_dataset = 'por-dummie.csv'
            
    df_modificado = eliminar_columnas(df_dummies, dummie_dataset, tipo_dataset, False)
    return df_modificado


def guardar_df(df_modificado: str, tipo_dataset: str, nuevo_archivo: str):
    carpeta_base = 'Datasets_modificados'
    carpeta_tipo = os.path.join(carpeta_base, tipo_dataset)
    
    # Crear la carpeta base y la específica si no existen
    os.makedirs(carpeta_tipo, exist_ok=True)
    
    # Guardar el DataFrame en la ruta especificada
    ruta_completa = os.path.join(carpeta_tipo, nuevo_archivo)
    df_modificado.to_csv(ruta_completa, index=False, sep=";")
    
    print(f"\n > Se ha generado un nuevo archivo en guardado en '{ruta_completa}'")
    
    # Generar reporte en HTML
    

    nombre_report = "Profiling Report - "+nuevo_archivo
    profile_por = ProfileReport(df_modificado, title=nombre_report)
    nombre_analisis = ruta_completa.split(".csv")[0]
    profile_por.to_file(output_file=nombre_analisis)
    
def agregar_media(df_merged, columnas, nueva_columna):
    new_df_merged = df_merged.copy()
    new_df_merged.drop(columnas, axis=1, inplace=True)
    new_df_merged[nueva_columna] = (df_merged[columnas[0]] + df_merged[columnas[1]]) / 2
    
    return new_df_merged

def agregar_maximo(df, columnas, nueva_columna):
    new_df_merged = df.copy()
    new_df_merged.drop(columnas, axis=1, inplace=True)
    new_df_merged[nueva_columna] = df[[columnas[0], columnas[1]]].max(axis=1)
    
    return new_df_merged


# Función para eliminar columnas y guardar el resultado en un nuevo archivo CSV en las carpetas correspondientes
def eliminar_columnas(df, nuevo_archivo: str, tipo_dataset: str, clean: bool):
    
    # Cargar las columnas a eliminar desde el JSON
    columnas_json = cargar_columnas_a_eliminar()
    columnas_a_eliminar = []
    
    if clean:
        columnas_a_eliminar = columnas_a_eliminar + columnas_json['ambos']['Correlacion_con_G3_minima'] + columnas_json[tipo_dataset]['Correlacion_con_G3_minima']
    else: 
        columnas_a_eliminar = columnas_json['ambos']['Causalidad'] 
    
    # Verificamos si las columnas a eliminar existen en el dataset
    columnas_no_existentes = [col for col in columnas_a_eliminar if col not in df.columns]
    
    if columnas_no_existentes:
        print(f"Las siguientes columnas no existen en el dataset: {', '.join(columnas_no_existentes)}")
    
    # Crear una copia del dataframe sin las columnas a eliminar
    df_modificado = df.drop(columns=columnas_a_eliminar, errors='ignore')
    if clean:
        if tipo_dataset == 'portugues':
            df_modificado = agregar_maximo(df_modificado, ['Dalc', 'Walc'], 'Alc')
        df_modificado = agregar_maximo(df_modificado, ['Medu', 'Fedu'], 'Parents_edu')
            
    guardar_df(df_modificado, tipo_dataset, nuevo_archivo)
    return df_modificado

# Función para leer el dataset y mostrar información basada en la opción seleccionada
def read_dataset(dataset_path: str, tipo_dataset: str):
    df = pd.read_csv(dataset_path, delimiter=';')
    
    print("\n¿Qué información te gustaría ver del dataset?")
    print("1. Primeras filas del dataset")
    print("2. Información general del dataset")
    print("3. Estadísticas descriptivas")
    print("4. Limpiar / Agrupar dataset")
    
    while True:
        opcion = input("\nSelecciona una opción (1/2/3/4): ")
        
        if opcion == '1':
            print(df.head()) 
        elif opcion == '2':
            print(df.info()) 
        elif opcion == '3':
            print(df.describe())  
        elif opcion == '4':
            return  
        else:
            print("Opción no válida. Por favor, selecciona 1, 2, 3 o 4.")
            

def limpiar_dataset(df, tipo_dataset: str):
    clean_choice = input("\n ¿Quieres limpiar el dataset? S/N: ")
    
    if clean_choice == "S":
        
        cleanned_dataset = ''
        if tipo_dataset=='matematicas':
            cleanned_dataset = 'mat-cleanned.csv'
        else:
            cleanned_dataset = 'por-cleanned.csv'
        df_modificado = eliminar_columnas(df, cleanned_dataset, tipo_dataset, True)
        
        agrupar_choice = input("\n ¿Quieres agrupar variables? S/N: ")
        if agrupar_choice == 'S' and cleanned_dataset !='':
            if tipo_dataset=='matematicas':
                grouped_dataset = 'mat-grouped-and-cleanned.csv'
            else:
                grouped_dataset = 'por-grouped-and-cleanned.csv'
                
            agregar_variables(df_modificado, tipo_dataset, grouped_dataset)
                
            
def agregar_variables(df, tipo_dataset: str, nuevo_nombre_dataset: str):
    json_config = cargar_columnas_a_agrupar()
    for columna, configuracion in json_config.items():
        if columna not in df.columns:
            print(f"La columna '{columna}' no se encontró en el DataFrame. Terminando el bucle.")
            break
        
        categorias_a_agrupar = configuracion["categorias_a_agrupar"]
        nuevo_nombre = configuracion["nuevo_nombre"]
        df = agrupar_categorias(df, columna, categorias_a_agrupar, nuevo_nombre)
        print(f'{columna} actualizada: ')
        print(df[columna].value_counts())
        
    guardar_df(df, tipo_dataset, nuevo_nombre_dataset)
    
        
def agrupar_categorias(dataset, columna, categorias_a_agrupar, nuevo_nombre):
    """
    Agrupa categorías en una columna de un dataset en una nueva categoría y mantiene las demás igual.
    
    Parámetros:
        - dataset (pd.DataFrame): El dataset original.
        - columna (str): Nombre de la columna que contiene las categorías.
        - categorias_a_agrupar (list): Lista de categorías que deseas agrupar.
        - nuevo_nombre (str): Nombre de la nueva categoría que agrupará a las seleccionadas.
    
    Retorno:
        - pd.DataFrame: El dataset modificado con las categorías agrupadas.
    """
    dataset_modificado = dataset.copy()

    freq_agregadas = dataset_modificado[dataset_modificado[columna].isin(categorias_a_agrupar)].shape[0]

    dataset_modificado[columna] = dataset_modificado[columna].apply(
        lambda x: nuevo_nombre if x in categorias_a_agrupar else x
    )
    
    return dataset_modificado
        
def elegir_dataset():

    while True:
        mostrar_menu()
        
        dataset_choice = input("Selecciona una opción (1/2/3): ")
        if dataset_choice == '1':
            procesar_dataset('matematicas', ORIGINAL_MAT_DATASET)
        elif dataset_choice == '2':
            procesar_dataset('portugues', ORIGINAL_POR_DATASET)
        elif dataset_choice == '3':
            print("Saliendo...")
            break
        else:
            print("Opción no válida. Por favor, selecciona 1, 2 o 3.")

def mostrar_menu():

    print("\nElige el dataset con el que quieras trabajar:")
    print("1. Student Math Dataset")
    print("2. Student Portuguese Dataset")
    print("3. Salir")
    

def procesar_dataset(tipo_dataset, dataset_path):
    
    print(f"\nProcesando dataset de {tipo_dataset.capitalize()}...")
    read_dataset(dataset_path, tipo_dataset)  # Llama a la función read_dataset con el tipo de dataset adecuado
    df = cargar_y_preparar_dataset(dataset_path, tipo_dataset)   # Carga y prepara el dataset
    limpiar_dataset(df, tipo_dataset)             # Limpia el dataset según el tipo
    
    

    

# Función principal para seleccionar el dataset
if __name__ == "__main__":

   elegir_dataset()
