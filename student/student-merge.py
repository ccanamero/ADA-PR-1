import pandas as pd

# Leer los archivos CSV
d1 = pd.read_csv("student-mat.csv", sep=";")
d2 = pd.read_csv("student-por.csv", sep=";")

# Combinar los DataFrames en base a las columnas especificadas
d3 = pd.merge(
    d1, d2,
    on=["school", "sex", "age", "address", "famsize", "Pstatus", 
        "Medu", "Fedu", "Mjob", "Fjob", "reason", "internet", "nursery"]
)

# Renombrar columnas con sufijos _x y _y
d3.rename(columns=lambda x: x.replace('_x', '_mat').replace('_y', '_por'), inplace=True)

# Listas de columnas a combinar porque no dependen de la materia
variables_a_combinar = [
    ('guardian', 'guardian_mat', 'guardian_por'),
    ('traveltime', 'traveltime_mat', 'traveltime_por'),
    ('romantic', 'romantic_mat', 'romantic_por'),
    ('famrel', 'famrel_mat', 'famrel_por'),
    ('freetime', 'freetime_mat', 'freetime_por'),
    ('goout', 'goout_mat', 'goout_por'),
    ('Dalc', 'Dalc_mat', 'Dalc_por'),
    ('Walc', 'Walc_mat', 'Walc_por'),
    ('health', 'health_mat', 'health_por'),
    ('studytime', 'studytime_mat', 'studytime_por'),
    ('failures', 'failures_mat', 'failures_por'),
    ('schoolsup', 'schoolsup_mat', 'schoolsup_por'),
    ('famsup', 'famsup_mat', 'famsup_por'),
    ('higher', 'higher_mat', 'higher_por'),
    ('absences', 'absences_mat', 'absences_por'),
    ('activities', 'activities_mat', 'activities_por')
]

for new_col, col_mat, col_por in variables_a_combinar:
    d3[new_col] = d3[[col_mat, col_por]].bfill(axis=1).iloc[:, 0]

columns_to_drop = [col_mat for _, col_mat, col_por in variables_a_combinar] + \
                  [col_por for _, col_mat, col_por in variables_a_combinar]

d3.drop(columns=columns_to_drop, inplace=True)

d3.to_csv("merged_students.csv", index=False, sep=";")

# Mostrar la cantidad de filas (estudiantes) en el DataFrame resultante
print(len(d3)) 
print(len(d3.columns))
