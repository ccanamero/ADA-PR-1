# ADA-PR-1


## 1. Build conda environment

```
conda create --name pr1
conda activate pr-1
```

## 2. Install dependencies

```
pip install numpy pandas matplotlib scikit-learn
pip install spacy
pip install sentiment-analysis-spanish
python -m spacy download es_core_news_sm
```

## 3. Import dataset
```
pip install ucimlrepo
```

## 4. Work

### Dataset analysis

1. Dataset description
2. Dataset analysis: exploratorio, distribución y ?
3. Eliminación de variables y rsquare variable. Me quedo con las variables con las que el p_value sea ridículo.
3. Regresión Logística y Clasificación


-------
NO dependen de la materia:
- school;sex;age;address;famsize;Pstatus;Medu;Fedu;Mjob;Fjob;reason;internet;nursery; 
- DIVIDIDAS PERO NO DEBERÏA: guardian_mat;traveltime_mat;romantic_por;famrel_por;freetime_por;goout_por;Dalc_por;Walc_por;health_por;romantic_mat;famrel_mat;health_mat;guardian_por;activities_mat;freetime_mat;higher_mat;goout_mat;Dalc_mat;Walc_mat;famsup_mat;famsup_por;

DIVIDIDAS y deberían: 
- G1_por;G2_por;G3_por; G1_mat;G2_mat;G3_mat;
- Si diferencia entre asignaturas: paid_mat;paid_por;


