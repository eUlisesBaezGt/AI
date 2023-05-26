#!/usr/bin/env python
# coding: utf-8

# # 5-number summary
# 
# This tutorial is meant to help to load, visualise and analyse datasets in order to prepare them for the data pre-processing

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#to save the plots to a pdf file
from matplotlib.backends.backend_pdf import PdfPages


# ## General variables configuration

# In[2]:


# dataset name and location
dataset_folder = 'Bristol_owd.csv'

# results folder
res_folder = 'results/'

# selecting the features and the time column
# only the features related to the weather that are measurements are selected. That is, features that contains
#  _id, _main, _description, and _icon are removed since they are use by open weather internally
# the city name, lat, lon, and timezone can be safely removed since we have the files separated by city
# we have two times format (dt and dt_iso). We select dt since it is the time in numeric format
fetures_to_preserve = ['temp', 'temp_min', 'temp_max', 'feels_like', 'pressure', 'humidity', 'wind_speed', 'wind_deg',
                       'clouds_all', 'Rain_id', 'Rain_main', 'Rain_description']

# features to ignore when performing the frequency counts
ignore_features = ['temp_min', 'temp_max']

######################################## Global variables ########################################
# original datasets
original_data = None

# datasets with the features of interest
new_dataset = None

# summary of nan values per feature
nan_summary = {}

# Give format to the features names as appears in the paper proposal
formated_features_names = []


# ## Functions

# In[3]:


# Load all the datasets
def load_dataset():
    global original_data
    original_data = pd.read_csv(dataset_folder, sep=",", on_bad_lines="skip", dtype="str")


# Generate the new dataset
def build_new_dataset():
    global new_dataset
    global original_data
    #format the new structure of the data(dataframe)
    data_format = pd.DataFrame()
    # append the features previously selected (without the time column)
    data_format = original_data[fetures_to_preserve]
    # append the column time
    data_format.index = original_data['dt']
    new_dataset = data_format


# Compute a summary for all nan values per feature per dataset
def compute_nan_summary(dataset):
    global nan_summary
    print('Computing Nan summary')
    summary = {}
    for feature in list(dataset.columns):
        na_total = dataset[feature].isnull().sum()
        na_percentage = (na_total * 100) / len(dataset.index)

        summary[feature] = [na_total, na_percentage]

    nan_summary = pd.DataFrame(summary, columns=list(dataset.columns), index=['Na No', 'Na %'])


# Write the nan summary resutls as a Excel file
def write_nan_summary():
    with pd.ExcelWriter(res_folder + 'Nan_summary.xlsx') as writer:
        nan_summary.to_excel(writer, sheet_name='values')
        # writer.save()


# Plot the behaviour of the features across time
def plot_feats_across_time(dataset):
    print('Ploting features behaviour')

    #transform each column type of the dataset to numeric(float) to perform the analysis
    dataset = dataset.iloc[:, :-3]
    dataset = dataset.astype(float)

    plots_output = res_folder + 'features_across_time.pdf'
    #taking a look of the features variation across time
    plt.rcParams["figure.figsize"] = (20, 30)
    dataset.plot(subplots=True)
    plt.savefig(plots_output)
    plt.close('all')


# Compute the 5-number summary
def five_number_summary(dataset):
    print('Computing the 5-number summary')

    #transform each column type of the dataset to numeric(float) to perform the analysis
    dataset = dataset.iloc[:, :-3]
    dataset = dataset.astype(float)

    with pd.ExcelWriter(res_folder + '5_number_summary.xlsx') as writer:
        #minimum
        dataset_min = dataset.min()
        #lower quartile
        dataset_lq = dataset.quantile(.25, axis=0)
        #median
        dataset_med = dataset.median()
        #uper quartile
        dataset_uq = dataset.quantile(.75, axis=0)
        #maximum
        dataset_max = dataset.max()

        summary = pd.concat([dataset_min.rename("Min"),
                             dataset_lq.rename("Lower quartile"),
                             dataset_med.rename("Median"),
                             dataset_uq.rename("Uper quartile"),
                             dataset_max.rename("Max")], axis=1)

        summary.to_excel(writer, sheet_name='values')


# Plot boxplots and histograms per feature
def plot_boxplots_histograms(dataset):
    print('Ploting boxplots and histograms')
    #transform each column type of the dataset to numeric(float) to perform the analysis
    dataset = dataset.iloc[:, :-3]
    dataset = dataset.astype(float)

    plots_output = res_folder + 'box_plots_&_histograms.pdf'
    pdf = PdfPages(plots_output)

    plt.rcParams["figure.figsize"] = (20, 8)

    for feature in list(dataset.columns):
        fig, axes = plt.subplots(1, 2)
        dataset.boxplot(column=feature, ax=axes[0])
        dataset.hist(column=feature, ax=axes[1])
        pdf.savefig(fig)

    pdf.close()
    plt.close('all')


# Plot all histograms of all features per dataset together
def plot_histograms(dataset):
    print('Ploting histograms')
    #transform each column type of the dataset to numeric(float) to perform the analysis
    dataset = dataset.iloc[:, :-3]
    dataset = dataset.astype(float)

    plots_output = res_folder + 'all_histograms.pdf'
    plt.rcParams["figure.figsize"] = (20, 15)
    dataset.hist()
    plt.savefig(plots_output)
    plt.close('all')


# Compute the frequency counts per feature per dataset
def freq_counts(dataset):
    print('Computing frequency counts')

    with pd.ExcelWriter(res_folder + 'frequency_counts.xlsx') as writer:
        #transform each column type of the dataset to numeric(float) to perform the analysis
        dataset = dataset.iloc[:, :-3]
        dataset = dataset.astype(float)

        for feature in list(dataset.columns):
            if feature not in ignore_features:
                min_val = dataset[feature].min()
                max_val = dataset[feature].max()
                # min_val = min_val.round(0).astype(int)
                min_val = int(round(min_val))
                # max_val = max_val.round(0).astype(int)
                max_val = int(round(max_val))

                ranges = []
                for i in range(min_val, max_val + 2):
                    ranges.append(i)

                temp_feat_df = dataset[feature]
                counts = temp_feat_df.groupby(pd.cut(temp_feat_df, ranges)).count()

                #counts = data[feature].value_counts(bins=((max_val - min_val)+1), dropna=True, sort=True)#, ascending=True)
                counts.to_excel(writer, sheet_name=feature)


# Generate all the analysis plots
def analysis_plots(dataset):
    # revise for NaN values
    compute_nan_summary(dataset)

    # write nan summary results for each dataset/city
    write_nan_summary()

    # generate the plots to see features behaviour across time
    plot_feats_across_time(dataset)

    # generate the 5-number summary
    five_number_summary(dataset)

    # plot the boxplots and histograms
    plot_boxplots_histograms(dataset)

    # plot the histograms together
    plot_histograms(dataset)

    # compute the frequency counts
    freq_counts(dataset)


# Plot nullness/nans percentages of the features in a dataset
def plot_nullness():
    global nan_summary

    plt.rcParams['axes.labelsize'] = 13
    plt.rcParams['axes.labelweight'] = 'bold'

    plots_output = res_folder + 'nullness.pdf'
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(15, 25), dpi=80)
    fig.subplots_adjust(hspace=.07, wspace=.05)

    ax.scatter(y=nan_summary.columns, x=nan_summary.iloc[1, :], color='purple', alpha=1)
    ax.set_xlabel('Nullness %')
    ax.set_yticks(nan_summary.columns)
    ax.set_yticklabels(nan_summary.columns, fontdict={'horizontalalignment': 'right'}, size=13, weight='bold')
    ax.set_xlim(-5, 105)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_title('Bristol', size=16, weight='bold')

    plt.tight_layout()
    plt.show()
    fig.savefig(plots_output)


# ## Load the dataset and visualise it

# In[4]:


load_dataset()


# In[5]:


# get a quick view of Bristol data
original_data.head()


# In[6]:


original_data['dt_iso']


# In[7]:


#see the name of the columns
original_data.columns


# ## Data pre-processing

# In[8]:


# Generate the new dataset
build_new_dataset()


# In[9]:


# view the dataset processed
new_dataset.head()


# In[10]:


# Perform all the analysis
analysis_plots(new_dataset)


# In[11]:


# Plot features nullness(nans)
plot_nullness()


# ## Ajustes

# In[12]:


new_dataset.iloc[4502, 9]  # Búscate uno de los que si tienen número


# In[13]:


columnas = ['Rain_id', 'Rain_main', 'Rain_description']


# In[19]:


# 1) Completar los valores de las celdas faltantes con el valor promedio de los registros respectivos de su columna
def reemplazar_con_promedio(dataset):
    dataset_copy = dataset.copy()
    # Conseguir el valor promedio de cada columna
    # De la columna Rain_id, ignorar todos los valores que no sean nulos, todos los que si tengan valor (convertir a float) para obterner el promedio
    valores_columna = dataset_copy['Rain_id'].dropna().astype(float)
    promedio = valores_columna.mean()
    print(f"Promedio: {promedio}")
    # Reemplazar los valores nulos de la columna Rain_id con el promedio
    dataset_copy['Rain_id'].fillna(promedio, inplace=True)
    return dataset_copy


new_dataset_promedio = reemplazar_con_promedio(new_dataset)
print("Reemplazando por media")
print(new_dataset_promedio)
print("====================")

analysis_plots(new_dataset_promedio)


# In[20]:


# Para las columnas categóricas, es posible que desee utilizar la moda (el valor más frecuente):
def reemplazar_con_moda(dataset, columnas):
    dataset_copy = dataset.copy()
    for columna in columnas:
        if dataset_copy[columna].dtype == object:
            valor_moda = dataset_copy[columna].mode()[0]
            dataset_copy[columna].fillna(valor_moda, inplace=True)
    return dataset_copy


new_dataset_moda = reemplazar_con_moda(new_dataset, columnas)
print("Reemplazando por moda")
print(new_dataset_moda)
print("====================")

analysis_plots(new_dataset_moda)


# In[22]:


# Combinar los anteriores dos métodos
def reemplazar_con_promedio_moda(dataset, columnas):
    dataset_copy = dataset.copy()
    for columna in columnas:
        if dataset_copy[columna].dtype == object:
            valor_moda = dataset_copy[columna].mode()[0]
            dataset_copy[columna].fillna(valor_moda, inplace=True)
        else:
            valores_columna = dataset_copy[columna].dropna().astype(float)
            promedio = valores_columna.mean()
            dataset_copy[columna].fillna(promedio, inplace=True)
    return dataset_copy


new_dataset_promedio_moda = reemplazar_con_promedio_moda(new_dataset, columnas)
print("Reemplazando por híbrido")
print(new_dataset_promedio_moda)
print("====================")

analysis_plots(new_dataset_promedio_moda)

# 1) Completar los valores de las celdas faltantes con el valor promedio de los registros respectivos 
# de su columna o en caso de ser 'strings' poner el valor de texto que se presenta con mayor frecuencia
# Este enfoque es adecuado cuando son numéricos, pero no para los strings, ya que no se puede calcular el promedio,
# por lo que se debe usar el valor que más se repite. Se sugiere usar cuando los valores faltantes son pocos.


# In[17]:


# 2) Completar los valores de las celdas faltantes con el valor de la celda más cercana de su respectiva columna
def reemplazar_con_cercano(dataset, columnas):
    dataset_copy = dataset.copy()
    for columna in columnas:
        dataset_copy[columna].fillna(method='ffill', inplace=True)
    return dataset_copy


new_dataset_cercano1 = reemplazar_con_cercano(new_dataset, columnas)


# Volver a revisar si quedan nulos y avanzar hasta entonctrar un valor no nulo y ponerlo en todas las filas anteriores
# Iterar sobre las columnas de interés y y buscar el primer valor no nulo
def primer_no_nulo(dataset, columnas):
    dataset_copy = dataset.copy()
    for columna in columnas:
        for i in range(len(dataset_copy[columna])):
            if not pd.isnull(dataset_copy[columna][i]):
                return dataset_copy[columna][i]


primer_valor_no_nulo = primer_no_nulo(new_dataset, columnas)


# Iterar sobre las columnas de interés y y ya teniendo el primer valor no nulo, reemplazar los nulos por ese valor
def reemplazar_nulos(dataset, columnas, valor):
    dataset_copy = dataset.copy()
    for columna in columnas:
        dataset_copy[columna].fillna(valor, inplace=True)
    return dataset_copy


new_dataset_cercano = reemplazar_nulos(new_dataset_cercano1, columnas, primer_valor_no_nulo)

print("Reemplazando por cercano")
print(new_dataset_cercano)
print("====================")

analysis_plots(new_dataset_cercano)
# 2) Completar los valores de las celdas faltantes con el valor de la celda más cercana de su respectiva 
# columna
# Este enfoque es adecuado cuando los valores faltantes son pocos y se encuentran agrupados, ya que se puede
# reemplazar por el valor anterior o posterior, así se mantiene la tendencia de los datos. Se sugiere usar 
# cuando los valores faltantes son pocos.


# In[18]:


# 3) Eliminar todos los registros/observaciones que presente al menos un valor NaN en cualquiera de las características/columnas
def eliminar_filas_nan(dataset, columnas):
    dataset_copy = dataset.copy()
    dataset_copy.dropna(subset=columnas, inplace=True)
    return dataset_copy


new_dataset_sin_nan = eliminar_filas_nan(new_dataset, columnas)
print("Eliminando valores nulos")
print(new_dataset_sin_nan)
print("====================")

# 3) Eliminar todos los registros/observaciones que presente al menos un valor NaN en cualquiera de 
# las características/columnas
# Este enfoque es adecuado cuando los valores faltantes son muchos, ya que se eliminan los registros que
# presentan valores faltantes, y así se evita que se altere la tendencia de los datos. Se sugiere usar
# cuando los valores faltantes son muchos, pero se debe tener cuidado de no eliminar muchos registros, ya
# que se puede perder información importante.


# ##### Elaborado por Enrique Gómez Tagle e Ivanna Tinedo
# Fecha: 25/05/2023 \
# Versión: 1.0 \
# Inteligencia Artificial \
# Profesor: Ari Yair Barrera Ánimas
#  
# PRÁCTICA: Preprocesamiento de datos
