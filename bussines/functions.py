import pandas as pd
import numpy as np
import sys
from os import listdir
from os.path import isfile, join

from scipy import stats
import matplotlib.pyplot as plt

class Functions():
    
    def __init__(self, path, test):
        self.path = path
        self.test = test
    
    # Modulo: Extraccion de los archivos dentro de una direccion
    def get_files_names(self):
        return [file for file in listdir(self.path) if (isfile(join(self.path, file))) & (file.endswith('.csv'))]

    # # Modulo: Visualizacion y seleecion de los archivos de una fuente
    def select_file(self):
        files = self.get_files_names()
        if len(files) == 0:
            print(" >> Error: No existen archivos en la direccion: {}".format(self.path))
            sys.exit()

        print("\n >>> Selecione el archivo (CSV) analizar <<< \n")
        for idx, file in enumerate(files):
            print(" > {}.- {}".format(idx + 1, file.split(".")[0]))

        while True:
            try:
                index = int(input('\n Ingrese el indice del archivo: '))
                if (index > 0) & (index <= len(files)):
                    index -= 1
                    name_file = files[index]
                    break

                else:
                    print("\n Opcion invalida !! \n")
                    
            except:
                print("\n >> Error: Ingrese un valor numerico !!! ")

        print("---"*20)

        return name_file

    # Modulo: Determinacion de la extraccion de fuente de datos
    def select_source_data(self):
        print("\n >>> Selecione la fuente de datos <<< \n\n 1.- Archivo \n 2.- BD \n")
        while True:
            try:
                option_file = int(input('Ingrese # opcion: '))    
                if (option_file > 0) & (option_file < 3):
                    break

                else:
                    print("\n Opcion invalida !! \n")

            except:
                print("\n >> Error: Ingrese un valor numerico !!! ")

        print("---"*20)

        return option_file

    # Modulo: Visualizacion de variables
    def visual_variables(self, columns):
        for idx, col_name in enumerate(columns):
            print(" > {} - {}".format(idx + 1, col_name))

        print(" > {} - Salir\n".format(len(columns) + 1))

    # Modulo: Determinacion de las variables analizar de la fuente de datos seleccionada
    def select_variables_data(self, data):
        print("\n >>> Selecione las variables analizar <<< \n")
        columns = data.columns.tolist()
        self.visual_variables(columns)

        list_col = []
        total = len(columns) + 1
        while True:
            try:
                x = int(input('Ingrese numero de columna: '))
                if x == total:
                    break

                elif (x < 1) & (x > total):
                    print("\n Indice incorrecto !!! \n")

                elif x in list_col:
                    print("\n Indice duplicado !!! \n")

                else:
                    list_col.append(x)

            except:
                print("\n >> Error: Ingrese un valor numerico !!! ")

        col_name = []
        for index in list_col:
            index -= 1
            col_name.append(columns[index])

        for col in col_name:
            columns.remove(col)

        print("---"*20)

        return col_name, columns

    # Modulo: Determinacion del periodo de analisis para el computo del intervalo medio de demanda (Semanal o mensual)
    def select_period(self):
        print("\n >>> Selecione los periodos analizar <<< \n\n 1.- Semana \n 2.- Mes \n")
        while True:
            try:
                period = int(input('Ingrese # opcion: '))    
                if period == 1:
                    period = "week"
                    break

                elif period == 2:
                    period = "month"
                    break

                else:
                    print("\n Opcion invalida !! \n")

            except:
                print("\n >> Error: Ingrese un valor numerico !!! ")

        print("---"*20)

        return period

    # Modulo: determinacion de las variables (Tiempo y observacion) para el analisis de series de tiempo y computo del intervalo medio de demanda
    def select_var_series(self, data):
        print("\n >>> Seleccion variable tiempo y observacion <<< \n")
        columns = data.columns.tolist()
        for idx, col_name in enumerate(columns):
            print(" > {} - {}".format(idx + 1, col_name))

        print()
        col_name = []
        total = len(columns)
        while True:
            try:
                x = int(input('Ingrese numero de columna (Tiempo): '))    
                if (x < 1) & (x > total):
                    print("\n Indice incorrecto !!! \n")

                else:
                    col_name.append(x)
                    break

            except:
                print("\n >> Error: Ingrese un valor numerico !!! ")

        while True:
            try:
                x = int(input('Ingrese numero de columna (Observacion): '))    
                if (x < 1) & (x > total):
                    print("\n Indice incorrecto !!! \n")

                else:
                    col_name.append(x)
                    break

            except:
                print("\n >> Error: Ingrese un valor numerico !!! ")

        col_serie = []
        for index in col_name:
            index -= 1
            col_serie.append(columns[index])

        print("---"*20)

        return col_serie

    # Modulo: Determinacion de la granularidad de los datos
    def select_gran_data(self, data):
        print("\n >>> Seleccion la granularidad <<< \n")
        columns = data.columns.tolist()
        self.visual_variables(columns)

        col_name = []
        total = len(columns) + 1
        while True:
            try:
                x = int(input('Ingrese numero de columna: '))    
                if x == total:
                    break

                elif (x < 1) & (x > total):
                    print("\n Indice incorrecto !!! \n")

                elif x in col_name:
                    print("\n Indice duplicado !!! \n")

                else:
                    col_name.append(x)

            except:
                print("\n >> Error: Ingrese un valor numerico !!! ")

        col_gran = []
        for index in col_name:
            index -= 1
            col_gran.append(columns[index])

        print("---"*20)

        return col_gran

    # Modulo: # Analisis y compute del coeficiente de variacion - CV2
    def get_cv(self, data, columns, potencia = 2):
        #cv = round(pow(data.Demand.std() / data.Demand.mean(), 2), 2)
        # Agrupamiento de valores en base a la variables de observacion seleccionada
        data_retail = data.groupby(columns).agg(sales = (columns[0], 'sum'), count_sales = (columns[0], 'count')).reset_index()
        for col in data_retail[columns[:-1]]:
            data_retail[col] = data_retail[col].astype(str)

        # Definicion de la etiqueta en base a la granularidad de las variables seleccionadas
        data_retail["label"] = data_retail[columns[:-1]].apply("_".join, axis = 1)

        print(data_retail.head())
        print("---"*20)

        # Computo de std y promedio de las ventas por cada valor de la columna analizar
        cv_data = data_retail.groupby("label").agg(std = ('sales', 'std'), mean = ('sales', 'mean')).reset_index()
        cv_data["mean"] = cv_data["mean"].round(2)
        cv_data["std"] = cv_data["std"].round(2)

        # Computo de coeficiente de variacion
        cv_data['cv'] = cv_data['std'] / cv_data['mean']
        cv_data["cv"] = cv_data["cv"].round(4)

        # Computo del coeficiente de variacion a la potencia
        cv_data['cv2'] = pow(cv_data['cv'], potencia)
        cv_data["cv2"] = cv_data["cv2"].round(4)
        #cv_data['cv'] = (cv_data['std'] / cv_data['mean'])**2
        
        return cv_data
    
    # Modulo: Analisis y computo del intervalo medio de demanda - ADI
    def get_ADI(self, data, columns):
        #ADI = round(data.Period.count() / data[data.Demand.notnull()].Demand.count(), 2)
        # Calcular la demanda (count) por fecha -> dia, mes o año por cada valor de la columna analizar
        data_retail = data.groupby(columns).agg(demand = (columns[0], 'count')).reset_index()
        for col in data_retail[columns[:-1]]:
            data_retail[col] = data_retail[col].astype(str)

        # Definicion de la etiqueta en base a la granularidad de las variables seleccionadas
        data_retail["label"] = data_retail[columns[:-1]].apply("_".join, axis = 1)

        print(data_retail.head())
        print("---"*20)

        # Extraccion del periodo por valor de la columna analizar
        adi_data = data_retail.label.value_counts().reset_index()
        #print(adi_data.head())
        #print("---"*20)
        
        # Extraccion del contador por demanda y valor de la columna analizar
        adi_data["demand"] = [data_retail[(data_retail.demand.notnull()) & (data_retail.label == col)].demand.count() for col in adi_data.label.unique().tolist()]
        adi_data["adi"] = round(adi_data["count"] / adi_data.demand, 4)
        #print(adi_data)
        #print("---"*20)
        
        return adi_data
    
    # Modulo: Catalogacion para determinar la dificultad de prediccion o clasificacion de patrones
    def get_category(self, df):
        # Predicciones sencillas
        if (df.adi <= 1.34) & (df.cv2 <= 0.49):
            return 'Smooth'

        # Predicciones complejas
        if (df.adi >= 1.34) & (df.cv2 >= 0.49):
            return 'Lumpy'

        # Predicciones dificiles
        if (df.adi < 1.34) & (df.cv2 > 0.49):
            return 'Erratic'

        # Predicciones dificiles
        if (df.adi > 1.34) & (df.cv2 < 0.49):
            return 'Intermittent'
        
    # Modulo: Generacion de las metricas del analisis de los intervalos de demanda
    def get_demand_classifier(self, data_frame_metric):
        data_final = pd.DataFrame()

        # Determinacion de etiquetado e insercion del resultado de los grados de prediccion de cada serie por granularidad seleccionada
        for key in data_frame_metric.keys():
            temp = data_frame_metric[key]
            temp["granularity"] = key
            temp["year"] = temp.granularity.str.split(" ", expand = True)[0]
            data_final = pd.concat([data_final, temp], axis = 0, ignore_index = False)
            
        return data_final
    
    # Modulo: Generacion de las caracteristicas del comportamiento sobre el grado de dificultad de prediccion o clasificacion de patrones
    def get_detail_demand(self, data_final):
        detail_data_gran = pd.DataFrame()
        profile = []
        number = []
        porc = []
        years = []

        # Generacion metricas de los intervalos de cada categoria del grado de prediccion
        for year in data_final.year.unique().tolist():
            temp = data_final[data_final.year == year]
            total = len(temp)
            category = temp.category.unique().tolist()
            part = [len(temp[temp.category == cat]) for cat in category]
            profile.extend(category)
            number.extend(part)
            porc.extend([(x/total) * 100 for x in part])
            years.append(year)

        # Construccion del dataframe detail
        detail_data_gran["profile"] = profile
        detail_data_gran["count"] = number
        detail_data_gran["percentage"] = porc
        detail_data_gran["year"] = years

        return detail_data_gran
    
    # Modulo: Identificacion de outliers (ruido o datos atipicos)
    def get_outliers(self, data, col_pred, threshold = 3):
        values = data[col_pred]
        # Handling outliers with a z-score threshold
        z_scores = np.abs(stats.zscore(values))

        # Adjust the threshold as needed
        filtered_data = values[z_scores < threshold]

        without_outliers = np.where(np.abs(z_scores) <= threshold)[0]
        outliers = np.where(np.abs(z_scores) > threshold)[0]

        # Plot the filtered data without outliers
        #"""
        plt.figure(figsize = (8, 4))
        plt.scatter(range(len(filtered_data)), filtered_data, label = 'Filtered Data')
        plt.xlabel('Data Point')
        plt.ylabel('Sale')
        plt.title('Filtered Data (Outliers Removed)')
        plt.legend()

        plt.savefig('plot.png', dpi=300, bbox_inches='tight')
        plt.show()
        #"""
        
        return outliers, without_outliers