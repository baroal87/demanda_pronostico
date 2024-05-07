import pandas as pd
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join

from scipy import stats
import matplotlib.pyplot as plt

class Functions():
    
    # Modulo: Constructor
    def __init__(self, path, test):
        self.path = path
        self.test = test
    
    # Modulo: Verificacion de carpeta de resultados
    def validate_path(self, name_folder):
        if not self.test:
            path = self.path + name_folder
            # Validacion de existencia
            if os.path.exists(path):
                print("\n > Verificacion de direccion de almacenamiento: correcta !!!")
            
            # Geenracion de la carpeta result
            else:
                print("\n > Verificacion de direccion de almacenamiento: No existe !!!")
                print("\n >> Procedimiento de generacion de carpeta para almacenamiento")
                os.mkdir(path)

                # Verificacion de la carpeta
                if os.path.exists(path):
                    print(" >> La carpeta fue creada correctamente")
                    
                else:
                    print(" >> Error: No se creo la carpeta en la ruta \'{}\' -> {} !!!".format(name_folder, self.path))
                    sys.exit()
            print("---"*20)

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
                print("\n >> Error: Ingrese un valor numerico !!! \n")

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
                print("\n >> Error: Ingrese un valor numerico !!! \n")

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
                print("\n >> Error: Ingrese un valor numerico !!! \n")

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
                print("\n >> Error: Ingrese un valor numerico !!! \n")

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
                print("\n >> Error: Ingrese un valor numerico !!! \n")

        while True:
            try:
                x = int(input('Ingrese numero de columna (Observacion): '))    
                if (x < 1) & (x > total):
                    print("\n Indice incorrecto !!! \n")

                else:
                    col_name.append(x)
                    break

            except:
                print("\n >> Error: Ingrese un valor numerico !!! \n")

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
                print("\n >> Error: Ingrese un valor numerico !!! \n")

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
    
        data_final = data_final.reset_index()
            
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
    def get_outliers(self, data, col_obs, col_name, threshold = 3):
        values = data[col_obs]
        # Handling outliers with a z-score threshold
        z_scores = np.abs(stats.zscore(values))

        # Adjust the threshold as needed
        filtered_data = values[z_scores <= threshold]
        outliers_data = values[z_scores > threshold]

        without_outliers = np.where(np.abs(z_scores) <= threshold)[0]
        outliers = np.where(np.abs(z_scores) > threshold)[0]

        # Plot the filtered data without outliers
        if not self.test:
            plt.figure(figsize = (8, 4))
            plt.scatter(range(len(filtered_data)), filtered_data, label = 'Filtered Data')
            plt.scatter(range(len(outliers_data)), outliers_data, label = 'Outlier Data')
            plt.xlabel('Data Points')
            plt.ylabel(col_obs)
            plt.title('Filtered Data (Outliers Removed)')
            plt.legend()

            print("\n >> Proceso de guardado (Grafica)")            
            name_folder = "graphics/"
            self.validate_path(name_folder)
            name_file = col_name + "_grap_outlier"
            plt.savefig(self.path + name_folder + name_file + '.png', dpi = 400, bbox_inches = 'tight')
            plt.close()
            #plt.show()
        
        return outliers, without_outliers

    # Modulo: Computo de tiempos sobre un proceso
    def get_time_process(self, seg):
        if seg < 60:
            print(" >>> Tiempo: {} seg.".format(seg))

        elif (seg >= 60) & (seg < 3600):
            minutes = int(seg / 60)
            seg = int(seg % 60)
            if seg == 0:
                print(" >>> Tiempo: {} min.".format(minutes))

            else:
                print(" >>> Tiempo: {} min - {} seg.".format(minutes, seg))

        elif seg >= 3600:
            hour = int(seg / 3600)
            minutes = int( (seg - (hour * 3600)) / 60)
            seg = int(seg - ((hour * 3600) - (minutes * 60)) )

            if (minutes == 0) & ((seg == 0) | (seg != 0)):
                print(" >>> Tiempo: {} hr(s).".format(hour))

            elif (minutes != 0) & (seg == 0):
                print(" >>> Tiempo: {} h - {} min.".format(hour, minutes))

            else:
                print(" >>> Tiempo: {} h - {} min - {} seg.".format(hour, minutes, seg))