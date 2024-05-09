#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join

import math

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

                elif (x < 1) | (x > total):
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
        formats = ["%d-%m-%Y", "%d/%m/%Y", "%m-%d-%Y", "%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"]

        # Proceso: Validacion de tipo formato y seleccion de variable (tiempo)
        while True:
            try:
                x = int(input('Ingrese numero de columna (Tiempo): '))                
                for idx, format in enumerate(formats):
                    try:
                        if idx <= 1:
                            data[columns[x - 1]] = pd.to_datetime(data[columns[x - 1]], format = format, dayfirst = True)
                            
                        else:
                            data[columns[x - 1]] = pd.to_datetime(data[columns[x - 1]], format = format)

                        flag = True
                        break       

                    except Exception as e:
                        #print(e)
                        flag = False

                if (x < 1) | (x > total):
                    print("\n Indice incorrecto !!! \n")

                elif flag:
                    col_name.append(x)
                    break

                else:
                    print("\n >>> Error: Variable seleccionada no contiene un formato tipo: \'Date\' !!! \n")

            except:
                print("\n >> Error: Ingrese un valor numerico !!! \n")

        # Proceso: Validacion de tipo formato y seleccion de variable (observacion)
        while True:
            try:
                x = int(input('Ingrese numero de columna (Observacion): '))
                type = data[columns[x - 1]].dtype
                if (x < 1) | (x > total):
                    print("\n Indice incorrecto !!! \n")

                if (type == "int64") | (type == "int") | (type == "float64") | (type == "float"):
                    col_name.append(x)
                    break

                else:
                    print("\n >>> Error: Variable seleccionado no contiene un formato tipo: \'Numerico\' !!! \n")

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

                elif (x < 1) | (x > total):
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
    def get_cv(self, data, columns, col_obs, potencia = 2):
        #cv = round(pow(data.Demand.std() / data.Demand.mean(), 2), 2)
        # Agrupamiento de valores en base a la variables de observacion seleccionada
        #data_retail = data.groupby(columns).agg(sales = (columns[0], 'sum'), count_sales = (columns[0], 'count')).reset_index()
        data_retail = data.groupby(columns).agg(sales = (col_obs, 'sum'), count_sales = (col_obs, 'count')).reset_index()
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
    def get_ADI(self, data, columns, col_obs, col_time, period):
        date = data.groupby(columns[:-1]).agg(start = (col_time, 'min'), end = (col_time, 'max')).reset_index()
        for col in date[columns[:-1]]:
            date[col] = date[col].astype(str)

        if period == "month":
            date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'M')
            date.period = date.period.apply(lambda x: math.ceil(x))

        else:
            date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'W')
            date.period = date.period.apply(lambda x: int(round(x, 0)))

        date["label"] = date[columns[:-1]].apply("_".join, axis = 1)
        print(date.head())
        print("---"*20)
        #print(date[date.label.isin(["1111", "4922"])])
        #print("---"*20)

        #ADI = round(data.Period.count() / data[data.Demand.notnull()].Demand.count(), 2)
        # Calcular la demanda (count) por fecha -> dia, mes o año por cada valor de la columna analizar
        #data_retail = data.groupby(columns).agg(demand = (columns[0], 'count')).reset_index()
        columns.insert(len(columns), "year")
        data_retail = data.groupby(columns).agg(demand = (col_obs, 'count')).reset_index()
        for col in data_retail[columns[:-1]]:
            data_retail[col] = data_retail[col].astype(str)

        columns.remove("year")
        # Definicion de la etiqueta en base a la granularidad de las variables seleccionadas
        data_retail["label"] = data_retail[columns[:-1]].apply("_".join, axis = 1)
        #print(data_retail.head())
        #print("---"*20)

        # Extraccion del periodo por valor de la columna analizar
        adi_data = data_retail.label.value_counts().reset_index()
        adi_data.rename(columns = {"count": "demand"}, inplace = True)
        #print(adi_data.head())
        #print("---"*20)
        #print(adi_data[adi_data.label.isin(["1111", "4922"])])
        
        # Extraccion del contador por demanda y valor de la columna analizar
        adi_data = pd.merge(adi_data, date[["label", "period"]], how = "left", on = ["label"])
        
        ##adi_data["demand"] = [data_retail[(data_retail.demand.notnull()) & (data_retail.label == col)].demand.count() for col in adi_data.label.unique().tolist()]
        #adi_data["adi"] = round(adi_data["count"] / adi_data.demand, 4)
        adi_data["adi"] = round(adi_data.period / adi_data.demand, 1)
        #print(adi_data.head())

        return adi_data
    
    # Modulo: Catalogacion para determinar la dificultad de prediccion o clasificacion de patrones
    def get_category(self, df):
        # Predicciones sencillas
        if (df.adi < 1.32) & (df.cv2 < 0.49):
            #print(" >> smooth <<")
            return 'smooth'

        # Predicciones dificiles
        elif (df.adi >= 1.32) & (df.cv2 < 0.49):
            #print(" >> Intermittent <<")
            return 'intermittent'

        # Predicciones dificiles
        elif (df.adi < 1.32) & (df.cv2 >= 0.49):
            #print(" >> Erratic <<")
            return 'erratic'

        # Predicciones complejas
        elif (df.adi >= 1.32) & (df.cv2 >= 0.49):
            #print(" >> Lumpy <<")
            return 'lumpy'

    # Modulo: Generacion de las metricas del analisis de los intervalos de demanda
    def get_demand_classifier(self, data_frame_metric):
        data_final = pd.DataFrame()

        # Determinacion de etiquetado e insercion del resultado de los grados de prediccion de cada serie por granularidad seleccionada
        for key in data_frame_metric.keys():
            temp = data_frame_metric[key]
            temp["granularity"] = key
            temp["year"] = temp.granularity.str.split(" ", expand = True)[0]
            data_final = pd.concat([data_final, temp], axis = 0, ignore_index = False)
    
        data_final.reset_index(inplace = True, drop = True)
        data_final.adi.fillna(0, inplace = True)
        data_final.cv.fillna(0, inplace = True)
        data_final.cv2.fillna(0, inplace = True)
        data_final.category.fillna("None", inplace = True)
            
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
            porc.extend([(x / total) * 100 for x in part])
            years.extend([year for _ in range(len(category))])

        # Construccion del dataframe detail
        detail_data_gran["profile"] = profile
        detail_data_gran["count"] = number
        detail_data_gran["percentage"] = porc
        detail_data_gran.percentage = detail_data_gran.percentage.round(2)
        detail_data_gran["year"] = years

        return detail_data_gran
    
    # Modulo: Identificacion de outliers (ruido o datos atipicos)
    def get_outliers(self, data, col_obs, name_graph, name_file, col_period, threshold = 3):
        values = data[col_obs]
        # Handling outliers with a z-score threshold
        z_scores = np.abs(stats.zscore(values))

        # Adjust the threshold as needed
        filtered_data = values[z_scores <= threshold]
        outliers_data = values[z_scores > threshold]

        without_outliers = np.where(np.abs(z_scores) <= threshold)[0]
        outliers = np.where(np.abs(z_scores) > threshold)[0]

        """
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
            # Validacion de la carpeta principal
            name_folder = "graphics/"
            self.validate_path(name_folder)

            # Validacion de subcarpetas - nombre del archivo
            name_folder = name_folder + name_file + "/"
            self.validate_path(name_folder)
            
            # Validacion de subcarpetas - periodo analisis (Semanal o Mensual)
            name_folder = name_folder + col_period + "/"
            self.validate_path(name_folder)

            # Validacion de subcarpetas - año
            name_folder = name_folder + str(name_graph.split("_")[-1]) + "/"
            self.validate_path(name_folder)

            # Validacion de subcarpetas - columna (variable de granuralidad)
            name_folder = name_folder + str(name_graph.split("_")[0]) + "/"
            self.validate_path(name_folder)

            # Definicion del nombre de la grafica
            name_file = name_graph + "_grap_outlier"
            plt.savefig(self.path + name_folder + name_file + '.png', dpi = 400, bbox_inches = 'tight')
            plt.close()
            #plt.show()
        """
        
        return outliers, without_outliers

    # Modulo: Computo de series (observaciones) en base a la granularidad de los datos y periodo de analisis
    def get_count_series(self, data, columns_gran, col_time, period = "month"):
        grouped = data.groupby(columns_gran)
        label = []
        count = []

        for idx, group in grouped:
            #print(idx)
            for col_name in group[columns_gran[1:]]:
                group[col_name] = group[col_name].astype(str)

            if period == "month":
                group['month'] = group[col_time].dt.month
                count.append(len(group.month.unique().tolist()))

            else:
                group['week'] = group[col_time].dt.isocalendar().week
                count.append(len(group.week.unique().tolist()))

            #group["label"] = group[columns_gran[:-1]].apply("_".join, axis=1)
            group["label"] = group[columns_gran[1:]].apply("_".join, axis = 1)
            #label.append(str(list(idx)[:-1]).replace("[", "").replace("]", ""))
            label.extend(group.label.unique().tolist())

        data_label = pd.DataFrame()
        data_label["label"] = label
        name_col = "count_" + str(period)
        data_label[name_col] = count
        #data_label = data_label.groupby(["label"]).agg(months = ("count_month", 'sum')).reset_index()
        data_label = data_label.groupby(["label"]).agg({name_col: 'sum'}).reset_index()
        data_label.rename(columns = {name_col: period}, inplace = True)

        return data_label

    # Modulo: Computo de valores atipicos en base a la granularidad de los datos, año y periodo de analisis
    def get_count_outliers(self, data, columns_gran, col_time, name_file, period, year):
        grouped = data.groupby(columns_gran)
        label = []
        outliers = []
        not_outliers = []

        for idx, group in grouped:
            #print(idx)
            for col_name in group[columns_gran]:
                group[col_name] = group[col_name].astype(str)

            group["label"] = group[columns_gran].apply("_".join, axis = 1)
            # Proceso: Identificacion de valores atipicos en base a la granularidad
            #name_graph = col[1] + "_" + str(list(idx)[1])
            name_graph = columns_gran[0] + "_" + str(list(idx)[0]) + "_" + str(year) # variable_valor_año
            outliers_index, without_outliers_index = self.get_outliers(group, col_obs = col_time, name_graph = name_graph, name_file = name_file, col_period = period)
            #print('\n >> # Outliers:', len(outliers_index))
            #print(' >> # Without Outliers: {} \n'.format(len(without_outliers_index)))

            label.extend(group.label.unique().tolist())
            outliers.append(len(outliers_index))
            not_outliers.append(len(without_outliers_index))

        data_label = pd.DataFrame()
        data_label["label"] = label
        data_label["outliers"] = outliers
        data_label["without_outliers"] = not_outliers
        data_label = data_label.groupby(["label"]).agg(count_outliers = ("outliers", "sum"), 
                                                    count_not_outliers = ("without_outliers", "sum")).reset_index()

        return data_label

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