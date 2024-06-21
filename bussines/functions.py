#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join

import math
from dateutil.relativedelta import relativedelta
import datetime

from scipy import stats
import matplotlib.pyplot as plt

from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

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
                #if period == 1:
                #    period = "daily"
                #    break

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

    # Modulo: determinacion de las variables (precio y ingresos) para la clasificacion inventario ABC
    def select_var_abc(self, data):
        print("\n >>> Seleccion variable precio y observacion (ingresos, ventas, etc.) <<< \n")
        columns = data.columns.tolist()
        for idx, col_name in enumerate(columns):
            print(" > {} - {}".format(idx + 1, col_name))

        print()
        col_name = []
        total = len(columns)

        # Proceso: Validacion de tipo formato y seleccion de variable (precio)
        while True:
            try:
                x = int(input('Ingrese numero de columna (precio): '))
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

        # Proceso: Validacion de tipo formato y seleccion de variable (ingresos, ventas, etc.)
        while True:
            try:
                x = int(input('Ingrese numero de columna (ingresos, ventas, etc.): '))
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

        col_var_abc = []
        for index in col_name:
            index -= 1
            col_var_abc.append(columns[index])

        print("---"*20)

        return col_var_abc

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
        columns.insert(len(columns), "year")
        # Agrupamiento de valores en base a la variables de observacion seleccionada
        #data_retail = data.groupby(columns).agg(sales = (columns[0], 'sum'), count_sales = (columns[0], 'count')).reset_index()
        data_retail = data.groupby(columns).agg(sales = (col_obs, 'sum'), count_sales = (col_obs, 'count')).reset_index()
        for col in data_retail[columns[:-1]]:
            data_retail[col] = data_retail[col].astype(str)

        columns.remove("year")
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
    #def get_ADI(self, data, date, columns, col_obs, col_time, period):
    def get_ADI(self, data, date, columns, col_obs):
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
        
        # Extraccion del contador por demanda y valor de la columna analizar
        adi_data = pd.merge(adi_data, date[["label", "period"]], how = "left", on = ["label"])
        
        ##adi_data["demand"] = [data_retail[(data_retail.demand.notnull()) & (data_retail.label == col)].demand.count() for col in adi_data.label.unique().tolist()]
        #adi_data["adi"] = round(adi_data["count"] / adi_data.demand, 4)
        #adi_data["adi"] = round(adi_data.period / adi_data.demand, 1)
        adi_data["adi"] = round(adi_data.period / adi_data.demand, 2)

        # Validacion para identificar la demanda sea menor al periodo total
        adi_data["flag"] = np.where(adi_data.demand <= adi_data.period, 0, 1)
        #adi_data.loc[adi_data.flag == 1, "adi"] = 0
        adi_data.loc[adi_data.flag == 1, "demand"] = adi_data.period
        adi_data.drop("flag", axis = 1, inplace = True)
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
            #for col_name in group[columns_gran]:
                group[col_name] = group[col_name].astype(str)

            if period == "month":
                group['month'] = group[col_time].dt.month
                count.append(len(group.month.unique().tolist()))

            else:
                group['week'] = group[col_time].dt.isocalendar().week
                count.append(len(group.week.unique().tolist()))

            #group["label"] = group[columns_gran[:-1]].apply("_".join, axis=1)
            group["label"] = group[columns_gran[1:]].apply("_".join, axis = 1)
            #group["label"] = group[columns_gran].apply("_".join, axis = 1)

            #label.append(str(list(idx)[:-1]).replace("[", "").replace("]", ""))
            label.extend(group.label.unique().tolist())

        data_label = pd.DataFrame()
        data_label["label"] = label
        #name_col = "count_" + str(period)
        name_col = str(period) + "s_behavior"
        data_label[name_col] = count
        #data_label = data_label.groupby(["label"]).agg(months = ("count_month", 'sum')).reset_index()
        data_label = data_label.groupby(["label"]).agg({name_col: 'sum'}).reset_index()
        #data_label.rename(columns = {name_col: period}, inplace = True)

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

    # Modulo: Identificacion de etiquetacion (6, 12, 24, etc) meses
    def identify_periods(self, date, period, num_per = 20):
        int_per = []

        # Identificacion y computo de intervalos de tiempo
        # Validacion por dias
        if period == "daily":
            total_days = 365
            for idx in range(num_per):        
                if idx == 0:
                    int_per.append(int(total_days / 2))

                else:
                    int_per.append(total_days * idx)

        # Validacion por meses
        elif period == "month":
            total_months = 12
            for idx in range(num_per):        
                if idx == 0:
                    int_per.append(int(total_months / 2))

                else:
                    int_per.append(total_months * idx)

        # Validacion por semanas
        else:
            total_weeks = 52
            for idx in range(num_per):        
                if idx == 0:
                    int_per.append(int(total_weeks / 2))

                else:
                    int_per.append(total_weeks * idx)

        int_per.reverse()
        total = len(int_per) - 1
        date["flag_periods"] = ""
        for idx, per in enumerate(int_per):
            if idx == total:
                date.loc[date.period <= per, "flag_periods"] = "6 months" #if period != "month" else str(per) + " months"

            else:
                date.loc[date.period <= per, "flag_periods"] = str((total - idx) * 12) + " months" if period != "month" else str(per) + " months"

        return date

    # Modulo: Validacion de minimo de semanas permitidas para pronosticar
    def validate_min_week(self, data, columns_gran, col_time, min_week = 6):
        grouped = data.groupby(columns_gran[1:])
        label = []
        count = []

        for idx, group in grouped:
            #print(idx)
            for col_name in group[columns_gran[1:]]:
            #for col_name in group[columns_gran]:
                group[col_name] = group[col_name].astype(str)

            group["label"] = group[columns_gran[1:]].apply("_".join, axis = 1)
            label.extend(group.label.unique().tolist())

            group['week'] = group[col_time].dt.isocalendar().week
            count.append(len(group.week.unique().tolist()))

        data_label = pd.DataFrame()
        data_label["label"] = label
        data_label["count_week"] = count
        data_label = data_label.groupby(["label"]).agg({"count_week": 'sum'}).reset_index()
        data_label["flag_week_min"] = np.where(data_label.count_week > min_week, 0, 1)

        return data_label

    # Modulo: Identificador de series activas en base a un intervalo de tiempo
    def get_data_active(self, data, columns, col_time, limit_date = 6):
        max_date = data[col_time].max()
        # Determinacion del limite de observaciones activas (6 meses)
        limit_date_ref = max_date + relativedelta(months = - limit_date)
        #print(max_date, limit_date_ref)

        data_active = data.copy()
        data_active = data_active[data_active[col_time] >= limit_date_ref]
        data_active['month'] = data_active[col_time].dt.month
        col = columns[:-1]
        col.append("month")
        data_active = data_active[col].drop_duplicates()
        data_active = data_active.groupby(columns[:-1]).agg({"month": 'count'}).reset_index()

        # Transformacion de tipo de valor
        for col in data_active[columns[:-1]]:
            data_active[col] = data_active[col].astype(str)

        # Generacion de identificador
        data_active["label"] = data_active[columns[:-1]].apply("_".join, axis = 1)

        # Validacion de datos activos
        data_active["active"] = np.where(data_active["month"] >= limit_date, 1, 0)
        data_active = data_active[["label", "active"]].drop_duplicates()
        
        return data_active

    # Modulo: Computo de total de dias, semanas o meses
    def get_compute_dif_dates(self, data, columns, col_time, period = "month", limit_date = 6):
        # Agrupamento por min y max fecha
        date = data.groupby(columns[:-1]).agg(start = (col_time, 'min'), end = (col_time, 'max')).reset_index()

        # Transformacion de tipo de valor
        for col in date[columns[:-1]]:
            date[col] = date[col].astype(str)

        # Computo del total de dias
        if period == "daily":
            #date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'D')
            #date.period = date.period.apply(lambda x: int(round(x, 0)))
            
            date['period'] = (date['end'].dt.to_period('D').sub(date['start'].dt.to_period('D')).apply(lambda x: x.n))
            date.period = date.period.apply(lambda x: math.ceil(x))
            date.period = date.period + 1

        # Computo del total de meses
        elif period == "month":
            #date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'M')
            #date.period = date.period.apply(lambda x: math.ceil(x))
            #date.period = date.period + 1
            
            date['period'] = (date['end'].dt.to_period('M').sub(date['start'].dt.to_period('M')).apply(lambda x: x.n))
            date.period = date.period.apply(lambda x: math.ceil(x))
            date.period = date.period + 1

        # # Computo del total de semanas
        else:
            #date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'W')
            #date.period = date.period.apply(lambda x: int(round(x, 0)))
            
            date['period'] = (date['end'].dt.to_period('W').sub(date['start'].dt.to_period('W')).apply(lambda x: x.n))
            date.period = date.period.apply(lambda x: math.ceil(x))
            #date.period = date.period + 1

        # Generacion de identificador
        date["label"] = date[columns[:-1]].apply("_".join, axis = 1)

        # Determinacion del limite de observaciones activas (6 meses)
        data_active = self.get_data_active(data, columns, col_time, limit_date)
        date = pd.merge(date, data_active, how = "left", on = ["label"])

        #date = self.identify_periods(date.copy(), period)
        date["total_periods"] = date.period #date.period.apply(lambda x: str(x) + " month" if x == 1 else str(x) + " months")
        #print(date.active.unique())
        #print(date.flag_periods.unique())

        return date

    # Modulo: Catalogacion de las categias ABC
    def classify_abc(self, percentage):
        """ 
        A -> representan 80% de los ingresos totales
        B -> representan 10%, donde A+B integran el 90% de los ingresos totales
        C -> representan el ultimo 10% de los ingresos totales
        """
        if (percentage > 0) & (percentage <= 0.80):
            return "A"

        elif (percentage > 0.80) & (percentage <= 0.90):
            return "B"

        else:
            return 'C'

    # Modulo: Modelo de clasificacion de inventario ABC
    def get_data_ABC(self, data, col_gran, period):
        col_gran.extend([period, "year"])
        #print(col_gran)

        # Sumatoria de los ingresos por granularidad
        #data = data.groupby(col_gran)['revenue'].sum().to_frame().reset_index()
        data = data.groupby(col_gran).agg(revenue = ('revenue', 'sum'), sales = ('sales', 'sum')).reset_index()
        data["sales"] = data["sales"].round()
        data["sales"] = data["sales"].astype(int)

        for col in data[col_gran[:-2]]:
            data[col] = data[col].astype(str)

        # Formateo, identificadores y eliminacion de variables
        data["label"] = data[col_gran[:-2]].apply("_".join, axis = 1)
        #data["month"] = data.month.map("{:02}".format)
        data[period] = data[period].map("{:02}".format)
        name_col = "year_" + period
        data[name_col] = data['year'].map(str) + '-' + data[period].map(str)
        data.drop(col_gran[:-2], axis = 1, inplace = True)
        
        df = data.copy()
        df = df.groupby("label")['sales'].sum().to_frame().reset_index()

        # Pivoteo en base al parseo del año/mes sobre las identificadores y sumatoria de los ingresos
        data = data.pivot(index = "label", columns = name_col, values = 'revenue').reset_index().fillna(0)
        data_xyz = self.get_data_XYZ(data.copy())

        # Computo del total de los ingresos
        data['total'] = data.iloc[:, 1:].sum(axis =  1, numeric_only = True)

        # Agrupamiento para el computo del total de ingresos por cada identificador de granularidad
        data = data.groupby('label').agg(total_revenue = ('total','sum')).sort_values(by = 'total_revenue', ascending = False).reset_index()

        # Computo de la suma de los ingresos acomulados y la equivalencia del porcentaje por cada identificador de las variables de granularidad
        # y el total de los ingresos
        data['rev_count'] = data['total_revenue'].cumsum()
        data['rev_total'] = data['total_revenue'].sum()
        data['rev_percent'] = round(data['rev_count'] / data['rev_total'], 2)

        # Clasificacion de las categorias A, B o C
        data['category_abc'] = data['rev_percent'].apply(self.classify_abc)

        # Clasificacion del ranking por categoria a, b y c (metodo de pareto)
        #data['rank_abc'] = data['rev_percent'].rank().astype(int)

        df = pd.merge(df, data[['label', 'category_abc']], how = "left", on = ["label"])
        grouped_rank = df.groupby(['label', 'category_abc']).agg(total_sales = ('sales', 'sum')).reset_index()
        #print(grouped_rank.head())
        #print(grouped_rank.shape)
        #print("----"*30)

        data = data.groupby(['label', 'category_abc']).agg(total_revenue = ('total_revenue', sum)).reset_index()
        data.total_revenue = data.total_revenue.astype(int)

        # Concatenacion de los modelos ABC y XYZ
        data = pd.merge(data, data_xyz, how = "left", on = ["label"])
        data = pd.merge(data, grouped_rank, how = "left", on = ["label", 'category_abc'])

        return data

    # Modulo: Catalogacion de las categorias XYZ
    def classify_xyz(self, cv):
        if cv <= 0.5:
            return "X"

        elif (cv >= 0.5) & (cv <= 1):
            return "Y"

        else:
            return 'Z'

    # Modulo: Modelo de clasificacion XYZ
    def get_data_XYZ(self, data):
        # Determinacion del total de meses
        total = len(data.columns.tolist()) - 1

        # Computo del total, promedio, std y CV (Coeficiente de variacion) de ingresos
        data['total'] = data.iloc[:, 1:].sum(axis =  1, numeric_only = True)
        data["average"] = data.total / total
        data["std"] = data.iloc[:, 1: total].std(axis = 1)
        data['cv'] = round(data['std'] / data['average'], 2)
        
        # Clasificacion de las categorias X, Y o Z
        data['category_xyz'] = data['cv'].apply(self.classify_xyz)

        return data[['label', 'category_xyz']].reset_index(drop = True)

    # Modulo: Catalogacion del grado de forecast
    def classify_fsct(self, fsct):
        if fsct == "A_smooth":
            return "easy"

        elif fsct == "A_erratic":
            return "harder"

        elif fsct == "A_intermittent":
            return "harder"

        elif fsct == "A_lumpy":
            return "very difficult"

        elif fsct == "B_smooth":
            return "easy"

        elif fsct == "B_erratic":
            return "harder"

        elif fsct == "B_intermittent":
            return "difficult"

        elif fsct == "B_lumpy":
            return "very difficult"

        elif fsct == "C_smooth":
            return "easy"

        elif fsct == "C_erratic":
            return "difficult"

        elif fsct == "C_intermittent":
            return "difficult"

        elif fsct == "C_lumpy":
            return "very difficult"

        else:
            return "none"

    # Modulo: Catalogacion del grado de prediccion
    def get_forecastability(self, data):
        data = data[["label", "category_abc", "category_behavior"]]

        columns = data.columns.tolist()
        data["label_fcst"] = data[columns[-2:]].apply("_".join, axis = 1)

        # Clasificacion del grado de prediccion
        data['forecastability'] = data['label_fcst'].apply(self.classify_fsct)
        #data.drop("label_fcst", axis = 1, inplace = True)

        return data

    # Modulo: Generacion de las metricas del analisis de los intervalos de demanda
    def get_classifier_inventory_abc(self, data_frame_abc):
        data_final = pd.DataFrame()

        # Determinacion de etiquetado e insercion del resultado de los grados de prediccion de cada serie por granularidad seleccionada
        for key in data_frame_abc.keys():
            temp = data_frame_abc[key]
            temp["granularity"] = key
            data_final = pd.concat([data_final, temp], axis = 0, ignore_index = False)

        data_final.reset_index(inplace = True, drop = True)

        return data_final

    # Modulo: Generacion de las caracteristicas del clasificador ABC y XYZ
    def get_detail_abc(self, data):
        grouped_abc = data.groupby(['granularity', 'category_abc']).agg(count_abc = ('label', 'count'), 
                                                                        count_label = ('label', 'nunique'), 
                                                                        total_revenue = ('total_revenue', 'sum')).reset_index()
        grouped_abc["%_rev"] =  grouped_abc.groupby(["granularity"])["total_revenue"].apply(lambda x:  round(100 * (x / x.sum()))).reset_index(drop = True)
        grouped_abc["%_abc"] =  grouped_abc.groupby(["granularity"])["count_abc"].apply(lambda x:  round(100 * (x / x.sum()))).reset_index(drop = True)
        grouped_abc['%_rank'] = round((grouped_abc['count_label'] / grouped_abc['count_label'].sum()) * 100, 2)
        #grouped_abc['rank_values'] = grouped_abc.groupby(["granularity", "count_label"])['count_label'].apply(lambda x:  x.sum()).reset_index(drop = True)
        grouped_abc.rename(columns = {"count_label": "rank"}, inplace = True)
        grouped_abc.drop("count_abc", axis = 1, inplace = True)

        grouped_xyz = data.groupby(['granularity', 'category_xyz']).agg(count_xyz = ('label', 'count')).reset_index()
        grouped_xyz["%_xyz"] =  grouped_xyz.groupby(["granularity"])["count_xyz"].apply(lambda x:  round(100 * (x / x.sum()))).reset_index(drop = True)

        #grouped_rank = data.groupby(['granularity', 'category_abc']).agg(total_sales = ('total_sales', 'sum')).reset_index()        
        #grouped_rank["total"] =  grouped_rank.groupby(["granularity"])["total_sales"].apply(lambda x:  round(100 * (x / x.sum()))).reset_index(drop = True)
        #grouped_rank["%_rank"] = round(grouped_rank.count_rank / grouped_rank.total, 2) * 100
        #print(grouped_rank)
        grouped = data.groupby(['granularity', "label", 'category_abc']).agg(count_abc = ('label', 'count'), 
                                                                    count_label = ('label', 'nunique'), 
                                                                    total_revenue = ('total_revenue', 'sum')).reset_index()

        grouped = grouped.groupby(['granularity'])
        rank = {"A": 20, "B": 50, "C": 100}
        for name, group in grouped:
            total = group.total_revenue.sum()
            for key in group.category_abc.unique().tolist():
                temp = group[group.category_abc == key]
                #print(group.head())
                #total = group.count_label.sum()
                #print(group.count_label / total)

                temp = temp.sort_values(['total_revenue'], ascending = False)
                temp["percentage"] =  round((temp.total_revenue / total) * 100, 2)
                temp["cumsum_perc"] = temp.percentage.cumsum()
                temp.loc[temp.cumsum_perc > 100, "cumsum_perc"] = 100
                #print(temp.tail(3))
                #print(temp.shape)

                """
                if "A" == list(name)[-1]:
                    print(len(temp[(temp.cumsum_perc >= 0) & (temp.cumsum_perc <= rank[key])]))

                elif "B" == list(name)[-1]:
                    print(len(temp[(temp.cumsum_perc > rank["A"]) & (temp.cumsum_perc <= rank[key])]))

                else:
                    print(len(temp[(temp.cumsum_perc > rank["B"]) & (temp.cumsum_perc <= rank[key])]))
                """
            #break
        #print("--------------------------+++++++++++++++++++++++++++++++")

        #data = pd.merge(grouped_abc, grouped_xyz, how = "left", on = ["label"])
        data = pd.concat([grouped_abc, grouped_xyz.drop("granularity", axis = 1)], axis = 1, ignore_index = False)
        #data = pd.concat([grouped_abc, grouped_rank.drop(["granularity", "category_abc"], axis = 1)], axis = 1, ignore_index = False)

        return data #grouped_abc

    # Modulo: Generacion de las metricas del analisis de los intervalos de demanda
    def get_demand_classifier(self, data_frame_metric):
        data_final = pd.DataFrame()

        # Determinacion de etiquetado e insercion del resultado de los grados de prediccion de cada serie por granularidad seleccionada
        for key in data_frame_metric.keys():
            temp = data_frame_metric[key]
            temp["granularity"] = key
            #temp["year"] = temp.granularity.str.split(" ", expand = True)[0]
            data_final = pd.concat([data_final, temp], axis = 0, ignore_index = False)

        data_final.reset_index(inplace = True, drop = True)
        data_final.adi.fillna(0, inplace = True)
        #data_final.cv.fillna(0, inplace = True)
        data_final.cv2.fillna(0, inplace = True)

        # Validacion - Nulos, Activo o Min Semanas
        data_final.category_behavior.fillna("N/A", inplace = True)
        #data_final.loc[data_final.active == 0, "category_behavior"] = "none"
        data_final.loc[data_final.flag_week_min == 1, "category_behavior"] = "limit_data"
        data_final.loc[data_final.adi == 0, "category_behavior"] = "N/A"
        #data_final.drop("flag_week_min", axis = 1, inplace = True)
        data_final.rename(columns = {"flag_week_min": "flag_new"}, inplace = True)

        return data_final
    
    # Modulo: Generacion de las caracteristicas del comportamiento sobre el grado de dificultad de prediccion o clasificacion de patrones
    def get_detail_demand(self, data_final):
        detail_data_gran = pd.DataFrame()
        profile = []
        number = []
        porc = []
        years = []
        granularity = []

        # Generacion metricas de los intervalos de cada categoria del grado de prediccion
        #for year in data_final.year.unique().tolist():
        for gran in data_final.granularity.unique().tolist():
            #temp = data_final[data_final.year == year]
            temp = data_final[data_final.granularity == gran]
            total = len(temp)
            category = temp.category_behavior.unique().tolist()
            part = [len(temp[temp.category_behavior == cat]) for cat in category]
            profile.extend(category)
            number.extend(part)
            porc.extend([(x / total) * 100 for x in part])
            #years.extend([year for _ in range(len(category))])
            granularity.extend([gran for _ in range(len(category))])

        # Construccion del dataframe detail
        detail_data_gran["profile"] = profile
        detail_data_gran["count"] = number
        detail_data_gran["percentage"] = porc
        detail_data_gran.percentage = detail_data_gran.percentage.round(2)
        #detail_data_gran["year"] = years
        detail_data_gran["granularity"] = granularity

        return detail_data_gran

    # Modulo: Integracion de la categoria de comportamiento de inventario de demanda
    def set_catgory_data(self, data, data_demand, col_gran):
        # Transformacion de tipo de valor
        fil_data = []
        for idx, col in enumerate(col_gran):
            data[col] = data[col].astype(str)
            fil_data.insert(0, col)

            # Generacion de identificador
            idx += 1
            col_name = "label_gran_" + str(idx)
            data[col_name] = data[fil_data].apply("_".join, axis = 1)

            # Join del tipo de categoria por granularidad
            #data = pd.merge(data, data_demand[["label", "category_behavior"]], how = "left", left_on = col_name, right_on = 'label')
            #data.rename(columns = {"category_behavior": "cat_beh_gran_" + str(idx)}, inplace = True)
            data = pd.merge(data, data_demand[["label", "category_behavior", "flag_new"]], how = "left", left_on = col_name, right_on = 'label')
            #data.rename(columns = {"category_behavior": "cat_beh_gran_" + str(idx), "flag_new": "flag_ben_gran_" + str(idx)}, inplace = True)
            data = data[(data.flag_new != 1) | (data.category_behavior != "N/A")]
            #data = data[(data.flag_new != 1)]
            data.rename(columns = {"category_behavior": "cat_beh_gran_" + str(idx)}, inplace = True)
            data.drop(["label", "flag_new"], axis = 1, inplace = True)

        return data

    # Modulo: Seleecion de la metrica predominante por cada modelo entrenado
    def get_best_metric(self, values, metric):
        """
        MAE (Error absoluto medio) -> Diferencia entre pronostico y real (promedio del error absoluto)
        RMSE (Raiz del error cuadratico medio) -> Magnitud del promedio errores y la desviacion del valor real (desviacion cuadratica media)
            + Un valor 0 indica que el modelo tiene un ajuste perfecto
            + Menor RMSE, mejor serán las predicciones
        MAPE (Error porcentaje medio absoluto) -> Calculo de errores relativos para comparar la precision de las predicciones (porcentaje absoluto pronosticos)
            + Porcenjage bajo (modelo ajustado)
        R2 (Coef determinacion) -> Variacion de la variable dependiente (prediccion) sobre la variable independiente (ajuste del modelo)
            + Datos sesgados, ajustes son buenos (parametros) o modelo adecuado
            + r2: < 0.5 (malo), 0.5 - 0.8 (moderado) y > 0.8 (bueno)
        """
        if metric == "mae":
            values.sort(reverse = False)
            min_mae = min(values)
            #print(min_mae)
            return min_mae

        elif metric == "rmse":
            values.sort(reverse = False)
            min_rmse = min(values)
            #print(min_rmse)
            return min_rmse

        elif metric == "mape":
            values.sort(reverse = False)
            min_mape = min(values)
            #print(min_mape)
            return min_mape

        elif metric == "acc":
            values.sort(reverse = True)
            max_acc = max(values)
            #print(max_acc)
            return max_acc

        elif metric == "r2":
            values.sort(reverse = True)
            max_r2 = max(values)
            #print(max_r2)
            return max_r2

    # Modulo: Evaluacion y determinacion del modelo predomiante
    def get_evaluate_model(self, data):
        best_models = []
        for label in data['label'].unique():
            temp = data[data.label == label]
            #temp.drop(temp.columns.tolist()[-3:], axis = 1, inplace = True)
            temp.drop(["label", "full_time"], axis = 1, inplace = True)
            temp = temp.sort_values('type_model', ascending = True)

            # Extraccion de los nombres de las columas (metricas)
            columns_metrics = temp.columns.tolist()[:-2]
            #print(columns_metrics)

            # Generacion del catalogo del mejor modelo por metrica
            models = {}
            for metric in columns_metrics:
                result = self.get_best_metric(temp[metric].values.tolist(), metric)
                models[metric] = temp[temp[metric] == result].type_model.values.tolist()[0]

            #print("\n >> Catalogo del mejor modelo por metrica <<<")
            #print(models)
            #print("---"*30)

            # Extraccion de los nombres de los modelo
            values = list(models.values())

            # Contedor para determinar las apariciones de cada modelo
            #print("\n >> Catalogo de conteo de modelo mas significativo <<<")
            count_best_model = dict(zip(values, map(lambda x: values.count(x), values)))
            #print(count_best_model)
            #print("---"*30)

            # Identificador del conteo maximo de apariciones del mejor modelo por metrica
            count_max_model = max(list(count_best_model.values()))

            # Si existe mas de un modelo representativo, determinar el modelo con el menor sesgo
            if len([i for i in list(count_best_model.values()) if i == count_max_model]) > 1:

                # Identificacion de los modelos mas representativos
                list_model = []
                for model, count in count_best_model.items():
                    if count == count_max_model:
                        list_model.append(model)

                models = {}
                df = temp[temp.type_model.isin(list_model)]
                for metric in columns_metrics:
                    result = self.get_best_metric(df[metric].values.tolist(), metric)
                    models[metric] = df[df[metric] == result].type_model.values.tolist()[0]

                # Extraccion de los nombres de los modelo
                values = list(models.values())

                # Contedor para determinar las apariciones de cada modelo
                count_best_model = dict(zip(values, map(lambda x: values.count(x), values)))

                # Identificador del conteo maximo de apariciones del mejor modelo por metrica
                count_max_model = max(list(count_best_model.values()))

                # Si existe mas de un modelo representativo, determinar el modelo por la variable de tiempo
                if len([i for i in list(count_best_model.values()) if i == count_max_model]) > 1:
                    #print("Validacion por tiempo")
                    min_sec = min(temp[temp.type_model.isin(list_model)].seconds.values.tolist())
                    best_models.append({"label": label, "model": df[df.seconds == min_sec].type_model.values.tolist()[0]})
                    print("\n >> Mejor modelo ({}): {}".format(label, df[df.seconds == min_sec].type_model.values.tolist()[0]))

                else:
                    best_models.append({"label": label, "model": list(count_best_model.keys())[list(count_best_model.values()).index(count_max_model)]})
                    print("\n >> Mejor modelo ({}): {}".format(label, list(count_best_model.keys())[list(count_best_model.values()).index(count_max_model)]))

            # Determinar el modelo mas representativo
            else:
                best_models.append({"label": label, "model": list(count_best_model.keys())[list(count_best_model.values()).index(count_max_model)]})
                print("\n >> Mejor modelo ({}): {}".format(label, list(count_best_model.keys())[list(count_best_model.values()).index(count_max_model)]))

            #break

        best_models = pd.DataFrame.from_dict(best_models)
        return best_models

    def stationarity_check(self, TS):
        # Perform the Dickey Fuller Test
        dftest = adfuller(TS) # change the passengers column as required 

        # Print Dickey-Fuller test results
        #print ('Results of Dickey-Fuller Test:')
        

        dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        #print (dfoutput)

        return dfoutput

    # Modulo: Identificador de tipo y metricas para determinar la estacionalidad
    def get_graph_series_data(self, data, col_serie, period):
        dict_period = {"daily": 7, "week": 52, "month": 12}
        data = data.set_index(col_serie[0])
        data.sort_index(inplace = True)

        # Calculo de estacionalidad mult y addiptive
        mult_decomp = seasonal_decompose(data[col_serie[1]], model = 'multiplicative', period = dict_period[period])
        add_decomp = seasonal_decompose(data[col_serie[1]], model = 'additive', period = dict_period[period])
        residual_m = mult_decomp.resid
        residual_a = add_decomp.resid
        residual_m.dropna(inplace = True)
        residual_a.dropna(inplace = True)

        # Evaluacion de los tipos de estacionalidad por fuller y acf
        add = self.stationarity_check(residual_a)
        mult = self.stationarity_check(residual_m)
        acf_m = round(sum(pd.Series(sm.tsa.acf(mult_decomp.resid)).fillna(0)), 2)
        acf_a = round(sum(pd.Series(sm.tsa.acf(add_decomp.resid)).fillna(0)), 2)

        # Evaluacion de metricas para determinar si la serie es mult o add
        if add["p-value"].round(4) == mult["p-value"].round(4):
            #print("opcion 1")
            if acf_a < acf_m:
                return [add["p-value"].round(4), acf_a, mult["p-value"].round(4), acf_m, "additive"], add_decomp

            else:
                return [add["p-value"].round(4), acf_a, mult["p-value"].round(4), acf_m, "multiplicative"], mult_decomp

        elif add["p-value"].round(4) < mult["p-value"].round(4):
            #print("opcion 2")
            return [add["p-value"].round(4), acf_a, mult["p-value"].round(4), acf_m, "additive"], add_decomp

        else:
            #print("opcion 3")
            return [add["p-value"].round(4), acf_a, mult["p-value"].round(4), acf_m, "multiplicative"], mult_decomp

    # Modulo: Validacion de intervalos de tiempo para el computo del fsct de medias moviles (MA)
    def validate_data_serie_ma(self, data, col_serie, period, size_period):
        end_date = data[col_serie[0]].max()
        if period == "month":
            days = 30 * size_period
            start_date = end_date + relativedelta(days = -days)
            data['month'] = data[col_serie[0]].dt.month

        elif period == "week":
            days = 7 * size_period
            start_date = end_date + relativedelta(days = -days)
            data_serie['week'] = data_serie[col_serie[0]].dt.isocalendar().week        

        elif period == "daily":
            days = size_period
            start_date = end_date + relativedelta(days = -days)

        data_serie = data[data[col_serie[0]] >= start_date]

        if len(data_serie) < days:
            #print("\n > La data contiene dias faltantes... \n")
            # Determinar los intervalos
            #start_date = data.fecha.min()
            #end_date = datetime.date.today()
            data_serie.drop_duplicates(col_serie[0], inplace = True)

            # Generacion del dataframe de fechas y rellenado de los campos vacios con la variable observacion
            data_serie.set_index(col_serie[0], inplace = True)
            date_index = pd.date_range(start = start_date, end = end_date, freq = 'd')
            #data_serie = data_serie.reindex(date_index, method = 'bfill')
            data_serie = data_serie.reindex(date_index)
            data_serie[col_serie[1]].fillna(0, inplace = True) 
            data_serie.fillna(method = 'bfill', inplace = True)
            data_serie = data_serie.reset_index().rename(columns = {'index': col_serie[0]})
            data_serie[col_serie[0]] = pd.to_datetime(data_serie[col_serie[0]])

        return data_serie

    # Modulo:
    def validate_data_serie_models(self, start_date, period, size_period):
        if period == "month":
            days = 30 * size_period
            end_date = start_date + relativedelta(days = days)

        elif period == "week":
            days = 7 * size_period
            end_date = start_date + relativedelta(days = days)

        elif period == "daily":
            days = size_period
            end_date = start_date + relativedelta(days = days)

        date_index = pd.date_range(start = start_date, end = end_date, freq = 'd')
        return pd.DataFrame(date_index, columns = ['date'])

    # Modulo: Computo de tiempos sobre un proceso
    def get_time_process(self, seg):
        if seg < 60:
            print(" >>> Tiempo: {} seg.".format(seg))
            return "{} seg".format(seg)

        elif (seg >= 60) & (seg < 3600):
            minutes = int(seg / 60)
            seg = int(seg % 60)
            if seg == 0:
                print(" >>> Tiempo: {} min.".format(minutes))
                return "{} min".format(minutes)

            else:
                print(" >>> Tiempo: {} min - {} seg.".format(minutes, seg))
                return "{} min - {} seg".format(minutes, seg)

        elif seg >= 3600:
            hour = int(seg / 3600)
            minutes = int( (seg - (hour * 3600)) / 60)
            seg = int(seg - ((hour * 3600) - (minutes * 60)) )

            if (minutes == 0) & ((seg == 0) | (seg != 0)):
                print(" >>> Tiempo: {} hr(s).".format(hour))
                return "{} hr(s)".format(hour)

            elif (minutes != 0) & (seg == 0):
                print(" >>> Tiempo: {} h - {} min.".format(hour, minutes))
                return "{} h - {} min".format(hour, minutes)

            else:
                print(" >>> Tiempo: {} h - {} min - {} seg.".format(hour, minutes, seg))
                return "{} h - {} min - {} seg".format(hour, minutes, seg)