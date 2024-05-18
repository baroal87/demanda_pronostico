#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from pandas import ExcelWriter
from dateutil.relativedelta import relativedelta

class Queries():
    
    # Modulo: Constructor
    def __init__(self, path, test):
        self.path = path
        self.test = test
    
    # Modulo: Extraccion de datos por fuente (API)
    def get_data_netezza(self):
        pass
    
    # Modulo: Extraccion de datos por archivo
    def get_data_file(self, name_file):
        # Carga y extraccion de datos
        data = pd.read_csv(self.path + name_file, low_memory = False)

        return data

    # Modulo: Agrupamiento de datos por la variable observacion seleccionada
    def grouped_data(self, data, col_grouped, var_obs):
        #data = data.groupby(col_grouped).agg({'sales': 'sum'}).reset_index()
        data = data.groupby(col_grouped).agg({var_obs: 'sum'}).reset_index()
    
        name_col = var_obs + "_sum"
        #data.rename(columns = {'sales_sum': 'sales'}, inplace = True)
        data.rename(columns = {name_col: var_obs}, inplace = True)

        return data

    # Modulo: Computo de total de dias, semanas o meses
    def get_compute_dif_dates(self, data, columns, col_time, period = "month", limit_date = 6, function = None):
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
        data_active["active"] = np.where(data_active["month"] >= limit_date, 1, 0)
        data_active = data_active[["label", "active"]].drop_duplicates()

        # Agrupamento por min y max fecha
        date = data.groupby(columns[:-1]).agg(start = (col_time, 'min'), end = (col_time, 'max')).reset_index()

        # Transformacion de tipo de valor
        for col in date[columns[:-1]]:
            date[col] = date[col].astype(str)

        # Computo del total de dias
        if period == "daily":
            date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'D')
            date.period = date.period.apply(lambda x: int(round(x, 0)))

        # Computo del total de meses
        elif period == "month":
            date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'M')
            date.period = date.period.apply(lambda x: math.ceil(x))
            date.period = date.period + 1

        # # Computo del total de semanas
        else:
            date['period'] = (date['end'] - date['start']) / np.timedelta64(1, 'W')
            date.period = date.period.apply(lambda x: int(round(x, 0)))

        # Generacion de identificador
        date["label"] = date[columns[:-1]].apply("_".join, axis = 1)

        # Determinacion del limite de observaciones activas (6 meses)
        #limit_date = max_date + relativedelta(months = limit_date)
        #print(max_date, limit_date)
        #date["active"] = 0
        #date.loc[date.end >= limit_date, "active"] = 1
        date = pd.merge(date, data_active, how = "left", on = ["label"])
        
        date = function.identify_periods(date.copy(), period)
        #print(date.active.unique())
        #print(date.flag_periods.unique())

        return date
    
    # Modulo: Determinacion del nombre y guardado del dataframe resultante (Archivo)
    def save_data_file_csv(self, data, name_folder, name_file):
        if not self.test:
            """
            print("\n >> Proceso de guardado (Archivo - CSV)")
            name_file = input("\n Ingrese el nombre del archivo: ")
            while True:
                validate = input('\n Nombre del archivo es correcto (y / n): ')
                if validate.lower() == "y":
                    break
                
                elif (validate.lower() != "y") & (validate.lower() != "n"):
                    print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")
                    
                elif validate.lower() == "n":
                    name_file = input("\n Ingrese el nombre del archivo: ")

            """
            #path = self.path + "result/"
            path = self.path + name_folder
            data.to_csv(path + name_file + ".csv", index = False)
            print(" >> Archivo Guardado correctamente")

    # Modulo: Guardado de un archivo excel
    def save_data_file_excel(self, data_demand, data_detail, name_folder):
        if not self.test:
            print("\n >> Proceso de guardado (Archivo - Excel)")
            name_file = input("\n Ingrese el nombre del archivo: ")
            while True:
                validate = input('\n Nombre del archivo es correcto (y / n): ')
                if validate.lower() == "y":
                    break
                
                elif (validate.lower() != "y") & (validate.lower() != "n"):
                    print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")
                    
                elif validate.lower() == "n":
                    name_file = input("\n Ingrese el nombre del archivo: ")

            path = self.path + name_folder
            with pd.ExcelWriter(path + name_file + ".xlsx", engine = "openpyxl") as writer:
                data_demand.to_excel(writer, sheet_name = 'demand_classifier', index = False)
                data_detail.to_excel(writer, sheet_name = 'demand_detail', index = False)

            print(" >> Archivo Guardado correctamente")

    # Modulo: Determinacion del nombre y guardado del dataframe resultante (BD)
    def save_data_bd(self, data):
        if not self.test:
            print("\n >> Proceso de guardado (Base datos)")