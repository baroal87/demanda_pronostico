#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
import sys
from pandas import ExcelWriter
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt

class Queries():
    
    # Modulo: Constructor
    def __init__(self, path, test):
        self.path = path
        self.test = test
        self.miles_translator = str.maketrans(".,", ".,")
    
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

    # Modulo: Agrupamiento de los precios y ventas por granularidad
    def get_grouped_data_ABC(self, data, col_gran, col_obs_abc):
        # Agrupamiento y computo de los precios y ventas por granularidad
        data = data.groupby(col_gran).agg(price = (col_obs_abc[0], 'mean'), sales = (col_obs_abc[1], 'sum')).reset_index()

        # Computo de los ingresos
        data["revenue"] = data.price * data.sales
        data["revenue"] = data["revenue"].round()
        data["revenue"] = data["revenue"].astype(int)

        # Transformaccion del tipo de formato a date y extraccion de los meses y años por fecha
        data["fecha"] = pd.to_datetime(data["fecha"])
        data['week'] = data["fecha"].dt.isocalendar().week
        data['month'] = data["fecha"].dt.month
        data['year'] = data["fecha"].dt.year

        return data

    # Modulo: Agrupamiento de datos para los modelos
    def get_grouped_data_model(self, data, col_gran, var_obs, type_model = 2):
        if type_model == 1:
            #data = data.groupby(col_gran[:-1]).agg(date = (col_gran[-1], 'min'), sales = (var_obs[1], 'sum')).reset_index()
            data = data.groupby(col_gran).agg(sales = (var_obs[1], 'sum')).reset_index()
            #data.rename(columns = {"date": col_gran[-1]}, inplace = True)

        elif type_model == 2:
            data = data.groupby(col_gran).agg(price = (var_obs[0], 'mean'), sales = (var_obs[1], 'sum')).reset_index()
            data.price = data.price.round(2)
            data.sales = data.sales.round()

        else:
            print("\n >>> Error: Modelo no seleccionado !!!")
            sys.exit()

        return data

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
    def save_data_file_excel(self, data_final, data_detail, detail_data_abc_xyz, detail_data_hml, data_metric, data_fsct_models, name_folder):
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
                data_final.to_excel(writer, sheet_name = 'data_final', index = False)
                data_detail.to_excel(writer, sheet_name = 'demand_classifier_detail', index = False)
                detail_data_abc_xyz.to_excel(writer, sheet_name = 'inventory_abc_xyz_detail', index = False)
                detail_data_hml.to_excel(writer, sheet_name = 'hml_detail', index = False)
                data_metric.to_excel(writer, sheet_name = 'metrics_models_detail', index = False)
                data_fsct_models.to_excel(writer, sheet_name = 'fsct_model', index = False)

                #data_final_abc.total_revenue = data_final_abc.total_revenue.apply(lambda x: f"{x:,}".translate(self.miles_translator))
                #data_final_abc.to_excel(writer, sheet_name = 'classifier_abc', index = False)

            print(" >> Archivo Guardado correctamente")
            print("---"*20)

    # Modulo: uardado de graficas de estacionalidades
    def save_graph_seasonal(self, graph, name_graph, name_file, col_gran, period, cat_xyz, functions):
        # Plot the filtered data without outliers
        if not self.test:
            print("\n >> Proceso de guardado (Grafica)")
            # Validacion de la carpeta principal
            name_folder = "graphics/"
            functions.validate_path(name_folder)

            # Validacion de subcarpetas - nombre del archivo
            name_folder = name_folder + name_file + "/"
            functions.validate_path(name_folder)

            # Validacion de subcarpetas - periodo analisis (Semanal o Mensual)
            name_folder = name_folder + period + "/"
            functions.validate_path(name_folder)

            # Validacion de subcarpetas - columna (variable de granuralidad)
            name_folder = name_folder + col_gran + "/"
            functions.validate_path(name_folder)

            # Validacion de subcarpetas - columna (variable categoria ABC - HML)
            name_folder = name_folder + "AH/"
            functions.validate_path(name_folder)

            # Validacion de subcarpetas - columna (variable categoria XYZ)
            name_folder = name_folder + cat_xyz + "/"
            functions.validate_path(name_folder)

            # Definicion del nombre de la grafica
            #graph.savefig(self.path + name_folder + name_graph + '.png', dpi = 400, bbox_inches = 'tight')
            #graph.close()
            #plt.show()
            graph.plot().savefig(self.path + name_folder + name_graph + '.png', dpi = 500)

    # Modulo: Determinacion del nombre y guardado del dataframe resultante (BD)
    def save_data_bd(self, data):
        if not self.test:
            print("\n >> Proceso de guardado (Base datos)")