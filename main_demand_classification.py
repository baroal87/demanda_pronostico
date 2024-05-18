#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
from time import time
import traceback

# Importacion de clases
from data.data import Queries
#from connection.connect import Conexion
from bussines.functions import Functions

import warnings
warnings.simplefilter("ignore", category = DeprecationWarning)
warnings.filterwarnings('ignore', message = 'Unverified HTTPS request')
warnings.filterwarnings('ignore', message = 'SettingWithCopyWarning')
warnings.simplefilter(action = 'ignore', category = FutureWarning)
os.environ["PYTHONIOENCODING"] = "utf-8"

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

class Main_Demand_Series_Times():
    
    # Modulo: Constructor
    def __init__(self):
        #path = "C:/Softtek/demanda_pronostico/source/"
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path = BASE_DIR + "/demanda_pronostico/source/"

    # Modulo: Determinacion de variables para el analisis de patrones de demanda para determinar el grado la facilidad de prediccion o clasificacion de patrones
    def select_options(self):
        # Seleccion del archivo analizar
        name_file = self.functions.select_file()
        print("\n >> Archivo seleccionado: {}\n".format(name_file.split(".")[0]))
        while True:
            validate = input(' >> El archivo seleccionado es correcto (y / n): ')
            if validate.lower() == "y":
                break

            elif (validate.lower() != "y") & (validate.lower() != "n"):
                print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")
                
            elif validate.lower() == "n":
                name_file = self.functions.select_file()
                print("\n >> Archivo seleccionado: {}\n".format(name_file.split(".")[0]))

        print("+*+*"*30)

        # Extraccion de datos
        data = self.queries.get_data_file(name_file)
        print(data.head())
        print("*+*+"*30)

        # Seleccion y filtracion de las variables analizar del conjunto de datos de entrada
        variables, columns = self.functions.select_variables_data(data)
        #columns = data.columns.tolist()
        print("\n >> Columnas seleccionadas: {}\n".format(str(variables).replace("[", "").replace("]", "")))
        while True:
            validate = input(' >> Las variables seleccionadas son correctas (y / n): ')
            if validate.lower() == "y":
                data.drop(columns, axis = 1, inplace = True)
                break

            elif (validate.lower() != "y") & (validate.lower() != "n"):
                print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")
                
            elif validate.lower() == "n":
                variables, columns = self.functions.select_variables_data(data)
                print("\n >> Columnas seleccionadas: {}\n".format(str(variables).replace("[", "").replace("]", "")))

        print("+*+*"*30)

        # Seleccion del periodo analizar (Semanal o Mensual)
        period = self.functions.select_period()
        print(" >> Periodo seleccionado: {}\n".format(period))
        while True:
            validate = input(' >> El periodo seleccionado es correcto (y / n): ')
            if validate.lower() == "y":
                break

            elif (validate.lower() != "y") & (validate.lower() != "n"):
                print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")

            elif validate.lower() == "n":
                period = self.functions.select_period()
                print(" >> Periodo seleccionado: {}\n".format(period))

        print("+*+*"*30)

        # Seleccion de las variables de series de tiempo (Tiempo y Observacion)
        col_serie = self.functions.select_var_series(data)
        print(" >> Variables analisis de tiempo: {}\n".format(str(col_serie).replace("[", "").replace("]", "")))
        while True:
            validate = input('Las variables seleccionadas son correctas (y / n): ')
            if validate.lower() == "y":
                break
            
            elif (validate.lower() != "y") & (validate.lower() != "n"):
                print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")
                
            elif validate.lower() == "n":
                col_serie = self.functions.select_var_series(data)
                print(" >> Variables analisis de tiempo: {}\n".format(str(col_serie).replace("[", "").replace("]", "")))
        
        print("+*+*"*30)

        # Seleccion de las variables para filtrar la granularidad de los datos
        col_gran = self.functions.select_gran_data(data)
        print(" >> Granularidad seleccionada: {}\n".format(str(col_gran).replace("[", "").replace("]", "")))
        while True:
            validate = input('Las variables seleccionadas son correctas (y / n): ')
            if validate.lower() == "y":
                break
            
            elif (validate.lower() != "y") & (validate.lower() != "n"):
                print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")
                
            elif validate.lower() == "n":
                col_gran = self.functions.select_gran_data(data)
                print(" >> Granularidad seleccionada: {}\n".format(str(col_gran).replace("[", "").replace("]", "")))

        print("+*+*"*30)
                
        return data, period, col_serie, col_gran, name_file.split(".")[0]

    # Modulo: Analisis y generacion de los patrones de demanda para la viabilidad de clasificacion de observaciones
    def data_demand(self, data, period, col_serie, col_gran, name_file):
        print("\n >> DataFrame: Sales <<<\n")
        #try:
        #    data[col_serie[0]] = pd.to_datetime(data[col_serie[0]], format = "%Y-%m-%d")
            
        #except:
        #    data[col_serie[0]] = data[col_serie[0]].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
        #    data[col_serie[0]] = pd.to_datetime(data[col_serie[0]], format = "%Y-%m-%d")

        data = data[data[col_serie[1]] > 0]
        data['year'] = data[col_serie[0]].dt.year
        print(data.head())
        print("---"*30)
        
        #col_gran = ["state_id", "cat_id", "dept_id", "store_id"] #"item_id"
        #years = data.year.unique().tolist()
        years = [data.year.min()]
        years.sort(reverse = True)
        data_frame_metric = {}

        for year in years:
            print("\n >> Year: ", year)

            # Analisis por la granularidad seleccionada
            fil_data = col_serie.copy() #["date", "sales"]
            for col in col_gran:
                # Filtracion de datos
                fil_data.insert(0, col)
                df = data[(data.year >= year)]
                print("\n > Filtrado por: ", fil_data)
                df = df[fil_data]

                # Agrupamiento de valores en base a la variables de observacion seleccionada
                print("\n >>> DataFrame: Agrupado por la variable - ({}) <<<".format(col_serie[1]))
                columns = df.columns.tolist()
                #columns.remove("sales")
                columns.remove(col_serie[1])
                print("\n > Agrupado por: {}\n".format(columns))
                df = self.queries.grouped_data(df, columns, col_serie[1])
                
                df[col_serie[0]] = pd.to_datetime(df[col_serie[0]])
                df['week'] = df[col_serie[0]].dt.isocalendar().week
                df['month'] = df[col_serie[0]].dt.month
                df['year'] = df[col_serie[0]].dt.year
                print(df.head())
                print("---"*20)
                
                # Computo de series por la variables (Granuralidad)
                print("\n >>> DataFrame: Contador de Series por granularidad <<<\n")
                col = columns[:-1]

                ##### Eliminacion debido a la iteracion por año #####
                col.insert(0, "year")
                #print(col)

                data_label = self.functions.get_count_series(df.copy(), columns_gran = col, col_time = col_serie[0], period = period)
                data_label = pd.merge(data_label, self.functions.get_count_outliers(df.copy(), columns_gran = col[1:], col_time = col_serie[1], name_file = name_file, period = period, year = year), 
                                      how = "left", on = ["label"])
                print(data_label.head())
                print("---"*20)

                # Proceso: Identificacion de minimo de semanas para prediccion - 6 semanas
                print("\n >>> DataFrame: Minimo de semansa <<<\n".format(period))
                data_min_week = self.functions.validate_min_week(df.copy(), columns_gran = col, col_time = col_serie[0])
                print(data_min_week.head())
                print("---"*20)

                # Proceso: Computo y generacion del intervalo medio de demanda - ADI
                print("\n >>> DataFrame: Computo de periodos - ({}) <<<\n".format(period))
                columns = columns[:-1]
                columns.insert(len(columns), period)
                #print(columns)

                # limit_date -> debe contener un valor numerico negativo para el computo de fechas a recorrer del punto de fecha hacia atras
                #data_date = self.queries.get_compute_dif_dates(df.copy(), columns, col_time = col_serie[0], period = period, limit_date = 6, function = self.functions)
                data_date = self.functions.get_compute_dif_dates(df.copy(), columns, col_time = col_serie[0], period = period, limit_date = 6)
                print(data_date.head())
                print("---"*20)

                print("\n >>> DataFrame: ADI <<<\n")
                #adi_data = self.functions.get_ADI(df, columns = columns, col_obs = col_serie[1], col_time = col_serie[0], period = period)
                adi_data = self.functions.get_ADI(df.copy(), data_date, columns, col_obs = col_serie[1])
                print(adi_data.head())
                print("---"*20)

                # Proceso: Compute y generacion del coeficiente de variacion - CV2
                print("\n >>> DataFrame: CV2 <<<\n")
                cv_data = self.functions.get_cv(df.copy(), columns = columns, col_obs = col_serie[1], potencia = 2)
                print(cv_data.head())
                print("---"*20)

                # Proceso: Union de los analisis del ADI y CV2
                print("\n >>> DataFrame: Analisis Finalizada - ({}) \n".format(col[0]))
                df = pd.merge(adi_data[["label", "adi"]], cv_data[["label", "cv", "cv2"]], how = "left", on = ["label"])

                # Proceso: Clasificacion del tipo de categoria sobre el intervalo de demanda por ADI y CV2
                df["category"] = df.apply(self.functions.get_category, axis = 1)
                df = pd.merge(df, data_label, how = "left", on = ["label"])
                df = pd.merge(df, data_date[["label", "active", "flag_periods"]], how = "left", on = ["label"])
                df = pd.merge(df, data_min_week[["label", "flag_week_min"]], how = "left", on = ["label"])
                print(df.head())
                print("+++"*30)

                columns = columns[:-1]
                #metric_name = str(year) + " - " + ' - '.join(columns)
                metric_name = ' - '.join(columns)
                data_frame_metric[metric_name] = df
                
                #break
            #break
                
        return data_frame_metric

    def main(self):
        # Bandera de prueba
        test = False
        try:
            start_time = time()
        
            # Incializacion de clases (Metodo Contructor)
            self.queries = Queries(self.path, test)
            self.functions = Functions(self.path, test)

            # Seleccion del tipo de fuente para la extraccion de los datos
            #source_data = self.functions.select_source_data()
            source_data = 1
            # Extraccion de datos por archivo y seleccion de variables analizar
            if source_data == 1:
                #data, period, col_serie, col_gran, name_file = self.select_options()
                pass

            # Extraccion de datos por base de datos (Fijar las variables analizar, si no aplicar el modulo "select_options")
            elif source_data == 2:
                data = self.queries.get_data_netezza()
                # Definir periodo, variables de series (tiempo y observacion), columnas o variables de granuralidad de los datos
                period, col_serie, col_gran = ""
                
                # Detefinir el nombre para la validacion de las rutas para almacenar las graficas (outliers)
                # puede ser el nombre de la tabla o subfijo (el campo debe ser dinamico para evitar re-escrituras de las graficas)
                name_file = "name_table"
                
            else:
                print(" >>> Error: Seleccion de fuente incorrecta !!!\n")
                sys.exit()

            #print(period, col_serie, col_gran, name_file)
            source_data = 1
            period = "month" # week, month
            col_serie = ['fecha', 'sales']
            col_gran = ['dept_nbr', 'store_nbr'] #'dept_nbr',
            name_file = "data_Atom_agu_3.csv"
            
            data = self.queries.get_data_file(name_file)
            data = data.dropna()
            #data[col_serie[0]] = pd.to_datetime(data[col_serie[0]], format = "%d/%m/%Y", dayfirst = True)
            #data[col_serie[0]] = data[col_serie[0]].apply(lambda x: pd.to_datetime(x).strftime("%Y-%m-%d"))
            data[col_serie[0]] = pd.to_datetime(data[col_serie[0]], format = "%Y-%m-%d")
            print(data.head())
            name_file = "data_Atom_agu"
            print("*+*+"*30)

            # Proceso: Clasificación de los patrones de demanda
            data_frame_metric = self.data_demand(data.copy(), period, col_serie, col_gran, name_file)

            # Proceso: Generacion de la estractura final del dataframe clasificacion en base a demanda
            print("\n >>> Dataframe: Demand Classifier <<< \n")
            data_final = self.functions.get_demand_classifier(data_frame_metric)
            print(data_final.head())
            print("\n > Volumen: ", data_final.shape)
            print("---"*20)

            # Proceso: Obtencion de los porcentajes por categoria
            print("\n >>> Dataframe: Detail Classifier <<< \n")
            detail_data_gran = self.functions.get_detail_demand(data_final)
            print(detail_data_gran.head(20))
            print("---"*20)

            # Guardado del analisis - Dataframe intervalos de demanda & Dataframe detail
            if source_data == 1:
                # Validacion de la carpeta principal
                name_folder = "result/"
                self.functions.validate_path(name_folder)

                # Validacion de subcarpetas - nombre del archivo
                name_folder = name_folder + name_file + "/"
                self.functions.validate_path(name_folder)

                # Validacion de subcarpetas - periodo analisis (Semanal o Mensual)
                name_folder = name_folder + period + "/"
                self.functions.validate_path(name_folder)

                # Proceso: Guardado de archivo csv - final
                self.queries.save_data_file_csv(data_final, name_folder, name_file = name_file + "_final")

                # Proceso: Guardado de archivo csv - detail
                self.queries.save_data_file_csv(detail_data_gran, name_folder, name_file = name_file + "_detail")

                # Proceso: Guardado de archivo excel
                self.queries.save_data_file_excel(data_final, detail_data_gran, name_folder)

            else:
                self.queries.save_data_bd(detail_data_gran)
            print("---"*20)

            # Transformacion de tipo de valor
            fil_data = []
            for idx, col in enumerate(col_gran):
                data[col] = data[col].astype(str)
                fil_data.insert(0, col)

                # Generacion de identificador
                idx += 1
                col_name = "label_" + str(idx)
                data[col_name] = data[fil_data].apply("_".join, axis = 1)

                # Join del tipo de categoria por granularidad
                #data = pd.merge(data, data_final[["label", "category", "granularity"]], how = "left", left_on = col_name, right_on = 'label')
                data = pd.merge(data, data_final[["label", "category"]], how = "left", left_on = col_name, right_on = 'label')

                data.rename(columns = {"category": "category_" + str(idx)}, inplace = True)
                #data.rename(columns = {"category": "category_" + str(idx), "granularity": "granularity_" + str(idx)}, inplace = True)
                data.drop("label", axis = 1, inplace = True)

            print(data.head())
            data.to_csv(self.path + "data_cat.csv", index = False)
            print("---"*20)
            #print(data.isnull().sum())

            end_time = time()
            print('\n >>>> El analisis tardo <<<<')
            self.functions.get_time_process(round(end_time - start_time, 2))

        except Exception as e:
            print(traceback.format_exc())
        
if __name__ == "__main__":
    # Proceso de analisis
    series = Main_Demand_Series_Times()
    series.main()