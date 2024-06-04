#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
from time import time
import traceback

from numba import jit, cuda
from timeit import default_timer as timer

# Importacion de clases
from data.data import Queries
#from connection.connect import Conexion
from bussines.functions import Functions
from model.model import Model_Series_Times

import warnings
warnings.simplefilter("ignore", category = DeprecationWarning)
warnings.filterwarnings('ignore', message = 'Unverified HTTPS request')
warnings.filterwarnings('ignore', message = 'SettingWithCopyWarning')
warnings.simplefilter(action = 'ignore', category = FutureWarning)
warnings.filterwarnings('ignore', message = 'ConvergenceWarning')

from sklearn.exceptions import ConvergenceWarning
ConvergenceWarning('ignore')
os.environ["PYTHONIOENCODING"] = "utf-8"

import logging
cmdstanpy_logger = logging.getLogger("cmdstanpy")
logging.getLogger('fbprophet').setLevel(logging.WARNING) 
cmdstanpy_logger.disabled = True

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

class Main_Model_Prediction():
    
    # Modulo: Constructor
    def __init__(self, col_gran, col_serie, col_obs_abc, period):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path = BASE_DIR + "/demanda_pronostico/source/"
        self.col_gran = col_gran
        self.col_serie = col_serie
        self.col_obs_abc = col_obs_abc
        self.period = period

    #@jit(target_backend='cuda', nopython=True)
    def main(self):
        test = False
        try:
            start_time = time()
            # Incializacion de clases (Metodo Contructor)
            self.queries = Queries(self.path, test)
            self.functions = Functions(self.path, test)
            self.model = Model_Series_Times(self.path, test)

            name_file = "data_cat.csv"
            data = self.queries.get_data_file(name_file)

            # Identificador de columnas (cat_beh_gran & fsct_gran)
            size = 2
            columns = data.columns.tolist()
            index = len(self.col_gran) * size
            col_cat = columns[-index:]
            fill_data = self.col_gran + self.col_serie + [self.col_obs_abc[0]] + col_cat

            col_cat = [col_cat[i: i + size] for i in range(0, len(col_cat), size)]
            print("\n >> Columnas: ", fill_data)
            print(" >> Segmentos catalogadas: {}\n".format(col_cat))

            data = data[fill_data]
            data = data[data[self.col_serie[1]] > 0]
            data[self.col_serie[0]] = pd.to_datetime(data[self.col_serie[0]])
            data['week'] = data[self.col_serie[0]].dt.isocalendar().week
            data['month'] = data[self.col_serie[0]].dt.month
            data['year'] = data[self.col_serie[0]].dt.year
            print(data.head())
            print("---"*30, "\n")

            data_frame_metric = {}
            for idx, segment in enumerate(col_cat):
                print("\n >>> DataFrame: Grouped - Parte 1 <<<\n")
                #col = self.col_gran + segment
                col = segment.copy()
                #col.extend([self.period, "year", self.col_serie[0]])
                col.append(self.col_serie[0])
                #print(col)
                df_model_1 = self.queries.get_grouped_data_model(data.copy(), col_gran = col, var_obs = self.col_serie, type_model = 1)
                print(" > Variables analizar: {}\n".format(df_model_1.columns.tolist()))
                df_model_1[segment] = df_model_1[segment].astype(str)
                df_model_1["segment"] = df_model_1[segment].apply("_".join, axis = 1)
                df_model_1[self.col_serie[1]] = df_model_1[self.col_serie[1]].astype(int)

                #df.drop(segment, axis = 1, inplace = True)
                #df = df.set_index(["index"])
                print(df_model_1.head())
                print(df_model_1.shape)
                print("---"*30)

                print("\n >>> Modelos: Prophet - AutoArima <<< \n")
                for name, group in df_model_1.groupby(segment):
                    #print(group.head(2))
                    start_time_model = time()
                    data_metric, col_pred = self.model.get_models_statsForecast(group.copy(), self.col_serie)
                    data_metric["label"] = list(name)[0]
                    end_time_model = time()
                    data_metric["time"] = self.functions.get_time_process(round(end_time_model - start_time_model, 2))
                    
                    print(data_metric)
                    for col in col_pred:
                        #metric_name = str(list(name)[0]) + "-AutoArima_stats"
                        metric_name = str(list(name)[0]) + col
                        data_frame_metric[metric_name] = data_metric[data_metric.model == col].drop("model", axis = 1)
                    
                    start_time_model = time()
                    data_metric = self.model.get_model_Arima(group.copy(), self.col_serie)
                    data_metric["label"] = list(name)[0]
                    end_time_model = time()
                    data_metric["time"] = self.functions.get_time_process(round(end_time_model - start_time_model, 2))
                    metric_name = str(list(name)[0]) + "-Arima"
                    data_frame_metric[metric_name] = data_metric

                    #start_time_model = time()
                    #data_metric = self.model.get_model_autoarima(group.copy(), col_serie = self.col_serie)
                    #data_metric["label"] = list(name)[0]
                    #end_time_model = time()
                    #data_metric["time"] = self.functions.get_time_process(round(end_time_model - start_time_model, 2))
                    #metric_name = str(list(name)[0]) + "-AutoArima_normal"
                    #data_frame_metric[metric_name] = data_metric

                    start_time_model = time()
                    col_serie = [period, "year", self.col_serie[1]]
                    data_metric = self.model.get_model_prophet(group.copy(), col_serie = self.col_serie)
                    data_metric["label"] = list(name)[0]
                    end_time_model = time()
                    data_metric["time"] = self.functions.get_time_process(round(end_time_model - start_time_model, 2))
                    metric_name = str(list(name)[0]) + "-Prophet"
                    data_frame_metric[metric_name] = data_metric

                    #self.model.get_model_forecasters(group, self.col_serie)
                    #break
 
                #"""
                print("\n >>> DataFrame: Grouped - Parte 2 <<< \n")
                col = segment.copy()
                col.extend([self.period, "year"])
                df_model_2 = self.queries.get_grouped_data_model(data.copy(), col_gran = col, var_obs = self.col_obs_abc, type_model = 2)
                print(" > Variables analizar: {}\n".format(df_model_2.columns.tolist()))
                print(df_model_2.head())
                print(df_model_2.shape)
                print("---"*30)
                #df[[self.period, "year"]] = df[[self.period, "year"]].astype(str)
                #columns_num, columns_cat = self.model.get_segmetation_variables(df, df.columns.tolist())
                columns_num = [self.col_obs_abc[0]]
                columns_cat = [segment[0], self.period, "year"]

                print("\n >>> Modelos: LGBM - CatBoost <<< \n")
                for name, group in df_model_2.groupby(segment):
                    start_time_model = time()
                    data_metric = self.model.get_model_LGBM(group.copy(), columns_num, columns_cat, col_pred = self.col_serie[1])
                    data_metric["label"] = list(name)[0]
                    end_time_model = time()
                    data_metric["time"] = self.functions.get_time_process(round(end_time_model - start_time_model, 2))
                    #metric_name = self.col_gran[idx] + "-LGBM"
                    metric_name = str(list(name)[0]) + "-LGBM"
                    data_frame_metric[metric_name] = data_metric
                    
                    start_time_model = time()
                    data_metric = self.model.get_model_CatBoost(group.copy(), columns_num, columns_cat, col_pred = self.col_serie[1])
                    data_metric["label"] = list(name)[0]
                    end_time_model = time()
                    data_metric["time"] = self.functions.get_time_process(round(end_time_model - start_time_model, 2))
                    #metric_name = self.col_gran[idx] + "-CatBoost"
                    metric_name = str(list(name)[0]) + "-CatBoost"
                    data_frame_metric[metric_name] = data_metric
                    #break
                #"""
                #break

            data_metric = pd.DataFrame()
            for index, df in data_frame_metric.items():
                index = str(index).split("-")
                #df["granularity"] = "_".join(index[:-1])
                df["type_model"] = index[-1]
                #df.set_index(['label'], inplace = True)
                data_metric = pd.concat([data_metric, df], axis = 0, ignore_index = False)

            data_metric = data_metric.reset_index(drop = True)
            print("\n >>> DataFrame: Metricas de Modelos <<<\n")
            print(data_metric.head())
            
            name_file = "data_Atom_agu" 
            # Validacion de la carpeta principal
            name_folder = "result/"
            self.functions.validate_path(name_folder)

            # Validacion de subcarpetas - nombre del archivo
            name_folder = name_folder + name_file + "/"
            self.functions.validate_path(name_folder)

            # Validacion de subcarpetas - periodo analisis (Semanal o Mensual)
            name_folder = name_folder + period + "/"
            self.functions.validate_path(name_folder)
            
            #data_metric.to_csv(self.path + "data_metrics.csv", index = False)
            self.queries.save_data_file_csv(data_metric, name_folder, name_file = name_file + "_metrics")

            end_time = time()
            print('\n >>>> El analisis tardo <<<<')
            self.functions.get_time_process(round(end_time - start_time, 2))

        except:
            print(traceback.format_exc())
        
if __name__ == "__main__":
    # Proceso de analisis
    col_gran = ['dept_nbr', 'store_nbr']
    col_serie = ['fecha', 'sales']
    col_obs_abc = ['price', "sales"]
    period = "month"
    series = Main_Model_Prediction(col_gran, col_serie, col_obs_abc, period)
    series.main()
    