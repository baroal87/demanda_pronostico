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
from model.model import Model_Series_Times

import warnings
warnings.simplefilter("ignore", category = DeprecationWarning)
warnings.filterwarnings('ignore', message = 'Unverified HTTPS request')
warnings.filterwarnings('ignore', message = 'SettingWithCopyWarning')
warnings.simplefilter(action = 'ignore', category = FutureWarning)
os.environ["PYTHONIOENCODING"] = "utf-8"

pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None

class Main_Model_Prediction():
    
    # Modulo: Constructor
    def __init__(self, col_gran, col_serie, period):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.path = BASE_DIR + "/demanda_pronostico/source/"
        self.col_gran = col_gran
        self.col_serie = col_serie
        self.period = period

    def main(self):
        test = True
        try:
            # Incializacion de clases (Metodo Contructor)
            self.queries = Queries(self.path, test)
            self.functions = Functions(self.path, test)
            self.model = Model_Series_Times(self.path, test)

            name_file = "data_cat.csv"
            data = self.queries.get_data_file(name_file)
            
            columns = data.columns.tolist()
            index = len(self.col_gran) * 2
            col_cat = columns[-index:]
            fill_data = self.col_gran + self.col_serie + col_cat

            size = 2
            col_cat = [col_cat[i: i + size] for i in range(0, len(col_cat), size)]
            print(fill_data)
            print(col_cat)
            
            data = data[fill_data]
            data = data[data[col_serie[1]] > 0]
            data[self.col_serie[0]] = pd.to_datetime(data[self.col_serie[0]])
            data['week'] = data[self.col_serie[0]].dt.isocalendar().week
            data['month'] = data[self.col_serie[0]].dt.month
            data['year'] = data[self.col_serie[0]].dt.year
            print(data.head())
            print("---"*30)
            
            for segment in col_cat:
                #col = self.col_gran + segment
                col = segment.copy()
                #col.extend([self.period, "year"])
                col.append(self.col_serie[0])
                print(col)
                df = self.queries.grouped_data(data.copy(), col_grouped = col, var_obs = self.col_serie[1])
                df[segment] = df[segment].astype(str)
                df["segment"] = df[segment].apply("_".join, axis = 1)
                df[self.col_serie[1]] = df[self.col_serie[1]].astype(int)

                #df.drop(segment, axis = 1, inplace = True)
                #df = df.set_index(["index"])
                print(df.head())
                print("---"*30)
                #df[self.col_serie[0]] = df[self.col_serie[0]].astype(str)
                #nums = df[self.col_serie[0]].values.tolist()
                #dup = [x for i, x in enumerate(nums) if i != nums.index(x)]
                #print(dup)
                
                self.model.forecasting(df, var_time = self.col_serie[0], var_obs = self.col_serie[1], period = self.period)

                break

        except Exception as e:
            print(traceback.format_exc())
        
if __name__ == "__main__":
    # Proceso de analisis
    col_gran = ['dept_nbr', 'store_nbr']
    col_serie = ['fecha', 'sales']
    period = "month"
    series = Main_Model_Prediction(col_gran, col_serie, period)
    series.main()