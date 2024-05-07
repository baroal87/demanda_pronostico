import pandas as pd
import numpy as np

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
        data = pd.read_csv(self.path + name_file)

        return data

    # Modulo: Agrupamiento de datos por la variable observacion seleccionada
    def grouped_data(self, data, col_grouped, var_obs):
        #data = data.groupby(col_grouped).agg({'sales': 'sum'}).reset_index()
        data = data.groupby(col_grouped).agg({var_obs: 'sum'}).reset_index()
    
        name_col = var_obs + "_sum"
        #data.rename(columns = {'sales_sum': 'sales'}, inplace = True)
        data.rename(columns = {name_col: var_obs}, inplace = True)

        return data

    # Modulo: Determinacion del nombre y guardado del dataframe resultante (Archivo)
    def save_data_file(self, data, name_folder):
        if not self.test:
            print("\n >> Proceso de guardado (Archivo)")
            name_file = input("\n Ingrese el nombre del archivo: ")
            while True:
                validate = input('\n Nombre del archivo es correcto (y / n): ')
                if validate.lower() == "y":
                    break
                
                elif (validate.lower() != "y") & (validate.lower() != "n"):
                    print("\n > Opcion invalida. Seleccione: y -> si o n -> no !!! \n")
                    
                elif validate.lower() == "n":
                    name_file = input("\n Ingrese el nombre del archivo: ")

            #path = self.path + "result/"
            path = self.path + name_folder
            data.to_csv(path + name_file + ".csv", index = False)
            print(" >> Archivo Guardado correctamente")
            
    # Modulo: Determinacion del nombre y guardado del dataframe resultante (BD)
    def save_data_bd(self, data):
        if not self.test:
            print("\n >> Proceso de guardado (Base datos)")