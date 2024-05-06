import pandas as pd
import numpy as np
import os
import sys
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

class Main_Series_Times():
    
    def __init__(self, path):
        self.path = path

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

        # Extraccion de datos
        data = self.queries.get_data_file(name_file)

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
                
        return data, period, col_serie, col_gran

    # Modulo: Analisis y generacion de los patrones de demanda para la viabilidad de clasificacion de observaciones
    def data_demand(self, data, period, col_serie, col_gran):
        print(" >> DataFrame: Sales <<<\n")
        data[col_serie[0]] = pd.to_datetime(data[col_serie[0]])
        data['year'] = data[col_serie[0]].dt.year
        print(data.head())
        print("---"*30)
        
        #col_gran = ["state_id", "cat_id", "dept_id", "store_id"] #"item_id"
        years = data.year.unique().tolist()
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

                #df["date"] = pd.to_datetime(df["date"])
                #df['week'] = df["date"].dt.isocalendar().week
                #df['month'] = df["date"].dt.month
                #df['year'] = df["date"].dt.year
                
                df[col_serie[0]] = pd.to_datetime(df[col_serie[0]])
                df['week'] = df[col_serie[0]].dt.isocalendar().week
                df['month'] = df[col_serie[0]].dt.month
                df['year'] = df[col_serie[0]].dt.year
                print(df.head())
                print("---"*20)
                
                # Computo de series por la variables (Granuralidad)
                print("\n >>> DataFrame: Contador de Series por granularidad <<<\n")
                col = columns[:-1]
                #col.insert(len(columns), "year")
                col.insert(0, "year")
                #print(col)
                grouped = df.groupby(col)
                label = []
                count = []
                for idx, group in grouped:
                    #print(idx)
                    for col_name in group[col[1:]]:
                        group[col_name] = group[col_name].astype(str)

                    group['month'] = group[col_serie[0]].dt.month
                    #group["label"] = group[col[:-1]].apply("_".join, axis=1)
                    group["label"] = group[col[1:]].apply("_".join, axis = 1)
                    #label.append(str(list(idx)[:-1]).replace("[", "").replace("]", ""))
                    label.extend(group.label.unique().tolist())
                    count.append(len(group.month.unique().tolist()))
                    
                data_label = pd.DataFrame()
                data_label["label"] = label
                data_label["count_month"] = count
                data_label = data_label.groupby(["label"]).agg(months = ("count_month", 'sum')).reset_index()
                print(data_label.head())
                print("---"*20)
                
                # Proceso: Computo y generacion del intervalo medio de demanda - ADI
                print("\n >>> DataFrame: ADI <<<\n")
                columns = columns[:-1]
                columns.insert(len(columns), period)
                #print(columns)
                adi_data = self.functions.get_ADI(df, columns = columns)
                print(adi_data.head())
                print("---"*20)

                # Proceso: Compute y generacion del coeficiente de variacion - CV2
                print("\n >>> DataFrame: CV2 <<<\n")
                cv_data = self.functions.get_cv(df, columns = columns, potencia = 2)
                print(cv_data.head())
                print("---"*20)

                # Proceso: Union de los analisis del ADI y CV2
                print("\n >>> DataFrame: Final\n")
                df = pd.merge(adi_data[["label", "adi"]], cv_data[["label", "cv", "cv2"]], how = "left", on = ["label"])

                # Proceso: Clasificacion del tipo de categoria sobre el intervalo de demanda por ADI y CV2
                df["category"] = df.apply(self.functions.get_category, axis = 1)
                df = pd.merge(df, data_label, how = "left", on = ["label"])
                print(df.head(20))
                print("+++"*30)

                columns = columns[:-1]
                metric_name = str(year) + " - " + ' - '.join(columns)
                data_frame_metric[metric_name] = df
                
        return data_frame_metric

    def main(self):
        # Bandera de prueba
        test = True
        
        # Incializacion de clases (Metodo Contructor)
        self.queries = Queries(path, test)
        self.functions = Functions(path, test)

        # Seleccion del tipo de fuente para la extraccion de los datos
        source_data = self.functions.select_source_data()
        # Extraccion de datos por archivo y seleccion de variables analizar
        if source_data == 1:
            data, period, col_serie, col_gran = self.select_options()

        # Extraccion de datos por base de datos (Fijar las variables analizar, si no aplicar el modulo "select_options")
        elif source_data == 2:
            data = self.queries.get_data_netezza()
            
        else:
            print(" >>> Error: Seleccion de fuente incorrecta !!!\n")
            sys.exit()

        # Proceso: Clasificación de los patrones de demanda
        data_frame_metric = self.data_demand(data, period, col_serie, col_gran)

        # Proceso: Generacion de la estractura final del dataframe clasificacion en base a demanda
        print("\n >>> Dataframe: Demand Classifier <<< \n")
        data_final = self.functions.get_demand_classifier(data_frame_metric)
        print(data_final.head(30))
        print(" > Volumen: ", data_final.shape)
        print("---"*20)

        # Guardado del analisis - Dataframe intervalos de demanda
        if source_data == 1:
            self.queries.save_data_file(data_final)
            
        else:
            self.queries.save_data_bd(data_final)
        print("---"*20)

        # Proceso: Obtencion de los porcentajes por categoria
        print("\n >>> Dataframe: Detail Classifier <<< \n")
        detail_data_gran = self.functions.get_detail_demand(data_final)
        print(detail_data_gran)
        print("---"*20)

        # Guardado del analisis - Dataframe detail
        if source_data == 1:
            self.queries.save_data_file(data_final)
            
        else:
            self.queries.save_data_bd(data_final)
        print("---"*20)
        
if __name__ == "__main__":
    path = "C:/Softtek/timeseries/source/"

    # Proceso de analisis
    series = Main_Series_Times(path)
    series.main()