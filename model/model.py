#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from prophet import Prophet

from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split, RepeatedStratifiedKFold
from feature_engine.selection import (DropFeatures, DropConstantFeatures, DropDuplicateFeatures, DropCorrelatedFeatures, 
                                      SmartCorrelatedSelection, SelectByShuffling, SelectBySingleFeaturePerformance, 
                                      RecursiveFeatureElimination)

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import plotly.graph_objects as go
import missingno as ms

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from prophet import Prophet
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  

from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
from scipy.stats import boxcox
from scipy.special import inv_boxcox

from math import log, exp, sqrt

from dateutil.relativedelta import relativedelta

#from pyInterDemand.algorithm.intermittent import plot_int_demand, classification, mase, rmse
#from pyInterDemand.algorithm.intermittent import croston_method

from statsforecast import StatsForecast
from statsforecast.models import CrostonClassic, ADIDA, IMAPA, CrostonSBA, TSB, AutoARIMA, CrostonOptimized
from datasetsforecast.losses import rmse

import skforecast
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

from statsmodels.tsa.seasonal import seasonal_decompose

#from numba import jit, cuda

#import torch

#use_cuda = torch.cuda.is_available()
#device = torch.device("cuda" if use_cuda else "cpu")
#print("Device: ",device)

class Model_Series_Times():

    # Modulo: Constructor
    def __init__(self, path, test):
        self.path = path
        self.test = test

        self.scaler = StandardScaler()
        self.OneHot = OneHotEncoder(handle_unknown = 'ignore')
        self.encoder = OrdinalEncoder()
        self.labelencoder = LabelEncoder()

    # Modulo:
    def get_metrics_pred(self, X_test, name_col_real, name_col_pred):
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
        mae = round(mean_absolute_error(X_test[name_col_real], X_test[name_col_pred]), 2)
        rmse = round((np.mean((X_test[name_col_real] - X_test[name_col_pred])**2))**.5, 2)
        #rmse = root_mean_squared_error(X_test[series_col_real], X_test[series_col_pred])
        #mape = np.sum(np.abs(X_test[series_col_pred] - X_test[series_col_real])) / (np.sum((np.abs(X_test[series_col_real]))))
        mape = round(mean_absolute_percentage_error(X_test[name_col_real], X_test[name_col_pred]), 2)
        acc = round(X_test[(X_test.accuracy > 0) & (X_test[name_col_real] > 0)].accuracy.mean(), 2)
        
        return mae, rmse, mape, acc

    # Modulo: Metrica de presicion
    def forecast_accuracy(self, sale, fcst):
        acc =  1 - abs(fcst - sale) / ((fcst + sale) / 2)

        return acc

    # Modulo: Identificador del tipo de variables (Categoricas y numericas)
    def get_segmetation_variables(self, data, columns_name):
        # Determinacion de las columnas categoricas y numericas
        columns_num = []
        columns_cat = []
        for col in columns_name:
            #print(" >> Col: {} -> {}".format(col, data[col].dtype))
            format_col = data[col].dtype
            if (format_col == "int64") | (format_col == int) | (format_col == "float64") | (format_col == float):
                columns_num.append(col)

            elif (format_col == object) | (format_col == str):
                columns_cat.append(col)

        return columns_num, columns_cat

    # Modulo: Tratamiento y procesamiento de datos normalizados
    def get_data_tranform_normal(self, data, numeric_features, categorical_features):

        if (len(numeric_features) != 0) & (len(categorical_features) == 0):
            #data_num = self.scale.fit_transform(data[numeric_features])
            data_num = pd.DataFrame(self.scale.fit_transform(data[numeric_features]), columns = self.scale.get_feature_names_out())
            #print(data_num.head())
            #print("---"*20)

            return data_num

        elif (len(numeric_features) == 0) & (len(categorical_features) != 0):
            # OneHot Encoder
            #data_cat = self.OneHot.fit_transform(data[categorical_features]).toarray()
            #transformer = make_column_transformer((self.OneHot, self.categorical_features), remainder = 'passthrough')
            #data_cat = pd.DataFrame(data_cat, columns = self.OneHot.get_feature_names_out())

            # Encoder
            #data_cat = pd.DataFrame(self.encoder.fit_transform(data[categorical_features]), columns = self.encoder.get_feature_names_out())

            # Label Encoder
            data_cat = pd.DataFrame()
            for label in categorical_features:
                data_cat[label] = self.labelencoder.fit_transform(data[label])
            #print(data_cat.head())
            #print("---"*20)
            #data_cat = pd.DataFrame(self.scale.fit_transform(data_cat), columns = self.scale.get_feature_names_out())

            return data_cat

        else:
            #data_cat = self.OneHot.fit_transform(data[categorical_features]).toarray()
            #data = pd.concat([pd.DataFrame(data_cat, columns = self.OneHot.get_feature_names_out()), 
            #data = pd.concat([pd.DataFrame(self.encoder.fit_transform(data[categorical_features]), columns = self.encoder.get_feature_names_out()), 
                             #pd.DataFrame(self.scale.fit_transform(data[numeric_features]), columns = self.scale.get_feature_names_out())], 
                             #axis = 1, ignore_index = False)

            data_dummy = pd.DataFrame(self.scale.fit_transform(data[numeric_features]), columns = self.scale.get_feature_names_out())
            for label in categorical_features:
                data_dummy[label] = self.labelencoder.fit_transform(data[label])

            data_dummy = pd.DataFrame(self.scale.fit_transform(data_dummy[categorical_features]), columns = self.scale.get_feature_names_out())

            return data_dummy

    # Modulo: Busqueda de hyperparametros
    def hyperparameters(self, X_train, Y_train, numeric_features, categorical_features, type_model = "LGBM"):
        # Procesamiento para tratamientos de datos numericos
        numeric_transformer = Pipeline(steps = [('scaler', StandardScaler()),
                                                ('drop_constant_values', DropConstantFeatures(tol = 1, missing_values = 'ignore')),
                                                #('drop_correlated', DropCorrelatedFeatures(variables = None, method = 'spearman', threshold = 0.7)) #pearson, spearman or kendal
                                                #('drop_duplicates', DropDuplicateFeatures())
                                                ])

        # Procesamiento para tratamientos de datos categoricos
        categorical_transformer = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False, drop = 'if_binary')),
                                                    ('drop_constant_values', DropConstantFeatures(tol = 1, missing_values = 'ignore')),
                                                    ('drop_correlated', DropCorrelatedFeatures(variables = None, method = 'pearson', threshold = 0.7)),
                                                    ('drop_duplicates', DropDuplicateFeatures())])

        preprocessor = ColumnTransformer(transformers = [('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

        # Busqueda de hyperparametros mediante modelo LGBM
        if type_model == "LGBM":
            #print("\n >> Model: ", type_model)
            params = {"num_leaves": [300, 350, 400, 450, 500, 550, 600], #st.randint(300, 600),
                        'min_data_in_leaf': [10, 12, 14, 16, 18, 20], #st.randint(10, 20),
                        'max_depth': [20, 22, 24, 26, 28, 30], #st.randint(20, 30),
                        'max_bin': [10, 12, 14, 16, 18, 20], #st.randint(10, 20),
                        'learning_rate': [0.01, 0.1, 1], #st.uniform(0.001, 1),
                        'n_estimators': [80, 90, 100, 110, 120, 130, 140, 150], #st.randint(80, 150), 
                        'boosting_type': ['gbdt', 'dart'], #, 'rf'
                        'metric': ['l2'],
                        'objective': ['regression'],
                        'feature_fraction': [0.7],
                        'bagging_fraction': [0.6, 0.7, 0.8],
                        #"num_threads": [6],
                        'device': ['gpu'],
                        #"force_col_wise": [True],
                        "verbose": [-1]}

            lgbm = LGBMRegressor()
            #grid_ = GridSearchCV(lgbm, params, scoring = 'neg_mean_squared_error', cv = 5)
            grid_ = RandomizedSearchCV(lgbm, params, scoring = 'neg_mean_squared_error', cv = 3)
            hyper = Pipeline(steps = [('preprocessor', preprocessor), ('classifier', grid_)])
            hyper.fit(X_train, Y_train)

            # Retorna los mejores parametros utilizados para el proceso de entrenamiento
            #print("BEST_SCORE_GOT: ", hyper.best_score_)
            print("\n The best parameters:\n", grid_.best_params_)
            best_params = grid_.best_params_
            
        # Busqueda de hyperparametros mediante modelo CatBoost
        elif type_model == "CatBoost":
            #print("\n >> Model: ", type_model)
            params = {"allow_const_label": [True], # [True], 
                    "depth": [6, 8], # 6
                    #"eval_metric": "RMSE", # ["RMSE"], 
                    "iterations": [1000, 1400], # 1000
                    "learning_rate": [0.5, 0.01], # 0.3
                    "metric_period": [8], # 8
                    #"od_type": "Iter", # ["Iter"], 
                    #"od_wait": 100, #[100],
                    "one_hot_max_size": [40, 60], # 40
                    "l2_leaf_reg": [10, 15], # 15
                    #"thread_count": [6], #10
                    "task_type": ["GPU"],
                    "logging_level": ["Silent"]}
            
            cat = CatBoostRegressor()
            #rsf = RepeatedStratifiedKFold(random_state = 6)
            grid_ = RandomizedSearchCV(cat, params, scoring = 'neg_mean_squared_error', cv = 3)
            hyper = Pipeline(steps = [('preprocessor', preprocessor), ('regressor', grid_)])
            hyper.fit(X_train, Y_train)

            # Retorna los mejores parametros utilizados para el proceso de entrenamiento
            #print("BEST_SCORE_GOT (CatBoost): ", grid_.best_score_)
            print("\n The best parameters (CatBoost):\n", grid_.best_params_)
            best_params = grid_.best_params_

        return best_params, preprocessor

    #@jit(target_backend='cuda', nopython=True)
    # Modulo: Modelo (Light Gradient-Boosting Machine - LGBM) para la prediccion de forecast
    def get_model_LGBM(self, data, columns_num, columns_cat, col_pred):
        column = columns_cat + columns_num
        data = data.dropna().reset_index(drop = True)
        if col_pred in column:
            column.remove(col_pred)
            columns_num.remove(col_pred)

        Y = data[col_pred]
        X = data[column]

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 6)
        new_params, preprocessor = self.hyperparameters(X_train, Y_train, columns_num, columns_cat, type_model = "LGBM")

        model = LGBMRegressor(**new_params)
        train_model = Pipeline(steps = [('preprocessor', preprocessor), ('model', model)])
        train_model.fit(X_train, Y_train)

        pred = train_model.predict(X_test)
        coef_det = r2_score(Y_test, pred, multioutput = 'variance_weighted')#['raw_values', 'uniform_average', 'variance_weighted'])
        if coef_det < 0:
            coef_det = abs(coef_det)
            
        name_col_real = col_pred + "_real"
        name_col_pred = col_pred + "_pred"
        X_test[name_col_real] = Y_test
        X_test[name_col_pred] = pred
        X_test.loc[X_test[name_col_pred] < 0 , 'sale_pred'] = 0
        X_test[name_col_pred] = X_test[name_col_pred].round(2)
        #X_test[name_col_pred] = X_test[name_col_pred].astype(int)

        # Metrica para evaluar la presicion del modelo vs real en base a formula de negocio
        X_test["accuracy"] = self.forecast_accuracy(X_test[name_col_real], X_test[name_col_pred])
        X_test.loc[X_test.accuracy.isnull(), "accuracy"] = 0
        mae, rmse, mape, acc = self.get_metrics_pred(X_test, name_col_real, name_col_pred)

        data_metric = pd.DataFrame()
        data_metric["mae"] = [mae]
        data_metric["rmse"] = [rmse]
        data_metric["mape"] = [mape]
        data_metric["acc"] = [acc]
        data_metric["r2"] = [round(coef_det, 2)]

        return data_metric
    
    #@jit(target_backend='cuda', nopython=True)
    # Modulo: Modelo Catboost
    def get_model_CatBoost(self, data, columns_num, columns_cat, col_pred):
        # Concatenacion de las valriables para las filtraciones del historico
        column = columns_cat + columns_num
        if col_pred in column:
            column.remove(col_pred)
            columns_num.remove(col_pred)

        # Particionamiento de variables predictorias y variable a predecir
        data = data.dropna().reset_index(drop = True)
        Y = data[col_pred]
        X = data[column]

        # Generacion del conjunot de entrenamiento y prueba
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 6)
        #X_train.drop("customer_id", axis = 1, inplace = True)
        
        # Procesos de paralelizamiento para la busqueda de hyperparametros
        new_params, preprocessor = self.hyperparameters(X_train, Y_train, columns_num, columns_cat, type_model = "CatBoost")

        # Obtencion de las clasificaciones en base a los hyperparametros
        model = CatBoostRegressor(**new_params) # loss_function = 'RMSE'
        train_model = Pipeline(steps = [('preprocessor', preprocessor), ('model', model)])
        train_model.fit(X_train, Y_train)
        #pred = train_model.predict(X_test.drop("customer_id", axis = 1))
        pred = train_model.predict(X_test)
        coef_det = r2_score(Y_test, pred, multioutput = 'variance_weighted')#['raw_values', 'uniform_average', 'variance_weighted'])
        if coef_det < 0:
            coef_det = abs(coef_det)

        name_col_real = col_pred + "_real"
        name_col_pred = col_pred + "_pred"
        X_test[name_col_real] = Y_test
        X_test[name_col_pred] = pred
        X_test.loc[X_test[name_col_pred] < 0 , 'sale_pred'] = 0
        X_test[name_col_pred] = X_test[name_col_pred].round(2)

        # Metrica para evaluar la presicion del modelo vs real en base a formula de negocio
        X_test["accuracy"] = self.forecast_accuracy(X_test[name_col_real], X_test[name_col_pred])
        X_test.loc[X_test.accuracy.isnull(), "accuracy"] = 0
        mae, rmse, mape, acc = self.get_metrics_pred(X_test, name_col_real, name_col_pred)

        data_metric = pd.DataFrame()
        data_metric["mae"] = [mae]
        data_metric["rmse"] = [rmse]
        data_metric["mape"] = [mape]
        data_metric["acc"] = [acc]
        data_metric["r2"] = [round(coef_det, 2)]

        return data_metric

    #@jit(target_backend='cuda', nopython=True)
    # Modulo: Modelo Prophet
    def get_model_prophet(self, data, col_serie):
        data = data.dropna().reset_index(drop = True)
        total_data = len(data)
        fill_data = int(round((total_data * 20) / 100))
        #print(fill_data)

        #date_max = data[col_serie[0]].max()
        #date_min = data[col_serie[0]].min()
        #dif = date_max - date_min

        #days = int((dif.days * 20) / 100)
        #start_date = date_max + relativedelta(days = -days)
        #print(start_date)

        #train = data[data["date"] <= start_date]
        #test = data[data["date"] > start_date]
        train = data.loc[:fill_data]
        name_col_transf = col_serie[1] + "_scaled"
        transformed, lam = boxcox(train[col_serie[1]])
        flag_boxcox = False
        if lam < -5:
            train[name_col_transf], lam = boxcox(train[col_serie[1]])
            #train[name_col_transf] = self.scaler.fit_transform(train[name_col_transf])
            flag_boxcox = True

        else:
            train[name_col_transf] = self.scaler.fit_transform(train[[col_serie[1]]])

        #train = train[[col_serie[0], name_col_transf]].set_index([col_serie[0]])

        test = data.loc[fill_data:]
        test = test[[col_serie[0], col_serie[1]]]
        #test[name_col_transf] = self.scaler.fit_transform(test[[col_serie[1]]])

        # seasonality_mode = 'multiplicative', daily_seasonality = True, weekly_seasonality = True
        model_fbp = Prophet()
        #model_fbp.add_seasonality('daily', period = 1, fourier_order = 3)
        #model_fbp.add_regressor('regressor', mode = 'additive')
        #model_fbp.fit(train[[col_serie[0], col_serie[1]]].rename(columns = {col_serie[0]: "ds", col_serie[1]: "y"}))
        #forecast = model_fbp.predict(test[[col_serie[0], col_serie[1]]].rename(columns = {col_serie[0]: "ds"}))

        model_fbp.fit(train[[col_serie[0], name_col_transf]].rename(columns = {col_serie[0]: "ds", name_col_transf: "y"}))
        #forecast = model_fbp.predict(test[[col_serie[0], name_col_transf]].rename(columns = {col_serie[0]: "ds"}))
        forecast = model_fbp.predict(test[[col_serie[0], col_serie[1]]].rename(columns = {col_serie[0]: "ds"}))
        #future_data = model_fbp.make_future_dataframe(periods = len(test))
        #forecast = model_fbp.predict(future_data)
        #print(len(test))

        name_col_real = col_serie[1] + "_real"
        name_col_pred = col_serie[1] + "_pred"
        test[name_col_pred] = forecast.yhat.values

        if flag_boxcox:
            test[name_col_pred] = inv_boxcox(test[name_col_pred], lam)

        else:
            test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])

        test[name_col_pred] = test[name_col_pred].round(2)
        test.rename(columns = {col_serie[1]: name_col_real}, inplace = True)
        
        test["accuracy"] = self.forecast_accuracy(test[name_col_real], test[name_col_pred])
        test.loc[test["accuracy"].isnull(), "accuracy"] = 0
        test["accuracy"] = test["accuracy"].round(2)
        mae, rmse, mape, acc = self.get_metrics_pred(test, name_col_real, name_col_pred)
        
        coef_det = r2_score(test[name_col_real], test[name_col_pred], multioutput = 'variance_weighted')
        if coef_det < 0:
            coef_det = abs(coef_det)

        data_metric = pd.DataFrame()
        data_metric["mae"] = [mae]
        data_metric["rmse"] = [rmse]
        data_metric["mape"] = [mape]
        data_metric["acc"] = [acc]
        data_metric["r2"] = [round(coef_det, 2)]
        #print(test.head(20))

        return data_metric

    # Modulo: invert box-cox transform
    def boxcox_inverse(self, value, lam):
        if lam == 0:
            return exp(value)

        return exp(log(lam * value + 1) / lam)
    
    # Modulo: Evaluacion del modelo ARIMA para un orden dado (p,d,q) y devolver RMSE
    def evaluate_arima_model(self, X, arima_order):
        X = X.astype('float32')
        
        # Computo de particionamiento
        train_size = int(len(X) * 0.10)
        #print(len(X))
        #print(train_size)
        train, test = X[0:train_size], X[train_size:]
        history = [x for x in train]

        # Computo de prediciones
        predictions = list()
        for t in range(len(test)):
            model = ARIMA(history, order = arima_order)
            model_fit = model.fit(method_kwargs = {"warn_convergence": False})
            yhat = model_fit.forecast()[0]
            predictions.append(yhat)
            history.append(test[t])

        # Computo de metrica
        rmse = sqrt(mean_squared_error(test, predictions))
        return rmse

    # Modulo: Evaluacion de combinaciones de valores p, d y q para un modelo ARIMA
    def evaluate_models(self, dataset, p_values, d_values, q_values):
        dataset = dataset.astype('float32')
        best_score, best_cfg = float("inf"), None
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    order = (p, d, q)

                    try:
                        # Proceso de vealuacion del modelo en base a las condiciones p, d, q
                        rmse = self.evaluate_arima_model(dataset, order)
                        if rmse < best_score:
                            best_score, best_cfg = rmse, order

                        print('ARIMA%s RMSE=%.3f' % (order,rmse))

                    except:
                        continue

        print('Mejor ARIMA: %s RMSE = %.3f' % (best_cfg, best_score))
        return best_cfg

    # Modulo: Modelo Auto Arima
    def get_model_Arima(self, data, col_serie):
        data = data.dropna().reset_index(drop = True)
        total_data = len(data)
        fill_data = int(round((total_data * 20) / 100))

        train = data.loc[:fill_data]
        name_col_transf = col_serie[1] + "_scaled"
        transformed, lam = boxcox(train[col_serie[1]])
        flag_boxcox = False
        if lam < -5:
            train[name_col_transf], lam = boxcox(train[col_serie[1]])
            #train[name_col_transf] = self.scaler.fit_transform(train[name_col_transf])
            flag_boxcox = True
            
        else:
            train[name_col_transf] = self.scaler.fit_transform(train[[col_serie[1]]])

        #train = train[[col_serie[0], col_serie[1]]].set_index([col_serie[0]])
        train = train[[col_serie[0], name_col_transf]].set_index([col_serie[0]])

        test = data.loc[fill_data:]
        test = test[[col_serie[0], col_serie[1]]]
        
        p_values = range(0,2)
        d_values = range(0,3)
        q_values = range(0,2)
        print("\n >> Proceso: Extraccion de parametros - p, d, q")
        #best_cfg = self.evaluate_models(data[col_serie[1]].values, p_values, d_values, q_values)
        best_cfg = self.evaluate_models(train.values.copy(), p_values, d_values, q_values)
        #best_cfg = list(best_cfg)
        #order = (p, d, q)
        #print(lam)
        #print(best_cfg)
        
        model = ARIMA(train, order = best_cfg)
        model_arima = model.fit()
        #pred = model_arima.predict(n_periods = len(test))
        #print(len(test))
        #print(model_arima.predict())
        pred = model_arima.forecast(len(test))
        
        name_col_real = col_serie[1] + "_real"
        name_col_pred = col_serie[1] + "_pred"
        test[name_col_pred] = pred.tolist()
        #test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])

        if flag_boxcox:
            test[name_col_pred] = inv_boxcox(test[name_col_pred], lam)
            
        else:
            test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])

        test[name_col_pred] = test[name_col_pred].round(2)
        test.rename(columns = {col_serie[1]: name_col_real}, inplace = True)
        
        test["accuracy"] = self.forecast_accuracy(test[name_col_real], test[name_col_pred])
        test.loc[test["accuracy"].isnull(), "accuracy"] = 0
        test["accuracy"] = test["accuracy"].round(2)
        mae, rmse, mape, acc = self.get_metrics_pred(test, name_col_real, name_col_pred)
        
        coef_det = r2_score(test[name_col_real], test[name_col_pred], multioutput = 'variance_weighted')
        if coef_det < 0:
            coef_det = abs(coef_det)

        data_metric = pd.DataFrame()
        data_metric["mae"] = [mae]
        data_metric["rmse"] = [rmse]
        data_metric["mape"] = [mape]
        data_metric["acc"] = [acc]
        data_metric["r2"] = [round(coef_det, 2)]
        #print(test.head(20))

        return data_metric

    # Modulo: Modelo Auto Arima
    def get_model_autoarima(self, data, col_serie):
        data = data.dropna().reset_index(drop = True)
        days = int((len(data) * 20) / 100)

        #date_max = data["date"].max()
        #start_date = date_max + relativedelta(days = -days)

        #train = data[data["date"] <= start_date]
        #train = train[["date", "sales"]].set_index(["date"])

        #test = data[data["date"] > start_date]
        #test = test[["date", "sales"]].set_index(["date"])

        total_data = len(data)
        fill_data = int(round((total_data * 20) / 100))

        train = data.loc[:fill_data]
        name_col_transf = col_serie[1] + "_scaled"
        transformed, lam = boxcox(train[col_serie[1]])
        flag_boxcox = False
        if lam < -5:
            train[name_col_transf], lam = boxcox(train[col_serie[1]])
            #train[name_col_transf] = self.scaler.fit_transform(train[name_col_transf])
            flag_boxcox = True
            
        else:
            train[name_col_transf] = self.scaler.fit_transform(train[[col_serie[1]]])

        #train = train[[col_serie[0], col_serie[1]]].set_index([col_serie[0]])
        train = train[[col_serie[0], name_col_transf]].set_index([col_serie[0]])

        test = data.loc[fill_data:]
        test = test[[col_serie[0], col_serie[1]]]
        #print(test[col_serie[0]].min())
        #print(test[col_serie[0]].max())
        #test = test[[col_serie[0], col_serie[1]]].set_index(["date"])

        model = auto_arima(train,
                    start_p = 0,
                    start_q = 0,
                    test = "adf", # Busqueda del optimo valor d
                    max_p = 3, 
                    max_q = 3,
                    d = None,
                    max_d = 3,
                    max_P = 2, 
                    max_D = 2,
                    max_Q = 2,
                    seasonal = True,
                    trace = True,
                    error_action = "ignore",
                    suppress_warnings = True,
                    alpha = 0.05,
                    scoring = 'mse',
                    call_me = 'arima3',
                    information_criterion = "aic",
                    stepwise = True)

        model_autoarima = model.fit(train)
        #model_autoarima.summary()
        pred = model_autoarima.predict(n_periods = len(test))

        name_col_real = col_serie[1] + "_real"
        name_col_pred = col_serie[1] + "_pred"
        test[name_col_pred] = pred.tolist()
        #test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])

        if flag_boxcox:
            test[name_col_pred] = inv_boxcox(test[name_col_pred], lam)
            
        else:
            test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])

        test[name_col_pred] = test[name_col_pred].round(2)
        test.rename(columns = {col_serie[1]: name_col_real}, inplace = True)
        
        test["accuracy"] = self.forecast_accuracy(test[name_col_real], test[name_col_pred])
        test.loc[test["accuracy"].isnull(), "accuracy"] = 0
        test["accuracy"] = test["accuracy"].round(2)
        mae, rmse, mape, acc = self.get_metrics_pred(test, name_col_real, name_col_pred)
        
        coef_det = r2_score(test[name_col_real], test[name_col_pred], multioutput = 'variance_weighted')
        if coef_det < 0:
            coef_det = abs(coef_det)

        data_metric = pd.DataFrame()
        data_metric["mae"] = [mae]
        data_metric["rmse"] = [rmse]
        data_metric["mape"] = [mape]
        data_metric["acc"] = [acc]
        data_metric["r2"] = [round(coef_det, 2)]
        #print(test.head(20))

        return data_metric

    # Modulo:
    def get_models_statsForecast(self, data, var_obs):
        data = data.dropna().reset_index(drop = True)
        name_col_transf = var_obs[1] + "_scaled"
        transformed, lam = boxcox(data[var_obs[1]])
        flag_boxcox = False
        if lam < -5:
            data[name_col_transf], lam = boxcox(data[var_obs[1]])
            #train[name_col_transf] = self.scaler.fit_transform(train[name_col_transf])
            flag_boxcox = True

        else:
            data[name_col_transf] = self.scaler.fit_transform(data[[var_obs[1]]])

        #data[name_col_transf] = self.scaler.fit_transform(data[[var_obs[1]]])
        #data.rename(columns = {var_obs[0]: "ds", var_obs[1]: "y"}, inplace = True)
        
        data.rename(columns = {var_obs[0]: "ds", name_col_transf: "y"}, inplace = True)
        data["ds"] = pd.to_datetime(data["ds"])
        data["unique_id"] = data.segment
        #data.drop("segment", axis = 1, inplace = True)

        #data = data[["unique_id", "ds", "y"]]
        data = data.dropna().reset_index(drop = True)
        total_data = len(data)
        fill_data = int(round((total_data * 20) / 100))
        train = data.loc[:fill_data]
        train = train[["unique_id", "ds", "y"]]
        test = data.loc[fill_data:]
        test = test[["unique_id", "ds", var_obs[1]]]

        season_length = 24 # Hourly data lam
        horizon = len(test) # number of predictions
        #models = [CrostonClassic(), CrostonSBA(), CrostonOptimized(), ADIDA(), IMAPA(), TSB(0.2, 0.2), AutoARIMA(season_length = season_length, blambda = lam)]
        models = [AutoARIMA(season_length = season_length), CrostonSBA(), CrostonClassic()]

        sf = StatsForecast(df = train, models = models, freq = 'D', n_jobs = -1)
        sf.fit()
        forecasts = sf.forecast(h = horizon, fitted = True)
        #values = sf.forecast_fitted_values()
        pred = sf.predict(h = horizon)
        cv_df = sf.cross_validation(df = train, h = season_length, step_size = season_length, n_windows = 5)
        #print(cv_df.head(20))
        #print(cv_df.shape)
        #print("---"*30)
        forecasts = forecasts.reset_index()

        name_col_pred = var_obs[1] + "_pred"
        name_col_real = var_obs[1] + "_real"
        #forecasts["AutoARIMA"] = self.scaler.inverse_transform(forecasts[["AutoARIMA"]])

        test = pd.merge(test, forecasts, how = "left", on = ["unique_id", "ds"])
        #test.rename(columns = {"AutoARIMA": name_col_pred, var_obs[1]: name_col_real}, inplace = True)
        test.rename(columns = {var_obs[1]: name_col_real}, inplace = True)
        test = test.dropna().reset_index(drop = True)
    
        #test.loc[test[name_col_pred].isnull(), name_col_pred] = 0
        #test[name_col_pred] = test[name_col_pred].abs()
        #test[name_col_pred] = test[name_col_pred].round(2)

        #rmse = rmse(cv_df['y'], cv_df['CrostonClassic'])
        #print(rmse)
        #mae = mean_absolute_error(cv_df['y'], cv_df['AutoARIMA'])
        #print(mae)
        #print("*"*30)
        
        data_metric = pd.DataFrame()
        size_model = len(models)
        col_pred = test.columns.tolist()[-size_model:]
        #print(col_pred)
        for col in col_pred:
            if flag_boxcox:
                #test[name_col_pred] = inv_boxcox(test[name_col_pred], lam)
                test[col] = inv_boxcox(test[col], lam)

            else:
                #test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])
                test[col] = self.scaler.inverse_transform(test[[col]])

            #test.loc[test[col].isnull(), col] = 0
            test[col] = test[col].abs()
            test[col] = test[col].round(2)

            test["accuracy"] = self.forecast_accuracy(test[name_col_real], test[col])
            test.loc[test["accuracy"].isnull(), "accuracy"] = 0
            test["accuracy"] = test["accuracy"].round(2)
            mae, rmse, mape, acc = self.get_metrics_pred(test, name_col_real, col)

            coef_det = r2_score(test[name_col_real], test[col], multioutput = 'variance_weighted')
            if coef_det < 0:
                coef_det = abs(coef_det)

            temp = pd.DataFrame()
            temp["model"] = [col]
            temp["mae"] = [mae]
            temp["rmse"] = [rmse]
            temp["mape"] = [mape]
            temp["acc"] = [acc]
            temp["r2"] = [round(coef_det, 2)]
        
            data_metric = pd.concat([data_metric, temp], axis = 0, ignore_index = False)

        #print(test.head(20))
        #print("---"*30)
        data_metric = data_metric.reset_index(drop = True)

        return data_metric, col_pred

    # Modulo:
    def get_model_forecasters(self, data, var_obs):
        data = data.dropna().reset_index(drop = True)
        total_data = len(data)
        fill_data = int(round((total_data * 20) / 100))

        train = data.loc[:fill_data]
        name_col_transf = var_obs[1] + "_scaled"
        train[name_col_transf] = self.scaler.fit_transform(train[[var_obs[1]]])
        train = train[[var_obs[0], name_col_transf]].set_index([var_obs[0]])
        train.rename(columns = {name_col_transf: "y"}, inplace = True)

        test = data.loc[fill_data:]
        test = test[[var_obs[0], var_obs[1]]]

        forecaster = ForecasterAutoreg(regressor = DecisionTreeRegressor(random_state = 123), lags = 30)
        # Fit the model using train data
        forecaster.fit(y = train["y"])
        
        # Parameter Grid for Regressor
        param_grid = {'max_depth' : [None, 1, 3, 5],
                        'min_samples_split' : [2, 3, 4],
                        'ccp_alpha' : [0.0, 0.001, 0.01]}

        # Lags Grid
        lags_grid = [30]

        # Grid Search with Refit and Increasing Train Size
        grid_forecaster = grid_search_forecaster(
            forecaster = forecaster,
            y = train["y"],
            param_grid = param_grid,
            lags_grid = lags_grid,
            steps = 30,
            refit = True,
            metric = 'mean_squared_error',
            initial_train_size = len(train),
            fixed_train_size = False,
            return_best = True,
            n_jobs = 'auto',
            verbose = False)
        print("#"*30)
        print(grid_forecaster)
        print("#"*30)

        # Predict the test period
        predicted_test = forecaster.predict(steps = len(test))
        print(predicted_test.head(10))
        print()
        name_col_pred = var_obs[1] + "_pred"
        name_col_real = var_obs[1] + "_real"
        test[name_col_pred] = predicted_test
        test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])
        test[name_col_pred] = test[name_col_pred].round(2)
        #test.rename(columns = {col_serie[1]: name_col_real}, inplace = True)
        print(test.head(10))
        print("---"*30)

        forecaster = ForecasterAutoreg(regressor = RandomForestRegressor(random_state = 123), lags = 6)
        forecaster.fit(y = train['y'])
        
        # Candidate values for lags
        lags_grid = [30]

        # Candidate values for regressor's hyperparameters
        param_grid = {'n_estimators': [100, 250],
                        'max_depth': [3, 5, 10]}

        grid_forecaster = grid_search_forecaster(
                        forecaster = forecaster,
                        y = train['y'],
                        param_grid = param_grid,
                        lags_grid = lags_grid,
                        steps = 30,
                        refit = False,
                        metric = 'mean_squared_error',
                        #initial_train_size  int(len(train)*0.5),
                        fixed_train_size = False,
                        return_best = True,
                        n_jobs  = 'auto',
                        verbose = False)
        print("#"*30)
        print(grid_forecaster)
        print("#"*30)

        predicted_test = forecaster.predict(steps = len(test))
        print(predicted_test.head(10))
        print()
        #print(type(predicted_test))

        name_col_pred = var_obs[1] + "_pred"
        name_col_real = var_obs[1] + "_real"
        test[name_col_pred] = predicted_test
        test[name_col_pred] = self.scaler.inverse_transform(test[[name_col_pred]])
        test[name_col_pred] = test[name_col_pred].round(2)
        #test.rename(columns = {col_serie[1]: name_col_real}, inplace = True)
        print(test.head(10))