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

#from pyInterDemand.algorithm.intermittent import plot_int_demand, classification, mase, rmse
#from pyInterDemand.algorithm.intermittent import croston_method

from statsforecast import StatsForecast
from statsforecast.models import CrostonClassic, ADIDA, IMAPA, CrostonSBA, TSB, AutoARIMA

from sklearn.metrics import mean_absolute_error
from datasetsforecast.losses import rmse


class Model_Series_Times():

    # Modulo: Constructor
    def __init__(self, path, test):
        self.path = path
        self.test = test

        self.scale = StandardScaler()
        self.OneHot = OneHotEncoder(handle_unknown = 'ignore')
        self.encoder = OrdinalEncoder()
        self.labelencoder = LabelEncoder()

    # Modulo: Metrica de presicion
    def forecast_accuracy(sale, fcst):
        acc =  1 - abs(fcst - sale) / ((fcst + sale) / 2)

        return acc
    
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
    def hyperparameters(X_train, Y_train, numeric_features, categorical_features, type_model = "LGBM"):
        # Procesamiento para tratamientos de datos numericos
        numeric_transformer = Pipeline(steps = [('scaler', StandardScaler()),
                                                ('drop_constant_values', DropConstantFeatures(tol = 1, missing_values = 'ignore')),
                                                ('drop_duplicates', DropDuplicateFeatures()),
                                                ('drop_correlated', DropCorrelatedFeatures(variables = None, method = 'spearman', threshold = 0.7)) #pearson, spearman or kendal
                                                ])

        # Procesamiento para tratamientos de datos categoricos
        categorical_transformer = Pipeline(steps = [('onehot', OneHotEncoder(handle_unknown = 'ignore', sparse = False, drop = 'if_binary')),
                                                    ('drop_constant_values', DropConstantFeatures(tol = 1, missing_values = 'ignore')),
                                                    ('drop_correlated', DropCorrelatedFeatures(variables = None, method = 'pearson', threshold = 0.7)),
                                                    ('drop_duplicates', DropDuplicateFeatures())])

        preprocessor = ColumnTransformer(transformers = [('num', numeric_transformer, numeric_features), ('cat', categorical_transformer, categorical_features)])

        # Busqueda de hyperparametros mediante modelo LGBM
        if type_model == "LGBM":
            print("\n >> Model: ", type_model)
            params = {"num_leaves": [300, 350, 400, 450, 500, 550, 600], #st.randint(300, 600),
                        'min_data_in_leaf': [10, 12, 14, 16, 18, 20], #st.randint(10, 20),
                        'max_depth': [20, 22, 24, 26, 28, 30], #st.randint(20, 30),
                        'max_bin': [10, 12, 14, 16, 18, 20], #st.randint(10, 20),
                        'learning_rate': [0.01, 0.1, 1], #st.uniform(0.001, 1),
                        'n_estimators': [80, 90, 100, 110, 120, 130, 140, 150], #st.randint(80, 150), 
                        'boosting_type': ['gbdt', 'dart'], #, 'rf'
                        'objective': ['regression'],
                        'feature_fraction': [0.7],
                        'bagging_fraction': [0.6, 0.7, 0.8],
                        "num_threads": [6],
                        "force_col_wise": [True],
                        "verbose": [-1]}
            
            lgbm = LGBMRegressor()
            #grid_ = GridSearchCV(lgbm, params, scoring = 'neg_mean_squared_error', cv = 5)
            grid_ = RandomizedSearchCV(lgbm, params, scoring = 'roc_auc', cv = 3)
            hyper = Pipeline(steps = [('preprocessor', preprocessor), ('classifier', grid_)])
            hyper.fit(X_train, Y_train)

            # Retorna los mejores parametros utilizados para el proceso de entrenamiento
            print("BEST_SCORE_GOT: ", grid_.best_score_)
            print("\n The best parameters:\n", grid_.best_params_)
            best_params = grid_.best_params_
            
        # Busqueda de hyperparametros mediante modelo CatBoost
        elif type_model == "CatBoost":
            print("\n >> Model: ", type_model)
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
                    "thread_count": [6], #10
                    "logging_level": ["Silent"]}
            
            cat = CatBoostRegressor()
            #rsf = RepeatedStratifiedKFold(random_state = 6)
            grid_ = RandomizedSearchCV(cat, params, scoring = 'roc_auc', cv = 3)
            hyper = Pipeline(steps = [('preprocessor', preprocessor), ('regressor', grid_)])
            hyper.fit(X_train, Y_train)

            # Retorna los mejores parametros utilizados para el proceso de entrenamiento
            print("BEST_SCORE_GOT (CatBoost): ", grid_.best_score_)
            print("\n The best parameters (CatBoost):\n", grid_.best_params_)
            #best_params = params
            best_params = grid_.best_params_

        return best_params, preprocessor

    # Modulo: Modelo (Light Gradient-Boosting Machine - LGBM) para la prediccion de forecast
    def get_model_LGBM(self, data, columns_num, columns_cat, col_pred):
        column = columns_cat + columns_num
        data = data.dropna().reset_index(drop = True)
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
            
        X_test["sale_real"] = Y_test
        X_test["sale_pred"] = pred
        X_test.loc[X_test['sale_pred'] < 0 , 'sale_pred'] = 0
        X_test.sale_pred = X_test.sale_pred.round()
        X_test.sale_pred = X_test.sale_pred.astype(int)

        # Metrica para evaluar la presicion del modelo vs real en base a formula de negocio
        X_test["Accuracy"] = self.forecast_accuracy(X_test.sale_real, X_test.sale_pred)
        X_test.loc[X_test.Accuracy.isnull(), "Accuracy"] = 0

        print("---"*30)
        print(" > Accuracy (r2_score): ", round(coef_det, 2))
        print(" > Accuracy: ", round(X_test[(X_test.Accuracy > 0) & (X_test.sale_real > 0)].Accuracy.mean(), 2))

        X_train["sales"] = Y_train
        #graph_pred(X_train, X_test)

        return train_model, X_test
    
    def get_models_statsForecast(self, data):
        models = [CrostonClassic(), ADIDA(), IMAPA(), TSB(0.2, 0.2), CrostonSBA(), AutoARIMA()]
        #models = [CrostonClassic()]

        #season_length = 24 # Hourly data 
        #horizon = len(X_test) # number of predictions
        #data_std = pd.DataFrame(self.scale.fit_transform(data[["y"]]), columns = self.scale.get_feature_names_out())
        #data_std["ds"] = data.ds
        #data_std["unique_id"] = 1
        #print(data_std)

        sf = StatsForecast(df = data, models = models, freq = 'D', n_jobs = -1)
        sf.fit()
        forecasts = sf.forecast(h=10)
        pred = sf.predict(h=10)
        cv_df = sf.cross_validation(df = data, h = 7, step_size = 7, n_windows = 1)
        print(cv_df.head(20))
        print(cv_df.shape)
        print("---"*30)
        print(forecasts)
        print("---"*30)
        print(pred)
        
        #rmse = rmse(cv_df['y'], cv_df['CrostonClassic'])
        #print(rmse)
        mae = mean_absolute_error(cv_df['y'], cv_df['CrostonClassic'])
        print(mae)

    # Modulo:
    def forecasting(self, data, var_time, var_obs, period):
        col_drop = data.columns.tolist()[:-3]
        data.drop(col_drop, axis = 1, inplace = True)

        segments = data.segment.unique().tolist()
        for seg in segments:
            print(seg)
            temp = data[data.segment == seg]
            temp.drop("segment", axis = 1, inplace = True)
            temp.rename(columns = {var_time: "ds", var_obs: "y"}, inplace = True)
            temp["unique_id"] = 1
            temp["ds"] = pd.to_datetime(temp["ds"])
            #temp = temp[['unique_id', 'ds', 'y']]
            print(temp.head())
            print("---"*30)
            
            self.get_models_statsForecast(temp)
            break
            
            #sf = StatsForecast(df = temp, models = models, freq = "D", n_jobs = -1)
            