# -*- coding: utf-8 -*-
"""

Mehmet Furkan Celik (Corresponding author)
celikmeh@itu.edu.tr, mfurkancelik89@gmail.com,

Date: Aug, 2023 

Please cite this paper:
M. F. Celik, M. S. Isik, G. Taskin, E. Erten and G. Camps-Valls, "Explainable Artificial Intelligence for Cotton Yield Prediction With Multisource Data," 
in IEEE Geoscience and Remote Sensing Letters, vol. 20, pp. 1-5, 2023, Art no. 8500905, doi: 10.1109/LGRS.2023.3303643.

"""




import math
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import warnings
warnings.simplefilter('ignore')
from interpret.glassbox import ExplainableBoostingRegressor
#%%
cotton = pd.read_csv("conus_cotton_yield.csv",index_col='index')

cotton_Y = cotton['yield']
cotton_X = cotton.drop(['yield','state','county','year'],axis=1)
#%% OPTUNA CV FOR EBM
RANDOM_SEED = 42

X_train, X_test, y_train, y_test = train_test_split(cotton_X,cotton_Y, test_size=0.20, random_state=RANDOM_SEED)

kfolds = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

def tune(objective):
    study = optuna.create_study(study_name="ebm_cotton_yield",
                               storage="sqlite:///ebm_cotton_yield.db",
                                direction="maximize")
    # If interrupted, load database
    #study = optuna.load_study(study_name="ebm_cotton_yield",
    #                            storage="sqlite:///ebm_cotton_yield.db")
    study.optimize(objective, n_trials=100)

    params = study.best_params
    best_score = study.best_value
    print(f"Best score: {best_score}\n")
    print(f"Optimized parameters: {params}\n")
    return params
##################
# EBM
##################
def ebm_hypers(trial):
    param={'interactions' : trial.suggest_int("interactions", 1, 10, step=1),
       'learning_rate': trial.suggest_float('learning_rate', 0.00001, 1, log=True),
       'max_leaves' : trial.suggest_int("max_leaves", 2, 5),
       'min_samples_leaf': trial.suggest_int("min_samples_leaf", 3, 500),
       'validation_size': trial.suggest_float("validation_size", 0.2, 1.0, log=True),
       'max_rounds': trial.suggest_int('max_rounds', 10, 10000),
       #'max_bins': trial.suggest_int('max_bins', 32, 1024),
       #'max_interaction_bins': trial.suggest_int('max_interaction_bins', 4, 64),
       #'binning': trial.suggest_categorical('binning', ['uniform', 'quantile']),
       #'outer_bags': trial.suggest_int('outer_bags', 2, 32),
       'early_stopping_rounds': 20,
       'early_stopping_tolerance': 0.001
       }
    ebm = ExplainableBoostingRegressor(**param, random_state=RANDOM_SEED, n_jobs=-1)
   
    scores = cross_val_score(
        ebm, cotton_X, cotton_Y, cv=kfolds,
        scoring="r2"
    )
    return scores.mean()

ebm_params = tune(ebm_hypers)
ebm = ExplainableBoostingRegressor(**ebm_params, random_state=RANDOM_SEED)

ebm_best = ebm.fit(X_train, y_train)

preds_ebm = ebm_best.predict(X_test)
##################
# Accuracy assessment
##################
mae_ebm = mean_absolute_error(y_test, preds_ebm)
rmse_ebm = math.sqrt(mean_squared_error(y_test, preds_ebm))
mape_ebm = mean_absolute_percentage_error(y_test, preds_ebm)
r_square_ebm = r2_score(y_test, preds_ebm)

print("Mean Absolute Error :",mae_ebm)
print("Root Mean Aquare Error : ", rmse_ebm)
print("Mean Absolute Percentage Error :",mape_ebm)
print("R^2 :",r_square_ebm)