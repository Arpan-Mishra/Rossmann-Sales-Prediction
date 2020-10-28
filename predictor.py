import pandas as pd
import xgboost 
import numpy as np
import streamlit as st
from streamlit.components.v1 import components
import matplotlib.pyplot as plt
from fastai.tabular.all import add_datepart
from sklearn import metrics
from pathlib import Path
import shap
import pickle


# will be using xgboost
model = pickle.load(open('Model/xgb-full.pkl', 'rb'))
features = pickle.load(open('Model/feature-xgb.pkl', 'rb'))

# datatype function
def change_dtypes(df):
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].astype(np.int64)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].astype(np.int64)

    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].astype(np.int64)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].astype(np.int64)

    return df

# feature engineering 
# feature engg
def get_features(df):

    df = df.copy()
    df = add_datepart(df, 'Date', drop = True).drop('Elapsed', axis = 1)

    # promo interval
    mapping = {1:'Jan', 2:'Feb', 3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    df['MonthInPromoInt']= df[['PromoInterval','Month']].apply(lambda x: 1 if str(mapping[x['Month']]) in str(x['PromoInterval']).split(',') else 0, axis = 1)
        
    
    # competition open since in months
    df[['CompetitionOpenSinceYear','CompetitionOpenSinceMonth']] = df[['CompetitionOpenSinceYear','CompetitionOpenSinceMonth']].astype(np.int64)
    df['compOpenMonths'] = 12*(df.Year - df.CompetitionOpenSinceYear) + (df.Month - df.CompetitionOpenSinceMonth).astype(np.float64)
    df['compOpenMonths'] = df['compOpenMonths'].apply(lambda x: x if x>0 and x<1000 else 0)
    
    
    # promo open since in months
    df[['Promo2SinceWeek','Promo2SinceYear']] = df[['Promo2SinceWeek','Promo2SinceYear']].astype(np.int64)
    df['promoOpenMonths'] = 12*(df.Year - df.Promo2SinceYear) + abs((df.Week - df.Promo2SinceWeek))/4
    df['promoOpenMonths'] = df['promoOpenMonths'].apply(lambda x: x if x>0 and x<1000 else 0)
    
    
    
    # comp distance bin
    def compbin(x):
        if np.isnan(x):
            return 0
        elif x<=3000:
            return 1
        elif x<=9000:
            return 2
        else:
            return 3
    
    df['DistBin'] = df['CompetitionDistance'].apply(lambda x: compbin(x))
    df['PromoInterval'] = df['PromoInterval'].astype('category').cat.codes


    # ordinals
    ordinal = ['StateHoliday', 'StoreType', 'Assortment']
    ord_map = {'0':0,'a':1,'b':2,'c':3,'d':4, 'nil':0}
    for o in ordinal:
        df[o] = df[o].apply(lambda x: ord_map[x])    

    df['Week'] = df['Week'].astype(np.int64)
    df = df[features]

    return df


# prediction
def predict(inp):
    p = model.predict(inp).item()    
    return np.exp(p)

# shap model explainability
def get_interp(inp):    
    explainerModel = shap.TreeExplainer(model)
    shp_values = explainerModel.shap_values(inp)
    print(shp_values)
    fp = shap.force_plot(explainerModel.expected_value, 
    shp_values, inp.iloc[0], matplotlib=True, figsize=(15, 8))
    st.pyplot(fp)
    plt.clf()
    

def get_importance():
    fig,axs = plt.subplots(1,1, figsize = (10,8))
    xgboost.plot_importance(model, ax = axs, importance_type='gain', show_values = False)
    st.pyplot(fig)
