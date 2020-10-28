import streamlit as st
import pandas as pd
from predictor import change_dtypes, get_features, predict, get_interp, get_importance
import datetime
st.set_option('deprecation.showPyplotGlobalUse', False)


st.title('Rossmann Drug Store Daily Sales Predictor')

# taking inputs from user

# date
date = st.date_input(label = 'Enter Date of Interest', value = datetime.date(2015, 8, 1), 
    min_value = datetime.date(2015, 8, 1))

# store id
storeid = st.number_input('Enter Store Id', min_value=1, max_value=1115, value=1)

# promo
promo_inp = st.checkbox('Will you be running a Promo?', value = False)
if promo_inp:
    promo = 1
else:
    promo = 0

# state holiday
state_inp = st.selectbox('Will there be a State Holiday?',
('Public holiday', 'Easter holiday', 'Christmas', 'Working Day'), index = 3)
mapping = {'Public holiday': 'a', 'Easter holiday':'b', 
'Christmas':'c', 'Working Day':'nil'}
state_holiday = mapping[state_inp]

predbt = st.button('Predict')
prediction = None
details = st.checkbox('Show Interpretation Plots', value = False)
if predbt:
        
    # creating df
    user_df = pd.DataFrame({'Date':date, 'Store':storeid, 'Promo': promo, 
    'StateHoliday':state_holiday}, index = [0])

    # merging with store constants
    store = pd.read_csv('Data/store.csv')

    inp = user_df.merge(store, on = 'Store')
    inp = inp.fillna(0)


    inp = change_dtypes(inp)
    inp = get_features(inp)
    
    prediction = predict(inp)
    st.write('The input feature are: \n')
    st.write(inp)
    st.write(f'\nThe predicted Sales for Store', storeid,' on ',date, ' is ', round(prediction, 4), ' €')
    
    try:

        if details:
            st.write('\nGetting Model Interpretations...\n')
            st.title('SHAP VALUE FORCE PLOT')
            print(get_interp(inp))
            st.title('FEATURE IMPORTANCE - LOSS REDUCTION\n')
            print(get_importance())    
    except:
        st.write('Prediction not provided.')

    

