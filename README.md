# Rossmann Sales Prediction: Project Overview
<b>Predicting the daily sales for Rossmann Drug Store</b> <br>
- Created a tool to predict the daily sales of any store of the Rossmann Drug Store Chain which is the 2nd largest drug store chain in Germany. The data was taken from Kaggle.<br>
- Created features referring the active promos run by the store, competitor distance, competitor time of opening, demarking the start of year/quarter etc.
- Tried 3 main modelling approaches, Random Forest, XGBoost & Neural Networks with Entity Embeddings. Selected XGBoost as the final model.
- Created a webapp using Streamlit and deployed on AWS EC2 Instance.<br>
Check out the webapp [here](https://cutt.ly/rossmann-app)

## Table of Contents
* [Technologies Used](#technologies-used)
* [Data](#data)
* [Pre-Processing](#pre-processing)
* [EDA](#eda)
* [Metric Used](#metric)
* [Modelling](#modelling)
* [Performance](#performance)
* [Productionisation](#productionisation)


## Technologies Used
- Python 3.6
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit Learn, XGBoost
- FastAi, Pytorch
- Streamlit
- AWS

## Data
 - The data has been taken from [Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data).
 - It contains 3 csv files, the train, test and store.
 - Train and Test contains information for the day for which the sales need to be predicted and store containes some features for the stores.
 - After mergtin the train and store dataframes, we get the following features <br>
  ![fts](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/fts.png)

## Pre-Processing
- Only the days when the store was opened have been used to train the model, hence rows where Sales == 0 (or Open != 1) have been dropped.
- All the nan values in categorical variables have been imputed with 0.
- For continuous variable as well 0 imputation has been done.

## EDA
- I have explored the distributions of the variables to identify important features and transformations. For the full EDA check out `EDA - RossMann Sales.ipynb`<br>
![1](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/download.png) 
![2](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/store.png) ![3](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/is_.png)

## Metric
The metric used is the root mean squared percentage error<br>
![m](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/rmspe.png)

## Modelling
- Cross Validation Scheme:  The model has been trained on data from 2013-1-1 to 2015-6-31 and then evalutaed on the next 6 weeks. The datapoints have been selected such that
all the strores are included in the trainning and validation sets. For the final model submission model was trained on data from 2013-1-1 to 2015-7-31 and tested on the next 42 days worth of data on Kaggle.<br>
- I have tried 3 models: <br>
1. Random Forest - Majority of the features being categorical, random forests were a good starting point to get a strong baseline and eliminate unimportant features using feature importances.
2. XGBooost - Since random forest gave decent performance I tried XGBoost as boosting techniques usually give a little better performance than bagging models<br>
3. Neural Networks - I used entity embeddings to encode categorical variables, this reduces the dimensionality which is usually increased by one hot encoding and is also able to capture the relationships between categorical datapoints.<br>

## Performance
- The model gets an rmspe of `0.11213` on the kaggle testing set. Here are the results on the validation set:<br>
![res](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/res.png)
![pred](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/pred5.png)
- We can check the feature importance according to XGBoost
![fi](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/fixgb.png)
- I have also plotted the force plot which is calculated using shap values. For a prediction it tells us the positive or negative "force" provided by the features to push the base value to the the predicted value. The base value is nothing but the expected value of the prediction or the mean of the training data.<br>
![fo](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/forcexgb.png)

## Productionisation
- Used streamlit to create a webapp.<br>
![a](https://github.com/Arpan-Mishra/Rossmann-Sales-Prediction/blob/main/figs/app.png)
- It takes in the following inputs from the user:<br>
1. Date - The data for which sales need to be predicted<br>
2. Store Id - The store for which we need the predictions <br>
3. Promo - If the store will be running a promo on that dat<br>
4. State Holiday - If the day is actually a state holiday or not <br>

- The rest of the features used are related to the store id, a database in the form of a csv contains all the information regarding every Rossmann store. <br>
- The webapp also generates interpretation plots (GAIN + SHAP Force Plot), as shown above.<br>
- Finally, the app has been deployed using AWS EC2 Instance with an elastic IP. Check it out: (https://cutt.ly/rossmann-app)

