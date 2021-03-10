#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import pickle
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# In[4]:


# the custom scaler class

class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns,copy=True, with_mean=True, with_std=True):
        self.scaler = StandardScaler(copy,with_mean,with_std)
        self.columns = columns
        self.mean_= None
        self.var_ = None
        
    def fit(self,X,y=None):
        self.scaler.fit(X[self.columns],y)
        self.mean_ =np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self
    
    def transform(self, X,y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        return pd.concat([X_not_scaled,X_scaled],axis=1)[init_col_order]


# In[ ]:


# create the special class that we are going to use from here on to predict new data
class Framingham_CHD_model_model():
      
        def __init__(self, model_file, scaler_file):
            # read the 'model' and 'scaler' files which were saved
            with open('Framingham_CHD_model','rb') as model_file, open('scaler', 'rb') as scaler_file:
                self.reg = pickle.load(model_file)
                self.scaler = pickle.load(scaler_file)
                self.data = None
        
        # take a data file (*.csv) and preprocess it in the same way as in the lectures
        def load_and_clean_data(self, data_file):
            
            # import the data
            df = pd.read_csv(data_file,delimiter=',')
            # store the data in a new variable for later use
            self.df_with_predictions = df.copy()
            
            #Dropping missing education data
            edu_missing = df[df['education'].isnull()].index
            df = df.drop(edu_missing)
            
            ## cigsPerDay can be both, so I will check if the same index of the missing data current smoker or not,
            ## if not null data will fill with 0, or median
            cigarette_index = df[df['cigsPerDay'].isnull()].index
            smokers = df[df['currentSmoker'] == 1].index
            
            ##I will create a cigarettes array using smokers indeces. So, I will get the median only from smokers (almost half of the participants are non smokers, reduces the mean( Median turns 0 without checking only smokers)
            cigarettes_by_smokers = []
            for i in smokers:
                 if df['cigsPerDay'][i] != 'nan':
                    cigarettes_by_smokers.append(df['cigsPerDay'][i])
            
            ## Finding the median cigarettes per day based on smokers
            smoker_median = statistics.median(cigarettes_by_smokers)
            
            ## All of the missing values in cigsPerDay actually current smokers so, i will replace missing values with mean
            df['cigsPerDay'] = df['cigsPerDay'].fillna(smoker_median)
            
            ## BPMed missing values: I made some research on Google, so if your blood pressure is higher than 140-90 
            ## Doctors are recommending to take BPMed. So, I will check if sysBP is higher than 140 and/or diaBP is higher 
            ## than 90, if so I will switch NaN values to 1 or 0
            BP_missing_index = df[df['BPMeds'].isnull()].index
            
            for i in BP_missing_index:
                if ( df['sysBP'][i] > 140 or df['diaBP'][i] > 90 ):
                    df.loc[i,'BPMeds'] = 1.0  
            else:
                df.loc[i,'BPMeds'] = 0.0
                
            
            ## I will going fill rest of the NaN value with mean values
            df['totChol'] = df['totChol'].fillna(round(df['totChol'].mean()))
            df['BMI'] = df['BMI'].fillna(df['BMI'].mean())
            df['glucose'] = df['glucose'].fillna(round(df['glucose'].mean()))
            
            ## There is only one missing value in heart rate, I will use bfill method for replacing NA value
            ## will bfill it replaces the value that comes directly after it in the same column
            df['heartRate'] = df['heartRate'].fillna(method='bfill', axis=0)
            
            ## I will re-group them 0: Less than High School and High School degrees, 1: College Degree and Higher
            df["education"] = df["education"].map({1.0:0, 2.0:0, 3.0:1, 4.0:1})

            # we have included this line of code if you want to call the 'preprocessed data'
            self.preprocessed_data = df.copy()
            
            # we need this line so we can use it in the next functions
            self.data = self.scaler.transform(df)
    
        # a function which outputs the probability of a data point to be 1
        def predicted_probability(self):
            if (self.data is not None):  
                pred = self.reg.predict_proba(self.data)[:,1]
                return pred
        
        # a function which outputs 0 or 1 based on our model
        def predicted_output_category(self):
            if (self.data is not None):
                pred_outputs = self.reg.predict(self.data)
                return pred_outputs
        
        # predict the outputs and the probabilities and 
        # add columns with these values at the end of the new data
        def predicted_outputs(self):
            if (self.data is not None):
                self.preprocessed_data['Probability'] = self.reg.predict_proba(self.data)[:,1]
                self.preprocessed_data ['Prediction'] = self.reg.predict(self.data)
                return self.preprocessed_data




