#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Converted from Jupyter Notebook: notebook.ipynb
Conversion Date: 2025-11-06T16:12:48.951Z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('C:/Users/DELL/Documents/machine_learning/project/train (1).csv')

df.head()

df.tail()

df.shape

df.columns

df.dtypes

df.isna().sum()

df.corr(numeric_only=True)

sns.heatmap(df.corr(numeric_only=True))

df.dtypes

df['MSZoning'].value_counts()

sns.countplot(x=df['MSZoning'],data=df,color='g')

df['Street'].value_counts()

sns.countplot(x=df['Street'],data=df,color='y')

df['Alley'].value_counts()

sns.countplot(x=df['Alley'],data=df,color='k')

df.isnull().sum()[df.isnull().sum() > 0]

df['LotFrontage'].unique()

df['LotFrontage']=df['LotFrontage'].fillna(df['LotFrontage'].mean())

df['Alley'].unique()

df['Alley']=df['Alley'].fillna(df['Alley'].mode()[0])

df['MasVnrType'].unique()

df['MasVnrType']=df['MasVnrType'].fillna(df['MasVnrType'].mode()[0])

df['MasVnrArea'].unique()

df['MasVnrArea']=df['MasVnrArea'].fillna(df['MasVnrArea'].mean())

df['BsmtQual'].unique()

df['BsmtQual']=df['BsmtQual'].fillna(df['BsmtQual'].mode()[0])

df['BsmtCond'].unique()

df['BsmtCond']=df['BsmtCond'].fillna(df['BsmtCond'].mode()[0])

df['BsmtExposure'].unique()

df['BsmtExposure']=df['BsmtExposure'].fillna(df['BsmtExposure'].mode()[0])

df['BsmtFinType1'].unique()

df['BsmtFinType1']=df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode()[0])

df['BsmtFinType2'].unique()

df['BsmtFinType2']=df['BsmtFinType2'].fillna(df['BsmtFinType2'].mode()[0])

df['Electrical'].unique()

df['Electrical']=df['Electrical'].fillna(df['Electrical'].mode()[0])

df['FireplaceQu'].unique()

df['FireplaceQu']=df['FireplaceQu'].fillna(df['FireplaceQu'].mode()[0])

df['GarageType'].unique()

df['GarageType']=df['GarageType'].fillna(df['GarageType'].mode()[0])

df['GarageYrBlt'].unique()

df['GarageYrBlt']=df['GarageYrBlt'].fillna(df['GarageYrBlt'].mean())

df['GarageFinish'].unique()

df['GarageFinish']=df['GarageFinish'].fillna(df['GarageFinish'].mode()[0])

df['GarageCond'].unique()

df['GarageCond']=df['GarageCond'].fillna(df['GarageCond'].mode()[0])

df['GarageQual'].unique()

df['GarageQual']=df['GarageQual'].fillna(df['GarageQual'].mode()[0])

df['PoolQC'].unique()

df['PoolQC']=df['PoolQC'].fillna(df['PoolQC'].mode()[0])

df['Fence'].unique()

df['Fence']=df['Fence'].fillna(df['Fence'].mode()[0])

df['MiscFeature'].unique()

df['MiscFeature']=df['MiscFeature'].fillna(df['MiscFeature'].mode())[0]

df.select_dtypes(include='object').columns

from sklearn.preprocessing import LabelEncoder

# make a copy of your data
df_encoded = df.copy()

# create the encoder
le = LabelEncoder()

# apply to all object columns
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
df_encoded

dft=pd.read_csv('C:/Users/DELL/Documents/machine_learning/project/test (1).csv')

dft.head()

dft.tail()

dft.dtypes

dft.shape

dft.isna().sum()[dft.isna().sum()>0]

dft['MSZoning'].unique()

dft['MSZoning']=dft['MSZoning'].fillna(dft['MSZoning'].mode()[0])

dft['LotFrontage'].unique()

dft['LotFrontage']=dft['LotFrontage'].fillna(dft['LotFrontage'].mean())

dft['Utilities'].unique()

dft['Utilities']=dft['Utilities'].fillna(dft['Utilities'].mode()[0])

dft['Exterior1st'].unique()

dft['Exterior1st']=dft['Exterior1st'].fillna(dft['Exterior1st'].mode()[0])

dft['Exterior2nd'].unique()

dft['Exterior2nd']=dft['Exterior2nd'].fillna(dft['Exterior2nd'].mode()[0])

dft['MasVnrType'].unique()

dft['MasVnrType']=dft['MasVnrType'].fillna(dft['MasVnrType'].mode()[0])

dft['MasVnrArea'].unique()

dft['MasVnrArea']=dft['MasVnrArea'].fillna(dft['MasVnrArea'].mean())

dft['BsmtQual'].unique()

dft['BsmtQual']=dft['BsmtQual'].fillna(dft['BsmtQual'].mode()[0])

dft['BsmtCond'].unique()

dft['BsmtCond']=dft['BsmtCond'].fillna(dft['BsmtCond'].mode()[0])

dft['BsmtExposure'].unique()

dft['BsmtExposure']=dft['BsmtExposure'].fillna(dft['BsmtExposure'].mode()[0])

dft['BsmtFinType1'].unique()

dft['BsmtFinType1']=dft['BsmtFinType1'].fillna(dft['BsmtFinType1'].mode()[0])

dft['MSZoning'].unique()

dft['MSZoning']=dft['MSZoning'].fillna(dft['MSZoning'].mode()[0])

dft['LotFrontage'].unique()

dft['LotFrontage']=dft['LotFrontage'].fillna(dft['LotFrontage'].mean())

dft['MSZoning'].unique()

dft['MSZoning']=dft['MSZoning'].fillna(dft['MSZoning'].mode()[0])

dft['LotFrontage'].unique()

dft['LotFrontage']=dft['LotFrontage'].fillna(dft['LotFrontage'].mean())

dft['BsmtFinSF1'].unique()

dft['BsmtFinSF1']=dft['BsmtFinSF1'].fillna(dft['BsmtFinSF1'].mean())

dft['BsmtFinType2'].unique()

dft['BsmtFinType2']=dft['BsmtFinType2'].fillna(dft['BsmtFinType2'].mode()[0])

dft['BsmtFinSF2'].unique()

dft['BsmtFinSF2']=dft['BsmtFinSF2'].fillna(dft['BsmtFinSF2'].mean())

dft['BsmtUnfSF'].unique()

dft['BsmtUnfSF']=dft['BsmtUnfSF'].fillna(dft['BsmtUnfSF'].mean())

dft['TotalBsmtSF'].unique()

dft['TotalBsmtSF']=dft['TotalBsmtSF'].fillna(dft['TotalBsmtSF'].mean())

dft['BsmtFullBath'].unique()

dft['BsmtFullBath']=dft['BsmtFullBath'].fillna(dft['BsmtFullBath'].mode()[0])

dft['BsmtHalfBath'].unique()

dft['BsmtHalfBath']=dft['BsmtHalfBath'].fillna(dft['BsmtHalfBath'].mode()[0])

dft['KitchenQual'].unique()

dft['KitchenQual']=dft['KitchenQual'].fillna(dft['KitchenQual'].mode()[0])

dft['Functional'].unique()

dft['Functional']=dft['Functional'].fillna(dft['Functional'].mode()[0])

dft['FireplaceQu'].unique()

dft['FireplaceQu']=dft['FireplaceQu'].fillna(dft['FireplaceQu'].mode()[0])

dft['GarageType'].unique()

dft['GarageType']=dft['GarageType'].fillna(dft['GarageType'].mode()[0])

dft['GarageYrBlt'].unique()

dft['GarageYrBlt']=dft['GarageYrBlt'].fillna(dft['GarageYrBlt'].mean())

dft['GarageFinish'].unique()

dft['GarageFinish']=dft['GarageFinish'].fillna(dft['GarageFinish'].mode()[0])

dft['GarageCars'].unique()

dft['GarageCars']=dft['GarageCars'].fillna(dft['GarageCars'].mode()[0])

dft['GarageArea'].unique()

dft['GarageArea']=dft['GarageArea'].fillna(dft['GarageArea'].mean())

dft['GarageQual'].unique()

dft['GarageQual']=dft['GarageQual'].fillna(dft['GarageQual'].mode()[0])

dft['GarageCond'].unique()

dft['GarageCond']=dft['GarageCond'].fillna(dft['GarageCond'].mode()[0])

dft['SaleType'].unique()

dft['SaleType']=dft['SaleType'].fillna(dft['SaleType'].mode()[0])

from sklearn.preprocessing import LabelEncoder

# make a copy of your data
dft_encoded = dft.copy()

# create the encoder
le = LabelEncoder()

# apply to all object columns
for col in dft_encoded.select_dtypes(include='object').columns:
    dft_encoded[col] = le.fit_transform(dft_encoded[col].astype(str))
dft_encoded

dft_encoded.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

df_encoded.drop(['Alley','PoolQC','Fence','MiscFeature'],axis=1,inplace=True)

dft.isna().sum()[dft.isna().sum()>0]

x=df_encoded.iloc[:,:-1]
y=df_encoded.iloc[:,-1]

z=dft_encoded

from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(x,y)
y_pred=model.predict(z)
y_pred

y