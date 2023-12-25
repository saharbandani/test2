import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("/kaggle/input/car-acceptability-classification-dataset/car.csv")
df

df.isnull().sum()

df.duplicated().sum()

sns.heatmap(df.isnull(), cmap = 'viridis')

df.info()

plt.figure(figsize = (20, 10))

for i in range(7):
    plt.subplot(2 , 4, i+1)
    sns.countplot(data = df, x = df.iloc[:, i])
    plt.grid()

    le = LabelEncoder()

    df['Buying_Price'].value_counts(), df['Maintenance_Price'].value_counts(), df['No_of_Doors'].value_counts(), df['Person_Capacity'].value_counts(), df['Size_of_Luggage'].value_counts(), df['Safety'].value_counts(), df['Car_Acceptability'].value_counts()
    
    df['Buying_Price'] = le.fit_transform(df['Buying_Price'])
    df['Maintenance_Price'] = le.fit_transform(df['Maintenance_Price'])
    df['No_of_Doors'] = le.fit_transform(df['No_of_Doors'])
    df['Person_Capacity'] = le.fit_transform(df['Person_Capacity'])
    df['Size_of_Luggage'] = le.fit_transform(df['Size_of_Luggage'])
    df['Safety'] = le.fit_transform(df['Safety'])
    df['Car_Acceptability'] = le.fit_transform(df['Car_Acceptability'])

    df['Buying_Price'].value_counts(), df['Maintenance_Price'].value_counts(), df['No_of_Doors'].value_counts(), df['Person_Capacity'].value_counts(), df['Size_of_Luggage'].value_counts(), df['Safety'].value_counts(), df['Car_Acceptability'].value_counts()

    df.describe().T

    df_corr = df.corr()
    df_corr

    sns.heatmap(df_corr, cmap = 'viridis', annot=True)

    #Decision Tree
    model_dt = DecisionTreeClassifier()
    model_dt.fit(x_train, y_train)
    y_pred_dt = model_dt.predict(x_test)
    perform(y_pred_dt)

    feature_names = df.columns[0: 6]
    target_names = df['Car_Acceptability'].unique().tolist()
    from sklearn.tree import plot_tree # tree diagram

    plt.figure(figsize=(25, 20))
    plot_tree(model_dt, feature_names = feature_names, class_names = ['Acceptable', 'Good', 'Un-acceptable', 'Very Good'], filled = True, rounded = False)

    plt.savefig('tree_visualization.png')

    viz_model = dtreeviz.model(model_dt,
                X_train=x_train, y_train=y_train,
                feature_names=feature_names,
                target_name='Car Acceptability',
                class_names=['Acceptable', 'Good', 'Un-acceptable', 'Very Good'])

v = viz_model.view()  
v.save("car acceptability dt.svg")  # save as svg
viz_model.view()

