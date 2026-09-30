import pandas as pd
import numpy as np
from scipy.spatial import distance

if __name__ == '__main__':
    data_df=pd.read_csv('./date_in/Teritorial_2022.csv')
    print(data_df.head()) #primele 5 randuri din tabel
    print(data_df.info()) #toate informatiile

    numeric_data_df=data_df[data_df.columns[4:]]
    print(numeric_data_df.head())
    print(numeric_data_df.info())

    print("Null values before:", data_df.isna().sum().sum())
    print("Is null:", data_df.isna().sum().unique())

    numeric_data_df=numeric_data_df.fillna(numeric_data_df.mean())
    print("Null values after:", data_df.isna().sum().sum())
    print("Is null:", numeric_data_df.isna().sum().unique())

    d_euclid=distance.cdist(numeric_data_df,numeric_data_df.values,metric='euclidean')
    d_euclid_df=pd.DataFrame(d_euclid)
    print(d_euclid)

    print(d_euclid_df) #afisare frumos -gen matrice

