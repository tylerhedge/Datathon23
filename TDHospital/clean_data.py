import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

def create_df (df,training):

    #cleaning sex data
    df.head()
    df["sex"]=df['sex'].str[0].str.lower()
    df["sex"]=df['sex'].str.replace('1', 'm')

    #impution
    columns_to_inputate = ['meals', 'temperature', 'blood', 'timeknown', 'cost', 'reflex', 'bloodchem1', 'bloodchem2', 'heart'] #imputation columns here
    for col in columns_to_inputate:
        df[col].fillna(df[col].mean(), inplace=True)

    #cleaning remaining NaN values
    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)

    #remove outliers
    def handle_outliers(df, columns, training):
        for column in columns:
            print(f'column: {column}')
            q75, q25 = np.percentile(df[column], [75, 25])
            intr_qr = q75 - q25
            upper_bound = q75 + (1.5 * intr_qr)
            lower_bound = q25 - (1.5 * intr_qr)
    
            df.loc[(df[column] < lower_bound) | (df[column] > upper_bound), column] = np.nan #replace values outside of bounds with NaN
        if training:
            df.dropna(subset=columns, inplace=True) #drop rows with Nan

    columns_to_handle = ['meals', 'temperature', 'blood', 'timeknown', 'cost', 'reflex', 'bloodchem1', 'bloodchem2', 'heart'] #remove outliers here
    #handle_outliers(df, columns_to_handle, training)
        

    # #pca-ing
    # columns_to_PCA = ['meals', 'temperature', 'blood', 'timeknown', 'cost', 'reflex', 'bloodchem1', 'bloodchem2', 'heart', 'psych1', 'glucose', 'psych2', 'psych3', 'bp', 'bloodchem3', 'confidence', 'bloodchem4', 'comorbidity', 'totalcost', 'breathing', 'age', 'sleep', 'bloodchem5', 'pain', 'urine', 'bloodchem6', 'education', 'psych5', 'psych6', 'information']
    # pca = PCA(n_components=len(columns_to_PCA))
    # pcaDF = df[columns_to_PCA]
    # print('AAAAA', len(columns_to_PCA), pcaDF.shape)
    # pcaDF = standardize(pcaDF)
    # pca.fit(pcaDF)
    # print('Explained variance ratios:')
    # for i, ratio in enumerate(pca.explained_variance_ratio_):
    #     print(f'{columns_to_PCA[i]} | {ratio}')
        
    #  plt.bar(columns_to_PCA, pca.explained_variance_ratio_, edgecolor='black', alpha=0.7, color='blue')
    # plt.xlabel('principle component')
    # plt.ylabel('explained variance ratio')
    # plt.title(f"explained variance by principle component")
    # plt.show()

    #one hot encoding
    to_one_hot_encode = ['sex', 'race','dnr']
    #df = pd.get_dummies(df, columns=to_one_hot_encode, dtype=int) 
    
    df['sex_f'] = (df['sex'] == 'f').astype(int)
    df['sex_m'] = (df['sex'] == 'm').astype(int)
    df = df.drop(columns=["sex"])

    df['cancer_m'] = (df['cancer'] == 'metastatic').astype(int)
    df['cancer_y'] = (df['cancer'] == 'yes').astype(int)
    df = df.drop(columns=['cancer'])

    # df['dnr_after'] = (df['dnr'] == 'dnr after sadm').astype(int)
    # df['dnr_before'] = (df['dnr'] == 'dnr before sadm').astype(int)
    # df['no_dnr'] = (df['dnr'] == 'no dnr').astype(int)

    # df = df.drop(columns=["dnr"])



    print(df.head())


    return df

