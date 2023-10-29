import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

from clean_data import create_df

#import for random forest model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#import for logistic regression model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib


def data_preprocessing(df):

    df.head(1)
    
    col_to_keep = ['death', 
                   'meals', 
                   'temperature', 
                   'blood', 
                   'timeknown', 
                   'cost', 
                   'reflex', 
                   'bloodchem1', 
                   'bloodchem2', 
                   'heart', 
                   'psych1', 
                #    'glucose', 
                #    'psych2', 
                #    'bp', 
                #    'bloodchem3', 
                #    'confidence', 
                #    'bloodchem4', 
                #    'comorbidity', 
                #    'totalcost', 
                #    'breathing', 
                #    'age', 
                #    'sleep', 
                #    'bloodchem5', 
                #    'pain', 
                #    'urine', 
                #    'bloodchem6', 
                #    'education', 
                #    'psych5', 
                #    'psych6', 
                #    'information',
                   'diabetes',
                  # 'race_asian', 'race_black', 'race_hispanic', 'race_other', 'race_white'
                  'sex_f', 'sex_m',
                  #'dnr_after','dnr_before','no_dnr'
                  'cancer_m', 'cancer_y'
                  ]
    df = df[col_to_keep]

    df.replace('', 0, inplace=True)
    df.fillna(0, inplace=True)
    return df
    
def split_feature_label(df):
    y = df['death']
    X = df.drop(columns=['death'])
    return y, X
    # print(X)
    # print(y)

    # death_0 = y.tolist().count(0)
    # death_1 = y.tolist().count(1)
    # percent_death_0 = 100 * death_0 / (death_0 + death_1)
    # percent_death_1 = 100 * death_1 / (death_0 + death_1)
    # print(f'Survived: {death_0}, or {percent_death_0:.2f}%')
    # print(f'Died: {death_1}, or {percent_death_1:.2f}%')

def standardize(X):
    scaler = StandardScaler()
    X_numeric = scaler.fit_transform(X.select_dtypes(include=['float64']))
    X[X.select_dtypes(include=['float64']).columns] = X_numeric
    return X

def train_model(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)

    # Define the neural network model
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),  # Input layer
        layers.Dense(32, activation='relu'),     # Hidden layer with 128 neurons and ReLU activation
        layers.Dense(16, activation='relu'),      # Another hidden layer with 64 neurons and ReLU activation
        layers.Dense(1, activation='sigmoid')     # Output layer with sigmoid activation for binary classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    
    model.save('example.h5')
    
    print(f'Test accuracy: {test_accuracy}')

    # Optionally, you can plot training history to visualize model performance
    import matplotlib.pyplot as plt

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    return model


def train_model_two(X, y):
    # Split data into training and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=.3, random_state=42)

    # Create a Random Forest classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the Random Forest model on the training data
    rf_model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = rf_model.predict(X_test)

    test_accuracy = accuracy_score(y_test, y_pred)

    joblib.dump(rf_model, "./random_forest.joblib")
    print(f'Test accuracy 2: {test_accuracy}')

    return rf_model

def meta_model(X, y, model, rf_model):
    #Combine the predictions into a new dataset
    combined_predictions = np.column_stack((model.predict(X), rf_model.predict(X)))

    #Split the combined predictions and obtain the true labels for the test set
    X_combined_train, X_combined_test, y_train, y_test = train_test_split(combined_predictions, y, test_size=0.2, random_state=42)

    #Train a logistic regression meta-model on the training data
    meta_model = LogisticRegression()
    print(type(X_combined_train))
    meta_model.fit(X_combined_train, y_train)

    #Step 3: Make predictions using the meta-model
    meta_predictions = meta_model.predict(X_combined_test)

    meta_accuracy = accuracy_score(y_test, meta_predictions)
    joblib.dump(meta_model, "./meta_model.joblib")
    print("Meta-Model Accuracy:", meta_accuracy)
    

if __name__ == "__main__":
    df = pd.read_csv('TDHospital/TD_HOSPITAL_TRAIN.csv')
    df = create_df(df, True)
    cleaned_data = data_preprocessing(df)
    y, X = split_feature_label(cleaned_data)
    X = standardize(X)
    meta_model(X, y, train_model(X, y), train_model_two(X,y))
    