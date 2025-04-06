import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from category_encoders import TargetEncoder, OneHotEncoder as CE_OneHotEncoder

def shuffle_and_split(data: pd.DataFrame, target_column='MSRP', test_size=0.2, random_state=100):
    shuffled = shuffle(data, random_state=random_state)
    X = shuffled.drop([target_column], axis=1)
    y = shuffled[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print("Train set shape:", X_train.shape, y_train.shape)
    print("Test set shape:", X_test.shape, y_test.shape)
    return X_train, X_test, y_train, y_test

def encode_features(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series) -> (pd.DataFrame, pd.DataFrame):
    for col in ['Year', 'Model', 'Make']:
        encoder = TargetEncoder(cols=col)
        encoder.fit(X_train[col], y_train)
        X_train[col] = encoder.transform(X_train[col])
        X_test[col] = encoder.transform(X_test[col])

    cat_cols = ['Engine Fuel Type', 'Transmission Type', 'Driven_Wheels', 'Vehicle Size', 'Vehicle Style']
    one_hot_encoder = CE_OneHotEncoder(cols=cat_cols)
    one_hot_encoder.fit(X_train)
    X_train = one_hot_encoder.transform(X_train)
    X_test = one_hot_encoder.transform(X_test)

    return X_train, X_test

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
