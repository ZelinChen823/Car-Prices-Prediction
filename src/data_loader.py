import pandas as pd
import missingno as msno


def load_data(filepath: str) -> pd.DataFrame:
    data = pd.read_csv(filepath)
    print("Data shape:", data.shape)
    print("Head of data:\n", data.head())
    return data


def initial_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    data.loc[:, 'Number of Doors'] = data['Number of Doors'].fillna(4.0)

    data = data[data['highway MPG'] < 350]
    data = data[data['highway MPG'] < 60]
    data['Present Year'] = 2025
    data['Years Of Manufacture'] = data['Present Year'] - data['Year']
    data.drop(['Present Year'], axis=1, inplace=True)

    data.loc[:, 'Engine HP'] = data['Engine HP'].fillna(data['Engine HP'].median())

    data.loc[:, 'Engine Cylinders'] = data['Engine Cylinders'].fillna(4)

    if 'Market Category' in data.columns:
        data.drop(['Market Category'], axis=1, inplace=True)

    return data


def show_missing_matrix(data: pd.DataFrame):
    msno.matrix(data, color=(0.5, 0.5, 0.5))
