from data_loader import load_data, initial_preprocessing
from eda import (countplot_feature, groupby_plot, scatter_plot, boxplot_feature,
                 lmplot_features, heatmap_corr, barplot_value_counts)
from preprocessing import shuffle_and_split, encode_features, scale_data
from models import run_all_models
from visualization import plot_regression_comparison, plot_error_bars
import matplotlib.pyplot as plt
from data_loader import show_missing_matrix

def main():
    data = load_data('../Data/data.csv')
    data = initial_preprocessing(data)
    show_missing_matrix(data)

    countplot_feature(data, 'Make', orient='h', figsize=(20,15), title="Car companies with their cars", palette='Set2')
    groupby_plot(data, 'Year', 'MSRP', agg_func='mean', kind='bar', title="The Average Price of cars in different years", color='g')
    scatter_plot(data, 'highway MPG', 'city mpg', color='salmon', title="Scatterplot between highway MPG and city mpg")
    boxplot_feature(data, 'highway MPG', color='skyblue', title="Boxplot of highway MPG")
    lmplot_features(data, 'Engine HP', 'Popularity', title="Engine HP vs Popularity", color='teal')
    numeric_columns = ['Engine HP', 'Engine Cylinders', 'Number of Doors', 'highway MPG', 'city mpg', 'Popularity']
    heatmap_corr(data, numeric_columns, cmap='BuPu')
    barplot_value_counts(data, 'Years Of Manufacture', title="Total number of cars with particular years of manufacture")

    X_train, X_test, y_train, y_test = shuffle_and_split(data, target_column='MSRP', test_size=0.2, random_state=100)
    X_train, X_test = encode_features(X_train, X_test, y_train)
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    summary_df, results_dict = run_all_models(X_train_scaled, y_train, X_test_scaled, y_test)

    if "Linear Regression" in results_dict:
        plot_regression_comparison(results_dict["Linear Regression"],
                                   title="Linear Regression: Actual vs Predicted",
                                   color='teal')

    plot_error_bars(summary_df, error_type='Mean Absolute Error',
                    title="Barplot of Machine Learning Models with Mean Absolute Error", palette='Paired')

    summary_df.to_csv('model_evaluation_summary.csv', index=False)

if __name__ == '__main__':
    main()
