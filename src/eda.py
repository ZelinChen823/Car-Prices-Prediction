import matplotlib.pyplot as plt
import seaborn as sns

def countplot_feature(data, column: str, orient: str = 'v', figsize=(10, 10), title: str = None, palette: str = 'viridis'):
    plt.figure(figsize=figsize)
    if orient == 'h':
        sns.countplot(y=data[column], palette=palette, legend=False)
    else:
        sns.countplot(x=data[column], palette=palette, legend=False)
    plt.title(title if title else f"Countplot of {column}")
    plt.show()

def groupby_plot(data, group_col: str, agg_col: str, agg_func: str = 'mean', kind: str = 'bar', title: str = None, color: str = None):
    plt.figure(figsize=(20, 10))
    grouped = data.groupby(group_col)[agg_col].agg(agg_func)
    grouped.sort_values(ascending=False).plot(kind=kind, color=color)
    plt.title(title if title else f"{agg_func} of {agg_col} grouped by {group_col}", fontsize=20)
    plt.show()

def scatter_plot(data, x: str, y: str, color: str = 'r', title: str = None):
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=x, y=y, data=data, color=color)
    plt.title(title if title else f"Scatterplot between {x} and {y}")
    plt.show()

def boxplot_feature(data, column: str, color: str = None, title: str = None):
    plt.figure(figsize=(10, 10))
    sns.boxplot(x=data[column], color=color)
    plt.title(title if title else f"Boxplot of {column}")
    plt.show()

def lmplot_features(data, x: str, y: str, title: str = None, color: str = None, scatter_kws: dict = None):
    sns.set(rc={'figure.figsize': (10, 10)})
    scatter_args = scatter_kws if scatter_kws else {"s": 40, "alpha": 0.6}
    if color is not None:
        scatter_args["color"] = color
        line_args = {"color": color}
    else:
        line_args = {}
    sns.lmplot(x=x, y=y, data=data, scatter_kws=scatter_args, line_kws=line_args)
    plt.title(title if title else f"lmplot between {x} and {y}", fontsize=15)
    plt.show()

def heatmap_corr(data, numeric_columns: list, cmap: str = 'BuPu'):
    plt.figure(figsize=(15, 15))
    corr_matrix = data[numeric_columns].corr()
    sns.heatmap(corr_matrix, cmap=cmap, annot=True)
    plt.show()

def barplot_value_counts(data, column: str, title: str = None):
    plt.figure(figsize=(10, 10))
    sns.barplot(x=data[column].value_counts().index, y=data[column].value_counts().values, hue=None)
    plt.title(title if title else f"Barplot of {column} value counts", fontsize=20)
    plt.show()
