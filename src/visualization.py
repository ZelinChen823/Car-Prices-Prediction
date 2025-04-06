import matplotlib.pyplot as plt
import seaborn as sns

def plot_regression_comparison(results, title: str, color: str):
    plt.figure(figsize=(10, 10))
    sns.regplot(x=results['MSRP'], y=results['Predicted Output'],
                scatter_kws={"s": 40, "alpha": 0.6, "color": color},
                line_kws={"color": color})
    plt.title(title, fontsize=20)
    plt.xlabel("Actual MSRP")
    plt.ylabel("Predicted MSRP")
    plt.show()

def plot_error_bars(summary_df, error_type: str, title: str, palette: str):
    plt.figure(figsize=(20, 10))
    ax = sns.barplot(x='Models', y=error_type, data=summary_df, palette=palette, legend=False)
    for p in ax.patches:
        ax.annotate(format(p.get_height(), '.0f'),
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points')
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(title, fontsize=20)
    plt.show()
