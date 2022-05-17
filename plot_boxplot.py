from matplotlib import pyplot as plt
import seaborn as sns


def plot_boxplot(data, subset, color, title):
    plt.figure(figsize=(11, 9))
    g = sns.boxplot(data=data[subset], color=color)
    g.set_title(title)
    plt.xticks(rotation=30)
    plt.show()
