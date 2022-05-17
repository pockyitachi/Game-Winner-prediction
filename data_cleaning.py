import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

from plot_boxplot import plot_boxplot
from remove_outliers import remove_outliers

RED = '#D10000'
BLUE = '#0082FF'


def remove_repeat(data):
    # first blood
    all(data['blueFirstBlood'] == data['redFirstBlood'].apply(lambda x: 0 if x == 1 else 1))

    # blue kills is red deaths
    all(data['blueKills'] == data['redDeaths'])

    # blue deaths is red kills
    all(data['blueDeaths'] == data['redKills'])

    # blue experience difference is the negative of red experience difference
    all(data['blueExperienceDiff'] == data['redExperienceDiff'].apply(lambda x: -1 * x))

    # blue gold difference is the negative of red gold difference
    all(data['blueGoldDiff'] == data['redGoldDiff'].apply(lambda x: -1 * x))

    y = data.blueWins

    data.drop(['redFirstBlood', 'redDeaths', 'redKills', 'redExperienceDiff', 'redGoldDiff', 'gameId'],
              axis=1, inplace=True)
    return data


def data_visual(data):
    blue_team = [column for column in data.columns if 'blue' in column]
    red_team = [column for column in data.columns if 'red' in column]

    data[blue_team].hist(color=BLUE,
                         figsize=(10, 10))
    plt.tight_layout()
    plt.show()

    data[red_team].hist(color=RED,
                        figsize=(10, 10))

    plt.tight_layout()
    plt.show()


def outliers(data):
    # collect the columns that are "normally distributed".
    normal_columns = ['blueKills', 'blueDeaths', 'blueAssists', 'blueTotalGold', 'blueAvgLevel', 'blueTotalExperience',
                      'blueTotalMinionsKilled', 'blueTotalJungleMinionsKilled',
                      'blueGoldDiff', 'blueExperienceDiff', 'blueCSPerMin', 'blueGoldPerMin',
                      'redAssists', 'redTotalGold', 'redAvgLevel', 'redTotalExperience',
                      'redTotalMinionsKilled', 'redTotalJungleMinionsKilled',
                      'redCSPerMin', 'redGoldPerMin']

    # we need to group the columns with similar range so that the boxplots look interpretable
    # For Blue team
    blue_normal = [column for column in normal_columns if 'blue' in column]
    data[blue_normal].describe().T

    # group the blue columns by relative size
    small_blue = ['blueKills', 'blueDeaths', 'blueAssists', 'blueAvgLevel', 'blueTotalMinionsKilled',
                  'blueTotalJungleMinionsKilled', 'blueCSPerMin']
    large_blue = [column for column in blue_normal if column not in small_blue]

    plot_boxplot(data, small_blue, BLUE, 'Outlier Analysis for Blue Team with Small Values')

    plot_boxplot(data, large_blue, BLUE, 'Outlier Analysis for Blue Team with Large Values')

    data_no_outliers = remove_outliers(data)

    # now we visualize after removing outliers
    plot_boxplot(data_no_outliers, small_blue, BLUE,
                 'Boxplot After Removing Statistical Outliers for Blue Team with Small Values')

    plot_boxplot(data_no_outliers, large_blue, BLUE,
                 'Boxplot After Removing Statistical Outliers for Blue Team with Large Values')

    # For Red Team
    red_normal = [column for column in normal_columns if 'red' in column]
    data[red_normal].describe().T

    small_red = ['redAssists', 'redAvgLevel', 'redTotalMinionsKilled', 'redTotalJungleMinionsKilled',
                 'redCSPerMin']
    large_red = [column for column in red_normal if column not in small_red]

    # visualize
    plot_boxplot(data, small_red, RED, 'Outlier Analysis for Red Team with Small Values')

    plot_boxplot(data, large_red, RED, 'Outlier Analysis for Red Team with Large Values')

    plot_boxplot(data_no_outliers, small_red, RED,
                 'Boxplot After Removing Statistical Outliers for Red Team with Small Values')

    plot_boxplot(data_no_outliers, large_red, RED,
                 'Boxplot After Removing Statistical Outliers for Red Team with Large Values')
    return data_no_outliers


def data_scale(data_no_outliers):
    scaler = StandardScaler()
    data_tr = pd.DataFrame(scaler.fit_transform(data_no_outliers.drop('blueWins', axis=1)),
                           columns=data_no_outliers.drop('blueWins', axis=1).columns)

    # add the bluewins column back in
    data_tr['blueWins'] = data_no_outliers['blueWins']

    # verify that the columns are scaled
    data_tr.describe().T

    return data_tr
