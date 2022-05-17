import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def selection(x, y):
    del x['blueWins']
    # Copy feature matrix and standardise
    data = x
    data_std = (data - data.mean()) / data.std()
    data = pd.concat([y, data_std.iloc[:, 0:9]], axis=1)
    data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Create violin plot of features

    sns.violinplot(x='Features', y='Values', hue='blueWins', data=data, split=True,
                   inner='quart', ax=ax[0], palette='Blues')
    fig.autofmt_xdate(rotation=45)

    data = x
    data_std = (data - data.mean()) / data.std()
    data = pd.concat([y, data_std.iloc[:, 9:18]], axis=1)
    data = pd.melt(data, id_vars='blueWins', var_name='Features', value_name='Values')

    # Create violin plot
    # plt.figure(figsize=(8,5))
    sns.violinplot(x='Features', y='Values', hue='blueWins',
                   data=data, split=True, inner='quart', ax=ax[1], palette='Blues')
    fig.autofmt_xdate(rotation=45)
    plt.show()
    drop_cols = ['blueWardsPlaced', 'blueWardsDestroyed', 'redWardsPlaced', 'redWardsDestroyed']
    x.drop(drop_cols, axis=1, inplace=True)

    # Tower
    x['towerDiff'] = x['blueTowersDestroyed'] - x['redTowersDestroyed']

    data = pd.concat([y, x], axis=1)

    towerGroup = data.groupby(['towerDiff'])['blueWins']
    print(towerGroup.count())
    print(towerGroup.mean())
    drop_cols = ['blueTowersDestroyed', 'redTowersDestroyed', 'towerDiff']
    x.drop(drop_cols, axis=1, inplace=True)

    # Kill and Assist
    x['killsDiff'] = x['blueKills'] - x['blueDeaths']
    x['assistsDiff'] = x['blueAssists'] - x['redAssists']

    x[['killsDiff', 'assistsDiff']].hist(figsize=(12, 10), bins=20)
    plt.show()
    drop_cols = ['blueFirstBlood', 'blueKills', 'blueDeaths', 'blueAssists', 'redAssists']
    x.drop(drop_cols, axis=1, inplace=True)

    # Dragon, herald and elite
    x['dragonsDiff'] = x['blueDragons'] - x['redDragons']
    x['heraldsDiff'] = x['blueHeralds'] - x['redHeralds']
    x['eliteDiff'] = x['blueEliteMonsters'] - x['redEliteMonsters']

    data = pd.concat([y, x], axis=1)

    eliteGroup = data.groupby(['eliteDiff'])['blueWins'].mean()
    dragonGroup = data.groupby(['dragonsDiff'])['blueWins'].mean()
    heraldGroup = data.groupby(['heraldsDiff'])['blueWins'].mean()

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))

    eliteGroup.plot(kind='bar', ax=ax[0])
    dragonGroup.plot(kind='bar', ax=ax[1])
    heraldGroup.plot(kind='bar', ax=ax[2])

    print(eliteGroup)
    print(dragonGroup)
    print(heraldGroup)

    plt.show()
    drop_cols = ['blueEliteMonsters', 'blueDragons', 'blueHeralds',
                 'redEliteMonsters', 'redDragons', 'redHeralds']
    x.drop(drop_cols, axis=1, inplace=True)

    # Drop total gold&Exp
    drop_cols = ['blueTotalGold', 'blueTotalExperience', 'redTotalGold', 'redTotalExperience']
    x.drop(drop_cols, axis=1, inplace=True)

    x.rename(columns={'blueGoldDiff': 'goldDiff', 'blueExperienceDiff': 'expDiff'}, inplace=True)

    return x
