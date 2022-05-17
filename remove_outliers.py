def remove_outliers(data):
    """
    Remove the outlier if it is 3 standard deviations away from the mean
    """
    for column in data.columns:
        column_mean = data[column].mean()
        column_std = data[column].std()
        threshold = column_std * 3  # three standard deviations
        lower, upper = column_mean - threshold, column_mean + threshold
        data = data[(data[column] >= lower) & (data[column] <= upper)]
    return data
