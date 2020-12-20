def remove_outliers(arr,threshold=[-50,50]):
    arr1 = arr[arr>threshold[0] & arr<threshold[1]]
    return arr1

def remove_outliers_df(df, outlier_threshold=1000):
    df[df > outlier_threshold] = 0
    df[df < -outlier_threshold] = 0
    # print(df)
    return df