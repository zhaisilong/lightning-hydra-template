from sklearn.utils import shuffle

def show_ratio(df, label, sort=None, n=5) -> None:
    """df 的标签中的各类比值
        Args:
            sort: 'value' or 'label'
    """
    n_all = len(df)
    if sort == 'value':
        n_classes = df[label].value_counts().reset_index().sort_values(by=label, ascending=False)
    elif sort == 'label':
        n_classes = df[label].value_counts().reset_index().sort_values(by='index')
    else:
        n_classes = df[label].value_counts().reset_index()

    n_classes = n_classes[:n]

    for i in n_classes.index:
        print(f'标签 {n_classes.at[i, "index"]} 比例为: {n_classes.at[i, label] / n_all * 100:.2f}%, 个数为: {n_classes.at[i, label]}')

def split_df(df, shuf=True, ):
    """Split df into train/val/test set and write into files
    ratio: 8:1:1
    """
    if shuf:
        df = shuffle(df)
    sep = int(len(df)*0.1)
    test_df = df.iloc[:sep]
    val_df = df.iloc[sep:sep*2]
    train_df = df.iloc[sep*2:]
    return train_df, val_df, test_df
