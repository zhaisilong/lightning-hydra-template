import pandas as pd

def metric_spearman_footrule(y, y_pred, topn, seq=[]):
    if not seq:
        df = pd.DataFrame(zip(y, y_pred), columns=['y', 'y_pred'])
    else:
        df = pd.DataFrame(zip(y, y_pred, seq), columns=['y', 'y_pred', 'seq'])
    df['y_rank'] = df['y'].rank(method='first', ascending=False)
    df['y_pred_rank'] = df['y_pred'].rank(method='first', ascending=False)

    data = sorted(df[['y_rank', 'y_pred_rank']].values.tolist())
    r = []
    for i in range(min(topn, len(y))):
        n = 0
        for j in range(i + 1):
            if data[j][1] <= i + 1:
                n += 1
        r.append(n)
    index = [i+1 for i in range(min(topn, len(y)))]
    return list(zip(index, r))