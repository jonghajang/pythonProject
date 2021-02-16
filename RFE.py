import pandas as pd
import numpy as np
from sklearn.metrics import auc, roc_curve
from catboost import CatBoostClassifier, Pool

def print_progressbar(total, i):
    """
    total : total iteration number.
    i : iteration count, starting from 0.
    """
    import math
    step = 25 / total

    # Print the progress bar
    print('\r' + f'Progress: '
        f"[{'=' * int((i+1) * step) + ' ' * (25 - int((i+1) * step))}]"
        f"({math.floor((i+1) * 100 / (total))} %) ({i+1}/{total})",
        end='')
    if (i+1) == total: print("")


# sample data
train = pd.read_csv("sample_train_df.csv", sep = ',')
test = pd.read_csv("sample_test_df.csv", sep = ',')

# sample label
y_train = pd.read_csv("y_train_df.csv", sep = ',')
y_test = pd.read_csv("y_test_df.csv", sep = ',')

# numeric features split
numeric_features = ["numeric1","numeric2","numeric3","numeric4"]

train_numeric = train[numeric_features]
train_category = train.drop(numeric_features,1)

test_numeric = test[numeric_features]
test_category = test.drop(numeric_features,1)

new_train = pd.concat([train_numeric,train_category],1)
new_test = pd.concat([test_numeric,test_category],1)


def RFE(train, y_train, test, y_test, num_round=10, num_cut=10):
    df = pd.DataFrame(columns=['remain_features', 'cut_features', 'test_AUC'])

    cut_features_list = []
    features = []
    test_auc_list = []

    ## If the numeric features are removed, the order must be changed.
    for i in range(0, num_round):
        train = train.drop(cut_features_list, 1)
        test = test.drop(cut_features_list, 1)
        mlist = list()

        for word in list(train.columns):
            for numer in numeric_features:
                if numer in word:
                    mlist.append(numer)

        # catboost categorical features position

        categorical_features = np.array(list(range(len(mlist), train.shape[1])))
        train_pool = Pool(train, y_train, cat_features=categorical_features)

        ### This model is GPU mode

        cat_model = CatBoostClassifier(iterations=100,
                                       #bootstrap_type='Poisson',
                                       random_strength=10,
                                       logging_level='Silent',
                                       task_type="CPU",
                                       eval_metric="AUC",
                                       cat_features=categorical_features)

        ## Calcurate AUC

        cat_model.fit(train_pool)
        y_test_proba2 = cat_model.predict_proba(test)
        fpr, tpr, thresholds3 = roc_curve(y_test, y_test_proba2[:, 1])
        test_auc = auc(fpr, tpr)

        ## remain features name
        features.append(train.columns)

        ## feature importance caculate
        col = train.columns
        col = pd.DataFrame(cat_model.get_feature_importance()).set_index(col).sort_values(by=0)
        col = col[col > 0].dropna()

        ## cut features name
        cut_features_list = list(col[0:num_cut].index)
        result = pd.DataFrame(list([features, test_auc_list])).T

        ## save dataframe

        df = df.append({
            'remain_features': features,
            'cut_features': cut_features_list,
            'test_AUC': test_auc
        }, ignore_index=True)

        print_progressbar(num_round, i)
        #print(f"  remain {len(result[0][i])} features")
    return df


result = RFE(new_train,y_train,new_test,y_test,num_round=10,num_cut=1)
print(result)

result['remain_features']










