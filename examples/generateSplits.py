from sklearn.model_selection import KFold

def getKF(n_folds=5):
    kf_outer = KFold(n_folds, shuffle=True, random_state=42)
    return kf_outer


### I run it using:
# for train_index, valid_index in kf_outer.split(X):
