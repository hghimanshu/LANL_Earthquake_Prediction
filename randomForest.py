import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from tqdm import tqdm 
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.metrics import mean_absolute_error as mae
import pickle

data = pd.read_csv('train.csv', dtype={'acoustic_data': np.int16, 'time_to_failure': np.float32})
print(data.head())

rows = 150_000
segments = int(np.floor(data.shape[0] / rows))

X = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['ave', 'std', 'max', 'min'])
Y = pd.DataFrame(index=range(segments), dtype=np.float64,
                       columns=['time_to_failure'])

for segment in tqdm(range(segments)):
    seg = data.iloc[segment*rows:segment*rows+rows]
    x = seg['acoustic_data'].values
    y = seg['time_to_failure'].values[-1]
    
    Y.loc[segment, 'time_to_failure'] = y
    X.loc[segment, 'ave'] = x.mean()
    X.loc[segment, 'std'] = x.std()
    X.loc[segment, 'max'] = x.max()
    X.loc[segment, 'min'] = x.min()


print('New reduced Training Data is || '+ str(X.shape[0]))
Nsplits = 5
Y = np.ravel(Y)
kf = KFold(n_splits=Nsplits, shuffle=True, random_state=10)
y_test_all = []
y_pred_all = []
randomF = rfr(n_estimators=500, criterion='mae', max_depth=1000, random_state=10)
mae_all = []
for train_in, test_in in kf.split(X):
    X_train, X_valid = X.iloc[train_in], X.iloc[test_in]
    y_train, y_valid = Y.iloc[train_in], Y.iloc[test_in]
    y_test_all.append(y_valid)
    randomF.fit(X_train, y_train)
    y_pred = randomF.predict(X_valid)
    y_pred_all.append(y_pred)
    score = mae(y_valid, y_pred)
    mae_all.append(score)
    print('Average MAE :: ' + str(sum(mae_all)/float(len(mae_all))))

# Saving our model
filename = 'models/finalized_model.weights'
pickle.dump(randomF, open(filename, 'wb'))
