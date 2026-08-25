from re import X
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('ml_dataset.csv')
data = data[data['Surprise_Pct'] > 0].reset_index(drop=True)
x = data[['Surprise_Pct']]
y = data['Profitable_20d']

split_index = int((len(data)*0.7))
X_train = x.iloc[:split_index]
X_test = x.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))



pro = model.predict_proba(X_test)
pro = pro[:,1]
p = roc_auc_score(y_test,pro)


car_test = data['CAR_20d'].iloc[split_index:]
Evaluation = pd.DataFrame({
    'prob': pro,
    'CAR_20d': car_test.values
})
high_pro = np.percentile(pro,75)
car_mean = car_test.mean()
high_trades = Evaluation[Evaluation['prob'] >= high_pro]
high_car = high_trades['CAR_20d'].mean()
edge = high_car-car_mean
from scipy import stats
low_trades = Evaluation[Evaluation['prob'] < high_pro]
t_stat, p_val = stats.ttest_ind(
    high_trades['CAR_20d'], 
    low_trades['CAR_20d'], 
    equal_var=False
)
print(t_stat, p_val)