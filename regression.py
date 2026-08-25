import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv("ml_dataset.csv")
data = data[data['Surprise_Pct'].abs() <= 200].reset_index(drop=True)

x = data[["Surprise_Pct"]]
y= data["CAR_20d"]


split_index = int((len(data)*0.7))
X_train = x.iloc[:split_index]
X_test = x.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

import numpy as np
from scipy import stats

Evaluation = pd.DataFrame({'pred_CAR': y_pred, 'actual_CAR': y_test.values})
cutoff = np.percentile(y_pred, 75)
high = Evaluation[Evaluation['pred_CAR'] >= cutoff]
low = Evaluation[Evaluation['pred_CAR'] < cutoff]

print(high['actual_CAR'].mean() - low['actual_CAR'].mean())
t_stat, p_val = stats.ttest_ind(high['actual_CAR'], low['actual_CAR'])
print(t_stat, p_val)

plt.plot(X_train, model.predict(X_train), color = 'r')
plt.scatter(x,y,color = 'b',alpha=0.4, s=15)
plt.xlabel('x')
plt.ylabel('y')
plt.show()