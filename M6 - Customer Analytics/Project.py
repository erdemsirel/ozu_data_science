# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import pandas as pd
pd.set_option("display.max_rows", 500)
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
# -

data = pd.read_excel("Final Project Data.xlsx", skiprows=1).drop("Unnamed: 10", axis=1)

data.head()
# Train Data:2019-1	2019-2	2019-3	2019-4	2019-5	2019-6	2019-7	2019-8	2019-9
# Test Data:2019-10	2019-11	2019-12	2020-1	2020-2


# ### Cohort survival rates over months

survival_rates = data.drop("Customer ID", axis=1).sum()
survival_rates = pd.DataFrame(survival_rates).rename({0:"Number of Custumer"},axis=1)
survival_rates["Survival Rate"] = (survival_rates["Number of Custumer"] / survival_rates.loc["2019-1", "Number of Custumer"])

survival_rates["term"] = survival_rates["Number of Custumer"].rank(ascending=False).astype(int) -1

survival_rates["Survival Rate"].plot(kind="bar")
plt.title("Survival Rates")
plt.show()

survival_rates_all = survival_rates.copy()

survival_rates = survival_rates_all[0:9]

survival_rates[0:9]

X = survival_rates.term.values.reshape(-1, 1)
y = survival_rates["Survival Rate"].values
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
#ax.set_title('axes title')
ax.set_xlabel('Tenure (years)')
ax.set_ylabel('% Surviving')
ax.plot(X+1,y*100)
ax.axvline(x=9.5, color='r', linestyle='--')
ax.axis([0, 13, 0, 110])
ax.set_xticks(np.arange(0, 14, 1.0))
plt.show()

# ### Linear

X = survival_rates.term.values.reshape(-1, 1)
y = survival_rates["Survival Rate"].values
regressor1 = LinearRegression()
regressor1.fit(X,y)
#model:
print("y =", round(regressor1.intercept_,3), "+ (" ,regressor1.coef_[0], ") t" )
print("R2:",round(regressor1.score(X, y),3))

# ### Quadratic

X = np.vstack((survival_rates.term.values, survival_rates.term.values**2)).T
y = survival_rates["Survival Rate"].values

# +
regressor2 = LinearRegression()
regressor2.fit(X,y)
#model:
      
print(f"y = {round(regressor2.intercept_,3)} + ( {round(regressor2.coef_[0],3)} ) t + ( {round(regressor2.coef_[1],3)} ) t2") 
print("R2:",round(regressor2.score(X, y),3))

# -

# ### Exponantial

X = survival_rates.term.values.reshape(-1, 1)
y = np.log(survival_rates["Survival Rate"].values)

regressor3 = LinearRegression()
regressor3.fit(X,y)
#model:
print("ln(y) =", round(regressor3.intercept_,3), "+ (" ,round(regressor3.coef_[0],3), ") t" )
print("R2:",round(regressor3.score(X, y),3))

# ### Model Fit
#

#linear,exp input
X = survival_rates.term.values.reshape(-1, 1)
#quad input
X1 = np.vstack((survival_rates.term.values, survival_rates.term.values**2)).T
y = survival_rates["Survival Rate"].values
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
#ax.set_title('axes title')
ax.set_xlabel('Tenure (years)')
ax.set_ylabel('% Surviving')
ax.axvline(x=9.5, color='r', linestyle='--')
ax.axis([0, 13, 0, 110])
ax.set_xticks(np.arange(0, 14, 1.0))
#actual
ax.plot(X+1,100*y, label="Actual",linestyle='-')
#linear
ax.plot(X+1,100*regressor1.predict(X), label="Linear", linestyle=':')
#quad
ax.plot(X+1,100*regressor2.predict(X1), label="Quadratic", linestyle='-.')
#exp
ax.plot(X+1,100*np.exp(regressor3.predict(X)), label="Exponential", linestyle='--')
ax.legend(loc='best')
plt.show()

# ### Predictions

#linear,exp input
X = survival_rates_all.term.values.reshape(-1, 1)
#quad input
X1 = np.vstack((survival_rates_all.term.values, survival_rates_all.term.values**2)).T
y = survival_rates_all["Survival Rate"].values
fig = plt.figure()
ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)
#ax.set_title('axes title')
ax.set_xlabel('Tenure (years)')
ax.set_ylabel('% Surviving')
ax.axvline(x=9.5, color='r', linestyle='--')
ax.axis([0, 13, 0, 110])
ax.set_xticks(np.arange(0, 14, 1.0))
#actual
ax.plot(X+1,100*y, label="Actual",linestyle='-')
#linear
ax.plot(X+1,100*regressor1.predict(X), label="Linear", linestyle=':')
#quad
ax.plot(X+1,100*regressor2.predict(X1), label="Quadratic", linestyle='-.')
#exp
ax.plot(X+1,100*np.exp(regressor3.predict(X)), label="Exponential", linestyle='--')
ax.legend(loc='best')
plt.show()m

# ### Cohort retention rates over months
#

retention_rate = data.drop("Customer ID", axis=1).sum()
retention_rate = pd.DataFrame(retention_rate).rename({0:"Number of Custumer"},axis=1)
retention_rate["Retention Rate"] = (retention_rate["Number of Custumer"] / retention_rate["Number of Custumer"].shift(1))

retention_rate

retention_rate["Retention Rate"].plot()
plt.title("Retention Rate")
plt.show()

# ## Prediction of total number of transactions per month

monthly = data.drop("Customer ID", axis=1).sum()
mothly_train = monthly.iloc[0:9]
mothly_test = monthly.iloc[9:]

mothly_train_retention = pd.DataFrame(mothly_train).rename({0:"Number of Custumer"},axis=1)
mothly_train_retention["Retention Rate"] = (mothly_train_retention["Number of Custumer"] / mothly_train_retention["Number of Custumer"].shift(1))

last_three_month_avg_retention = mothly_train_retention[-3:]["Retention Rate"].mean()

pred = pd.concat([pd.DataFrame(mothly_train, columns=["train"]), 
           pd.DataFrame(mothly_test, columns=["test"]) ])
pred = pred.fillna(0)

for row in range(len(pred)):
    if pred.iloc[row, 0] == 0:
        pred.iloc[row, 0] = pred.iloc[row -1, 0] * last_three_month_avg_retention

pred = pred[9:]

metrics.mean_squared_error(pred.test, pred.train)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / y_true)


mean_absolute_percentage_error(pred.test, pred.train)

# **Benchmark MSE: 1233, MAPE:0.036**

# ## Prediction of cumulative transactions (over months)

pred.sum()

pred.sum().train - pred.sum().test

np.power((pred.sum().train - pred.sum().test),2)

mean_absolute_percentage_error(pred.sum().test, pred.sum().train)



# ## Regression Model

train = data.drop("Customer ID",axis=1).iloc[:,0:9]


data_melt = data.melt( id_vars=["Customer ID"], 
          var_name="Month",
          value_name='t',)
data_melt["Month"] = pd.to_datetime(data_melt["Month"])
data_melt = data_melt.sort_values(["Customer ID","Month"], ascending=[True, True])

data_melt["t-1"] = data_melt.groupby("Customer ID").shift(1).t
data_melt["t-2"] = data_melt.groupby("Customer ID").shift(2).t
data_melt["t-3"] = data_melt.groupby("Customer ID").shift(3).t

data_melt = data_melt.dropna(subset=["t", "t-1", "t-2", "t-3"])

train_set = data_melt[data_melt.Month <= "2019-09-01"]
test_set = data_melt[data_melt.Month == "2019-10-01"]
train_set = train_set.set_index(["Customer ID", "Month"])
test_set = test_set.set_index(["Customer ID", "Month"])

x_train, y_train = train_set.drop(["t"], axis=1), train_set.t
x_test, y_test = test_set.drop(["t"], axis=1), test_set.t

model = LogisticRegression()

model.fit(x_train,y_train)

y_train_pred = pd.DataFrame(model.predict_proba(x_train))[1].apply(lambda x: 1 if x >= 0.25 else 0)

y_train_pred.sum()

train_set = train_set.reset_index()
train_set["pred"] = y_train_pred

train_set

metrics.accuracy_score(train_set.t, train_set.pred)

metrics.confusion_matrix(train_set.t, train_set.pred)
train_set.t.sum(), train_set.pred.sum()

y_test_pred = pd.DataFrame(model.predict_proba(x_test))[1].apply(lambda x: 1 if x >= 0.25 else 0)

test_set = test_set.reset_index()
test_set["pred"] = y_test_pred

metrics.accuracy_score(test_set.t, test_set.pred)

metrics.confusion_matrix(test_set.t, test_set.pred)

test_set.t.sum(), test_set.pred.sum()


