# survival_analysis

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from matplotlib import pyplot as plt
data= pd.read_csv('C:\work\data1.csv')
data.describe()
data.isnull().sum()

T = data['Delay'] #survival time in days
E= data['Event'] # status
plt.figure(figsize=(15,8))
plt.hist(T,bins=50)
plt.show()

![image](https://user-images.githubusercontent.com/66779651/197969626-9a66c779-20bf-444b-b3c3-dddac255c036.png)

kmf = KaplanMeierFitter()
kmf.fit(durations = T, event_observed = E)
kmf.plot_survival_function()
plt.ylabel("probability of survival") **what is the probability that person lives upto 2000 days is 0.5!**

plt.title("survival curve")

plt.show()

![image](https://user-images.githubusercontent.com/66779651/197969746-0e1b3a5a-f67c-4a0c-b04f-e4ae1d2c0cef.png)

kmf.survival_function_.plot() **same plot without the 95% confidence interval** 

plt.title('Survival function')

![image](https://user-images.githubusercontent.com/66779651/197969870-876d5354-0ad7-4444-ad1e-02912ec1949d.png)

plt.figure(figsize=(10,5))
ax = plt.subplot(111)
m = (data["gender"] == 1)
kmf.fit(durations = T[m], event_observed = E[m], label = "Male")
kmf.plot_survival_function(ax = ax)
kmf.fit(T[~m], event_observed = E[~m], label = "Female")
kmf.plot_survival_function(ax = ax, at_risk_counts = True)

plt.title("Survival of different gender group")

![image](https://user-images.githubusercontent.com/66779651/197970001-8c8be40d-876b-49dc-8fd2-6c557affcacd.png)

plt.figure(figsize=(10,5))

ax = plt.subplot(111)

A = (data["race"] == "ASIAN")
kmf.fit(durations = T[A], event_observed = E[A], label = "ASIANS")
kmf.plot_survival_function(ax = ax)

W = (data["race"] == "WHITE")
kmf.fit(T[W], event_observed = E[W], label = "WHITE")
kmf.plot_survival_function(ax = ax)

B = (data["race"] == "BLACK OR AFRICAN AMERICAN")
kmf.fit(T[B], event_observed = E[B], label = "BLACK OR AFRICAN AMERICAN")
kmf.plot_survival_function(ax = ax)

AI = (data["race"] == "AMERICAN INDIAN OR ALASKA Native")
kmf.fit(T[AI], event_observed = E[AI], label = "AMERICAN INDIAN OR ALASKA Native")
kmf.plot_survival_function(ax = ax)

H = (data["race"] == "Native HAWAIIAN OR OTHER PACIFIC ISLANDER")
kmf.fit(T[H], event_observed = E[H], label = "Native HAWAIIAN OR OTHER PACIFIC ISLANDER")
kmf.plot_survival_function(ax = ax)

plt.title("Survival of different Race group")

![image](https://user-images.githubusercontent.com/66779651/197970052-66d36e6a-cba5-43aa-b66e-f3b39bd97e9e.png)

sns.set(style= 'darkgrid')
plt.figure(figsize=(10,8))
his= sns.distplot(data["age_at_initial_pathologic_diagnosis"],color = 'b')
plt.title("age_at_initial_pathologic_diagnosis", fontsize= "14")
plt.ylabel('FREQUENCIES', color= "b", fontsize= "14")
plt.xlabel('AGE', color = "b", fontsize= "14")

![image](https://user-images.githubusercontent.com/66779651/197970161-b194d409-593d-4098-9c5d-22cf679817e9.png)

