import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv(r"F:\M.tech\2nd sem\data\carfuel.csv")

data=df[['Engine size','Cylinders','Combined','CO2 emissions','Smog rating']]
#correlation between features
c=data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(c,cmap="BrBG",annot=True)
print(c)
plt.show()

#finding outliers
detail = dict(zip(df['Model'], df['CO2 emissions']))
q1=np.percentile(df['CO2 emissions'],25)
q3=np.percentile(df['CO2 emissions'],75)
iqr=q3-q1
upper_fence = q3+1.5*iqr
lower_fense = q1-1.5*iqr
print("\n\noutliers=")
for i in range (len(df['CO2 emissions'])):
    if df['CO2 emissions'][i]<lower_fense:
        print(f"Model: {df['Model'][i]}, CO2 emissions: {df['CO2 emissions'][i]}")
    elif df['CO2 emissions'][i]>upper_fence:
        print(f"Model: {df['Model'][i]}, CO2 emissions: {df['CO2 emissions'][i]}")
    else:
        continue
sns.boxplot(x=df['CO2 emissions'])
plt.show()
