import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r"F:\M.tech\2nd sem\data\carfuel.csv")
print(df.head(10))
print(df.info())
print(df.describe())

#unique vehicle classes
unique_vehicle_classes = df['Vehicle class'].unique()
print(unique_vehicle_classes)

#no of vehicle in each vehicle class
diff_classes= df.Vehicle_class.value_counts()
print(diff_classes)

#comparison between make and vehicle class\n",
class_make=df.groupby('Vehicle_class')['Make'].value_counts()
print(class_make)

#how does average fuel comsumption vary with different vehicle classes\n",
avg_fuelcons_by_enginesize=df.groupby('Vehicle_class')['Combined'].mean()
print("Avg fuel consumption by vehicle class")
print(avg_fuelcons_by_enginesize)

#mean CO2 emission for each fuel type\n",
mean_c02em_by_fueltype=df.groupby('Fuel type')['CO2 emissions'].mean()
print("Average CO2 emission by each fuel type")
print(mean_c02em_by_fueltype)

#grouping cylinders and finding city average for each class\n",
grouping_averaging=df.groupby('Cylinders')['City'].mean()
print(grouping_averaging)

#adding a new column for low, moderate and high \n",
def combined_classification(x):
    if x < 9:
        return "Low"
    elif 9<x<14:
        return "Moderate"
    else:
        return "High"
df['Combined_Classification']=df['Combined'].apply(combined_classification)
print(df.head(15))

#cars by their CO2 emissions
top10=df.sort_values(by= 'CO2 emissions')
dat=top10[['Make','Model','CO2 emissions']]
print("\t\tLeast CO2 emission\n")
print(dat.head(5))
print("\n\t\tMost CO2 emission\n")
print(dat.tail(5))