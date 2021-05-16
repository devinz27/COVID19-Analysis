import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.linear_model import LinearRegression


newcases = pd.read_csv("csv/newCases.csv")
deaths = pd.read_csv("csv/deaths.csv")
hospital = pd.read_csv("csv/activeHospital.csv")
ern = pd.read_csv("csv/ERN.csv")
positive = pd.read_csv("csv/positiveResults.csv")
tests = pd.read_csv("csv/testsDaily.csv")
vaccine = pd.read_csv("csv/vaccine.csv")
cumulativeCases = pd.read_csv("csv/cumulativeCases.csv")

allDf = [newcases, cumulativeCases, deaths, hospital, ern,
         positive, tests, vaccine]
df = reduce(lambda x, y: pd.merge(
    x, y, how="outer", on="category"), allDf)


df.rename(columns={"category": "Date",
          "Active (including people who are hospitalized)": "Active cases", "Hospitalized (including people in ICU)": "Hospitalized", "Positive results": "Positive result (%)", "Total tests completed": "Total tests done", "previous_day_doses_administered": "Previous day doses adminstered", "total_doses_administered": "Total doses adminstered", "total_individuals_fully_vaccinated": "Total fully vaccinated"}, inplace=True)
df.drop(columns=["Critical threshold",
        "total_doses_in_fully_vaccinated_individuals"], inplace=True)

# Dropping outlier date
df.drop(index=462, inplace=True)

# Removes 1 from Total cases
df["Total cases"] = df["Total cases"] - 1

# fill empty values with 0
df = df.fillna(0)

# print(df)

df.to_excel("df.xlsx")

# print(df.describe())

len = int(len(df) * 0.8)
# print(len)
# print(len)

training = df.iloc[:len]
trainingx = training["New cases"].values.reshape(-1, 1)
# print(trainingx)

trainingy = training["New deaths"].values.reshape(-1, 1)
# print(trainingy)


ml = LinearRegression()
ml.fit(trainingx, trainingy)
predy = ml.predict(trainingx)
# print(predy)

plt.xlabel("New cases")
plt.ylabel("New Deaths")
# plt.figure(figsize=(15, 7))
plt.scatter(trainingx, trainingy)
plt.plot(trainingx, predy)
plt.show()


# predicty = ml.predict(validationx)
# print(validationy)
# print(predicty)


# linear regression to predict future values
# powerbi
# create monthly dataset and total everything (ERN cant be summed) ## remove critical threshhold
# ratios

# COVID-19 Analysis
# collected covid19 data and manipulated it using python and pandas
# forcasted __ 30 days in to future using machine learning (linear regression)
# visualized data in interactiver dashboard in powerbi
# created different metrics that gave better insights to data by creating different ratios such as ___
