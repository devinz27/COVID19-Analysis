import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
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

df.to_excel("df.xlsx")

print(f'Final Dataframe \n {df}')

# print(df.describe())

df["Date"] = pd.to_datetime(df["Date"])

monthdf = pd.DataFrame(df)
monthdf["year"] = monthdf["Date"].dt.year
monthdf["month"] = monthdf["Date"].dt.month

monthdf = monthdf.groupby(["year", "month"], as_index=False).sum()
print(f'Monthly Sums \n {monthdf}')

print("Welcome to my Covid-Analysis program. Please enter two variables to be the axis\n")
print("New cases (1), 7-day average (2), Total cases (3), New deaths (4), Active cases (5), Hospitalized (6), Effective reproduction number (7), Positive result (%) (8), Total tests done (9), Previous day doses adminstered (10), Total doses adminstered (11), Total fully vaccinated (12)\n")
x = input("Input for the x axis ")
y = input("Input for the y axis ")

if x.lower() == "new cases" or x == str(1):
    x = "New cases"
elif x.lower() == "7-day average" or x == str(2):
    x = "7-day average"
elif x.lower() == "total cases" or x == str(3):
    x = "Total cases"
elif x.lower() == "new deaths" or x == str(4):
    x = "New deaths"
elif x.lower() == "active cases" or x == str(5):
    x = "Active cases"
elif x.lower() == "hospitalized" or x == str(6):
    x = "Hospitalized"
elif x.lower() == "effective reproduction number" or x == str(7):
    x = "Effective reproduction number"
elif x.lower() == "positive result (%)" or x == str(8):
    x = "Positive result (%)"
elif x.lower() == "total tests done" or x == str(9):
    x = "Total tests done"
elif x.lower() == "previous day doses adminstered" or x == str(10):
    x = "Previous day doses adminstered"
elif x.lower() == " total doses adminstered" or x == str(11):
    x = "Total doses adminstered"
elif x.lower() == "total fully vaccinated" or x == str(12):
    x = "Total fully vaccinated"

if y.lower() == "new cases" or y == str(1):
    y = "New cases"
elif y.lower() == "7-day average" or y == str(2):
    y = "7-day average"
elif y.lower() == "total cases" or y == str(3):
    y = "Total cases"
elif y.lower() == "new deaths" or y == str(4):
    y = "New deaths"
elif y.lower() == "active cases" or y == str(5):
    y = "Active cases"
elif y.lower() == "hospitalized" or y == str(6):
    y = "Hospitalized"
elif y.lower() == "effective reproduction number" or y == str(7):
    y = "Effective reproduction number"
elif y.lower() == "positive result (%)" or y == str(8):
    y = "Positive result (%)"
elif y.lower() == "total tests done" or y == str(9):
    y = "Total tests done"
elif y.lower() == "previous day doses adminstered" or y == str(10):
    y = "Previous day doses adminstered"
elif y.lower() == " total doses adminstered" or y == str(11):
    y = "Total doses adminstered"
elif y.lower() == "total fully vaccinated" or y == str(12):
    y = "Total fully vaccinated"

# x = "New cases"
# y = "Effective reproduction number"
# linear regression between new deaths and active cases
training = df.iloc[:]
trainingx = training[x].values.reshape(-1, 1)
# print(trainingx)

trainingy = training[y].values.reshape(-1, 1)
# print(trainingy)

# training set
ml = LinearRegression()
ml.fit(trainingx, trainingy)
predy = ml.predict(trainingx)

plt.xlabel(x)
plt.ylabel(y)
plt.title(f'Linear Regression graph of {x} and {y}')
# plt.figure(figsize=(15, 7))
plt.scatter(trainingx, trainingy, color="blue")
plt.plot(trainingx, predy, color="red")
plt.show()


compare = pd.DataFrame({"Actual": trainingy.flatten(),
                        "Predicted": predy.flatten()})
print(f'Actual vs Predicted values \n {compare}')
