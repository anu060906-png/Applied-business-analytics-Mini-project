# BUSINESS ANALYTICS PROJECT
# Supermarket Sales Dataset - Complete Python Code
# Dataset: Supermarket Sales by Aung Pyae (Kaggle)
# Save the downloaded CSV as: supermarket_sales.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_1samp, chi2_contingency, f_oneway
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.formula.api import ols
import statsmodels.api as sm

# 1. LOAD DATA
df = pd.read_csv("supermarket_sales.csv")
print(df.shape)
print(df.head())
print(df.tail())
print(df.info())
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# 2. PREPROCESSING
df["Date"] = pd.to_datetime(df["Date"])
df["Time"] = pd.to_datetime(df["Time"], format="%H:%M").dt.time
df["Month"] = df["Date"].dt.month_name()
df["Month_Num"] = df["Date"].dt.month
df["Day_Name"] = df["Date"].dt.day_name()
df["Hour"] = pd.to_datetime(df["Time"].astype(str)).dt.hour

numeric_cols = ["Unit price","Quantity","Tax 5%","Total","cogs",
                "gross margin percentage","gross income","Rating"]
for c in numeric_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.drop_duplicates()
print(df.describe(include="all"))

# 3. EDA - REQUIRED CHARTS
plt.figure(figsize=(8,5))
df.groupby("Branch")["Total"].sum().plot(kind="bar")
plt.title("Total Sales by Branch"); plt.ylabel("Sales"); plt.tight_layout(); plt.show()

plt.figure(figsize=(10,5))
df.groupby("Product line")["Total"].sum().sort_values().plot(kind="barh")
plt.title("Total Sales by Product Line"); plt.xlabel("Sales"); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
df["Total"].hist(bins=20)
plt.title("Histogram of Transaction Total"); plt.xlabel("Total"); plt.ylabel("Frequency"); plt.show()

plt.figure(figsize=(8,5))
df["Quantity"].hist(bins=10)
plt.title("Histogram of Quantity"); plt.xlabel("Quantity"); plt.ylabel("Frequency"); plt.show()

plt.figure(figsize=(8,4))
plt.boxplot(df["Total"], vert=False)
plt.title("Box Plot of Transaction Total"); plt.show()

df["Payment"].value_counts().plot(kind="pie", autopct="%1.1f%%")
plt.title("Payment Method Share"); plt.ylabel(""); plt.show()

plt.figure(figsize=(8,5))
plt.scatter(df["Quantity"], df["Total"], alpha=.5)
plt.xlabel("Quantity"); plt.ylabel("Total"); plt.title("Quantity vs Total")
plt.show()

# 4. DESCRIPTIVE STATISTICS
for col in ["Unit price","Quantity","Total","gross income","Rating"]:
    print("\n", col)
    print("Mean:", df[col].mean())
    print("Median:", df[col].median())
    print("Mode:", df[col].mode().iloc[0])
    print("Min:", df[col].min())
    print("Max:", df[col].max())
    print("Range:", df[col].max()-df[col].min())
    print("Variance:", df[col].var())
    print("Std:", df[col].std())
    print("Skewness:", df[col].skew())

# 5. CORRELATION
corr_cols = ["Unit price","Quantity","Tax 5%","Total","cogs","gross income","Rating"]
corr = df[corr_cols].corr()
print(corr)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap"); plt.show()

# 6. REGRESSION: QUANTITY -> TOTAL
X = df[["Quantity"]]
y = df["Total"]
model = LinearRegression()
model.fit(X,y)
pred = model.predict(X)
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])
print("R2:", model.score(X,y))
print("RMSE:", mean_squared_error(y,pred,squared=False))

plt.scatter(X,y,alpha=.4)
plt.plot(X, pred)
plt.xlabel("Quantity"); plt.ylabel("Total")
plt.title("Linear Regression: Quantity vs Total")
plt.show()

# 7. PROBABILITY - DICE
np.random.seed(42)
rolls = np.random.randint(1,7,1000)
print("First 20 rolls:", rolls[:20])
for i in range(1,7):
    print(i, np.mean(rolls==i))

# Normal, Uniform, Exponential
normal_data=np.random.normal(0,1,1000)
uniform_data=np.random.uniform(0,1,1000)
exponential_data=np.random.exponential(1,1000)
fig,ax=plt.subplots(1,3,figsize=(15,4))
ax[0].hist(normal_data); ax[0].set_title("Normal")
ax[1].hist(uniform_data); ax[1].set_title("Uniform")
ax[2].hist(exponential_data); ax[2].set_title("Exponential")
plt.tight_layout(); plt.show()

# 8. DECISION MAKING UNDER UNCERTAINTY
# Illustrative promotion decision
alternatives = {
    "No Promotion": [(0.6, 12000), (0.4, 8000)],
    "Moderate Promotion": [(0.6, 16000), (0.4, 9000)],
    "Aggressive Promotion": [(0.6, 19000), (0.4, 5000)]
}
for a, outcomes in alternatives.items():
    ev=sum(p*payoff for p,payoff in outcomes)
    print(a, "Expected Value =", ev)

# 9. SAMPLING DISTRIBUTION
population=df["Total"].dropna()
for n in [30,50,100]:
    means=[population.sample(n,replace=False).mean() for _ in range(200)]
    print(n, np.mean(means), np.std(means,ddof=1), np.std(population,ddof=1)/np.sqrt(n))
    plt.hist(means,bins=20)
    plt.title(f"Sampling Distribution of Mean (n={n})")
    plt.show()

# 10. CONFIDENCE INTERVALS
x=df["Total"].dropna()
n=len(x); m=x.mean(); s=x.std(ddof=1)
for conf in [0.90,0.95,0.99]:
    alpha=1-conf
    tcrit=stats.t.ppf(1-alpha/2,n-1)
    margin=tcrit*s/np.sqrt(n)
    print(conf, (m-margin,m+margin))

# 11. BOOTSTRAP
np.random.seed(42)
boot_medians=[np.median(np.random.choice(x,size=len(x),replace=True)) for _ in range(2000)]
print("Bootstrap 95% CI:", np.percentile(boot_medians,[2.5,97.5]))
plt.hist(boot_medians,bins=30); plt.title("Bootstrap Distribution of Median"); plt.show()

# 12. HYPOTHESIS TEST
# H0: mean transaction total = 500
# H1: mean transaction total != 500
stat,p=ttest_1samp(x,500)
print("t statistic:",stat,"p-value:",p)

# 13. A/B TESTING (illustrative/simulated because no A/B field exists)
np.random.seed(42)
A=np.random.binomial(1,0.10,500)
B=np.random.binomial(1,0.13,500)
rate_A=A.mean(); rate_B=B.mean()
table=np.array([[A.sum(),len(A)-A.sum()],[B.sum(),len(B)-B.sum()]])
chi2,p,_,_=chi2_contingency(table)
print("A rate:",rate_A,"B rate:",rate_B,"p:",p)
plt.bar(["Version A","Version B"],[rate_A,rate_B])
plt.ylabel("Conversion Rate"); plt.show()

# 14. TIME SERIES
daily=df.groupby("Date")["Total"].sum().sort_index()
daily.plot(figsize=(12,4),title="Daily Sales")
plt.show()

train=daily.iloc[:-14]
test=daily.iloc[-14:]

holt=ExponentialSmoothing(train,trend="add",seasonal=None).fit()
holt_fc=holt.forecast(len(test))

arima=ARIMA(train,order=(1,1,1)).fit()
arima_fc=arima.forecast(len(test))

def accuracy(actual,pred):
    mae=mean_absolute_error(actual,pred)
    mse=mean_squared_error(actual,pred)
    rmse=np.sqrt(mse)
    return mae,mse,rmse

print("Holt:", accuracy(test,holt_fc))
print("ARIMA:", accuracy(test,arima_fc))

# 15. OPTIMIZATION
# Example LP using scipy.optimize.linprog
from scipy.optimize import linprog
# maximize 40x + 30y
# labor: 2x+y <= 100
# machine: x+2y <= 80
c=[-40,-30]
A_ub=[[2,1],[1,2]]
b_ub=[100,80]
res=linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=[(0,None),(0,None)],method="highs")
print(res.x, -res.fun)

# 16. SIMULATION
np.random.seed(42)
scenarios={"Conservative":0.90,"Base":1.00,"Aggressive":1.10}
base=daily.mean()
results={}
for name,mult in scenarios.items():
    demand=np.random.normal(base*mult,base*0.15,1000)
    results[name]=[demand.mean(),demand.std(),np.maximum(demand,0).sum()]
print(pd.DataFrame(results,index=["Mean Demand","Std Dev","Total Demand"]).T)

# 17. ONE-WAY ANOVA
groups=[g["Total"].values for _,g in df.groupby("Product line")]
F,p=f_oneway(*groups)
print("ANOVA F:",F,"p:",p)

# 18. EXPERIMENTAL DESIGN / FACTORIAL
# Illustrative 2x2 factorial experiment
np.random.seed(42)
rows=[]
for ad in ["Social Media","Email"]:
    for disc in [5,15]:
        for rep in range(20):
            sales=300 + (35 if ad=="Social Media" else 15) + (20 if disc==15 else 0)
            interaction=12 if (ad=="Social Media" and disc==15) else 0
            rows.append([ad,disc,sales+interaction+np.random.normal(0,20)])
exp=pd.DataFrame(rows,columns=["Advertising","Discount","Sales"])
fit=ols("Sales ~ C(Advertising)*C(Discount)",data=exp).fit()
print(sm.stats.anova_lm(fit,typ=2))
sns.pointplot(data=exp,x="Discount",y="Sales",hue="Advertising")
plt.title("Factorial Experiment"); plt.show()

# 19. BUSINESS INSIGHTS
print("Total sales:",df["Total"].sum())
print("Top branch:",df.groupby("Branch")["Total"].sum().idxmax())
print("Top product line:",df.groupby("Product line")["Total"].sum().idxmax())
print("Top payment method:",df["Payment"].value_counts().idxmax())
print("Top month:",df.groupby(df["Date"].dt.month)["Total"].sum().idxmax())

# 20. SAVE CLEAN DATA
df.to_csv("supermarket_sales_clean.csv",index=False)
