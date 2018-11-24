import pip #To install packages
#pip.main(["install","pandas"]) #Scientific Computing With Python
#pip.main(["install","numpy"]) #Data Analysis and Data Manipulation tool
#pip.main(["install","matplotlib"]) #2D Plotting library
#pip.main(["install","sklearn"]) #Data Science

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


os.chdir("C:\K2Analytics\Linear_Regression")
inc_exp = pd.read_csv("Inc_Exp_Data.csv")
inc_exp.head()

#lmScatterplot
#import pip
#pip.main(["install","seaborn"])
import seaborn as sns
sns.lmplot(x ="Mthly_HH_Income" ,
           y = "Mthly_HH_Expense", 
           data = inc_exp)

#Another way to fit regression line
#fit = np.polyfit(inc_exp["Mthly_HH_Income"],inc_exp["Mthly_HH_Expense"],1)
#fit_fn = np.poly1d(fit)
#plt.plot(inc_exp["Mthly_HH_Income"],inc_exp["Mthly_HH_Expense"],
 #        'yo',inc_exp["Mthly_HH_Income"] , fit_fn(inc_exp["Mthly_HH_Income"]), '--k')

import statsmodels.formula.api as sm
from statsmodels.api import graphics


#Linear Regression Model
linear_mod = sm.ols(formula ="Mthly_HH_Expense ~ Mthly_HH_Income" , 
                    data = inc_exp).fit()

#Get the coefficient and intercept
linear_mod.params
linear_mod.rsquared

linear_mod.summary()

#Multiple Regression
inc_exp.corr()

m_linear_mod = sm.ols(formula ="Mthly_HH_Expense ~ Mthly_HH_Income+\
                      No_of_Fly_Members+ Emi_or_Rent_Amt+\
                      Annual_HH_Income" ,data = inc_exp).fit()

m_linear_mod.summary()

def VIF(formula,data):
    import pip #To install packages
    #pip.main(["install","dmatrices"])
    #pip.main(["install","statsmodels"])
    from patsy import dmatrices
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    y , X = dmatrices(formula,data = data,return_type="dataframe")
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) \
       for i in range(X.shape[1])]
    vif["features"] = X.columns
    return(vif.round(1))

VIF=VIF("Mthly_HH_Expense ~ Mthly_HH_Income+\
        No_of_Fly_Members+ Emi_or_Rent_Amt+\
        Annual_HH_Income" ,data = inc_exp)
VIF     


#Multiple Linear Regression
m_linear_mod = sm.ols(formula = "Mthly_HH_Expense ~ Mthly_HH_Income+\
                      No_of_Fly_Members+ Emi_or_Rent_Amt",
                      data = inc_exp).fit()

m_linear_mod.params

m_linear_mod.summary()


#prediction
predict = m_linear_mod.predict(inc_exp[["Mthly_HH_Income",
                                        "No_of_Fly_Members",
                                        "Emi_or_Rent_Amt"]])
    
result = pd.DataFrame()
result["Expected"] = inc_exp["Mthly_HH_Expense"] 
result["Observed"] = predict 
result
 
    

