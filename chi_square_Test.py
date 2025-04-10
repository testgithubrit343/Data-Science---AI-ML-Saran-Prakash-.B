import numpy as np
from scipy.stats import chi2

observed = np.array([[50,70],[80,100],[60,90],[30,50],[20,50]])
rows_total = observed.sum(axis = 1)
columns_total = observed.sum(axis = 0)
grand_total = observed.sum()

expected = np.outer(rows_total,columns_total)/grand_total

chi_squared_statistic = ((observed- expected) **2/expected).sum()

rows,cols = observed.shape
df = (rows-1)*(cols-1)

alpha = 0.05
critical_value = chi2.ppf(1-alpha,df)

decision = "Reject the null hypothesis" if chi_squared_statistic > critical_value else "Fail to rejected null hypothesis"

print(f"Chi Squared statistic :{chi_squared_statistic}")
print(f"Critical_value : {critical_value}")
print(f"Decision of Freedom : {df}")
print(f"Decision : {decision}")