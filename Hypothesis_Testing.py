import scipy.stats as stats
import math

sample_mean = 3050
theoretical_mean = 4000
sigma = 5*25
n = 25

standard_error = sigma/math.sqrt(n)
t_statistic = (sample_mean - theoretical_mean)/standard_error

alpha = 0.05
z_critical = stats.norm.ppf(1-alpha)

decision = "Reject the null hypothesis" if t_statistic > z_critical else "Fail to reject the null hypothesis"

print(f"Test Statistic (t): {t_statistic}")
print(f"Critical Value (Z_{alpha}): {z_critical}")
print(f"Decision: We {decision}.")

if decision == "reject the null hypothesis":
    print("Conclusion: There is strong evidence to support the restaurant owners' claim that the weekly operating costs are higher than the model suggests.")
else: 
    print("Conclusion: There is insufficient evidence to support the restaurant owners' claim that the weekly operating costs are higher than the model suggests.")    