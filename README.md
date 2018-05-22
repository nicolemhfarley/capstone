# capstone

## Predicting Business Needs Through Time Series Analysis

### Background:
WellPsyche is a company that manages mental health services.  The healthcare providers that they employ fall into three main categories: doctors/psychiatrists, RN/PAs and therapists.  The current lead time to hiring a new provider is between two and three weeks.  WellPsyche hoped to increase this to between two and three months.  

### Objective:
Given the past 3 years of appointment data, the goal was to use time series analysis to model demand over time and be able to predict future demand, thus providing WellPsyche with at least 2-3 months advance notice to hire new providers if needed.  

### Challenges:
I represented demand for mental health services by the weekly sum of appointment hours logged by each category of provider.  Although there was a general increasing trend over time, there was significant variation in the data throughout any given year.  Some of these affects were directly attributable to yearly events such as major holidays.  However this could not account for all of the variability that I saw.  After predicting the number of appointment hours that will be needed for each category, I used the average number of hours per each type of provider to estimate the necessary number of each provider that would be need at these times.

### Primary Models:
The two main models I employed were ARIMAX (using the number of providers over time as an exogenous variable) and Facebook Prophet (with the addition of the following holidays: Thanksgiving, Christmas, New Years).
