# Business Problem and Context
Fraudulent credit card transactions has always being a bane for e commerce companies as it negatively impact their brand image among consumers and in the long run drive customers away and increase churn. It is more pressing for e commerce companies to deal with this now as credit card disputes have continued to rise past their pandemic boom. (https://www.straitstimes.com/business/credit-card-disputes-keep-rising-at-visa-as-e-commerce-booms). Disputes are often expensive for both credit card companies and merchants to process and it would be in the interest of both if a potential fraudulent transaction is pre-empted early.

Our company has only done some exploratory analysis on our transaction data and have a rough sense of potential indicators of fraud. However, this isn't sufficent because the analysis is still stuck in jupyter notebooks or powerpoint slides. It would be preferable if we have a ML application that can predict the likelihood of fraud for each transaction in real time. This will substantially help to reduce fradulent transactions and help stem losses in revenue due to these cases.

# Data availability
Data in this case is readily available, as we (the company) tracks each and every POS transactions made by each individual. We do have details such as:
1) Transaction time
2) credit card number (Pii info)
3) Transaction Amount
4) Gender
5) Merchant
6) City
7) State
8) DOB (Allows us to back calculate age)
9) First & Last name (Pii info)

Both credit card number and the user names are Pii information and we must *ensure* that these fields will not be used in any form throughout the ML modeling process.


