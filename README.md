## Business Problem and Context
Fraudulent credit card transactions has always being a bane for e commerce companies as it negatively impact their brand image among consumers and in the long run drive customers away and increase churn. It is more pressing for e commerce companies to deal with this now as credit card disputes have continued to rise past their pandemic boom. (https://www.straitstimes.com/business/credit-card-disputes-keep-rising-at-visa-as-e-commerce-booms). Disputes are often expensive for both credit card companies and merchants to process and it would be in the interest of both if a potential fraudulent transaction is pre-empted early.

Our company has only done some exploratory analysis on our transaction data and have a rough sense of potential indicators of fraud. However, this isn't sufficent because the analysis is still stuck in jupyter notebooks or powerpoint slides. It would be preferable if we have a ML application that can predict the likelihood of fraud for each transaction in real time. This will substantially help to reduce fradulent transactions and help stem losses in revenue due to these cases. In particular we would be focusing more on the recall metric - as would want the ML model to get it right most of the time out of those actual fradulent cases that the system is exposed to.

## Data availability
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

## ML Solution Type determination
Given that we are going to predict fradulent transactions, this will obviously be a typical classification ML problem.  The  I would be quite inclined to go straight to tree based ensemble methods such as random forests and extreme gradient boosting. Preliminary analysis have already shown that baseline XGBoost already has a higher recall score as compared to random forest and this coupled with its easier scalability makes me lean towards it. With appropriate fine tuning of hyperpameters, I believe that the model's performance will further improve.

## Important command to build docker locally with mounting
`docker build -t fraud_mlops .`

To run with mounting
* `docker run -v $(pwd)/data:/opt/ml/input/data -v $(pwd)/output:/opt/ml/output -v $(pwd)/model:/opt/ml/model fraud_mlops`

To execute the build and push.sh file
* chmod +x ./build_and_push.sh
* ./build_and_push.sh

## How are we serving
Given that we are needing the ML model to prevent potential fradulent transactions, we would be needing it to predict in real time. At the very least, we would require a functional API of production quality, and we are thinking of deployment the model as an endpoint first before hooking it up to an API gateway which can be consumed by our online transaction systems.


