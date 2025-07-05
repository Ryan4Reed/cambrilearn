# Approach Overview

After reading through the assignment and spending some time with the data, it’s clear that the project is not a typical data science project. Reason being that the datapoint that needs to be predicted (ie. `account_revenue_distribution`) and the raw data provided as input to the prediction task lives on different levels of aggregation. More specifically, the provided information wrt `account_revenue_distribution` has been aggregated across claims wrt a particular account. 

Let me illustrate what I mean. In our `claims_data` dataset, we have an entry per claim. But our `account_revenue_distribution` data is given wrt a particular account. Typically, for a project like this we would expect to receive the revenue distribution for each claim entry, but we don’t have that. What we have is the aggregated distribution based on all claims per account. 

The practical impact of this is that the size of our data decreases significantly as features we use/engineer will need to live on an account level, not a claim level. So we have as many datapoints as there are accounts.

Using such an aggregated distribution for training is equivalent to training a model on the output of another model, which is not something you want to do if you can avoid it (especially when you don’t have clarity on exactly how the aggregated distributions were calculated). However, in this case it is the best option available to us. 

#### Prediction tasks
The overall goal of the project is to prediction the distribution of revenue to be expected for each account post the date of interest (`2024-30-06`). However due to the fact that for the accounts that effect this metric there are some for which we are not given the:
- `median_account_arpc`
- `account_hit_success_rate`
- `account_revenue_distribution`

We will need to train a prediction model in order to determine these values. Once we have these values, we will calculate the expected revenue per account, per month by counting the number of open claims, multiplying that by both the expected `median_account_arpc`  and the `account_hit_success_rate`. This is not the only option, but for now it is deemed our best available option.


---


# Data Preparation
Lets step through some of the more important points wrt how we go about preparing our data.

#### Claims_data notes:

    This dataset provides historic data wrt the claims that have been processed for particular accounts. Importantly, we do not know particular amounts wrt claims, simply when they occured. 

Here follows a list of bullets details points to note wrt preparting this dataset:
- Based on the data and the assignment description payments never take longer than 18 months. That means that only claims with `date_received` of `2023-02-01` and later can effect the revenue received post `2024-06-30`.

- The particular day of a month a claim is received is negligent. Payment is assumed to start during the same month regardless.

- data will be limited to entries with `date_received` prior to `2024-07-01`

- `is_deleted` is `False` for all accounts (can be ignored)

- `type` is `Full Service` for all accounts (can be ignored)

- We cannot do anything with data for which both `account_id` and `account_name` is `Unknown Account`- cannot be linked to `account_revenue_distribution` dataset (these rows will be dropped)

- `case_numbers` adds no value and will be dropped. 

- `industry_segment` 
    - `industry_segment` is `NaN` for a select number of entries. 
    - In the data you can see that some `account_id`, maps to multiple `industry_segment`s. As such, we cannot impute `industry_segment` from other entries that have it filled out for a particular `account_id`.
    - Further in order to make use of the information in industry_revenue_distribution, we need to be able to map an account to a particular industry. 
    - Note that no accounts have industry_segment entries of NaN for are all its rows.
    - As such we will adopt the following strategy:
        - We will encode the `industry_segment` column by creating columns for each category containing the proportion of an accounts entries represented by the category (including an unnassigned/nan category).
        - When we use the revenue_proportion data from industry_revenue_proportion we will weight the revenue_proportion for each month according to the proportion of an accounts entries represented by each industry.
    
- `commercial_subtype` has a many to one relationship with `industry_segment` so its worth initially adding as a feature. It also has a one to one mapping with account_id (so it exists on an account level). This means that we can use it directly for training (one hot encoded). 
    - In theory it would be possible to impute `commercial_subtype` when missing if another entry against the same `account_id` has it filled out. Unfortunaly, for accounts where it is missing it is missing for all entries.
    - There are six accounts for which `commercial_subtype` is NaN after our cleaning steps. As none of these account effect the revenue to be collected, or represent unique behaviour to be learned, and as we have sufficient data remaining, they will be dropped (accounts in question also not present in `arpc_values`).
- Quite a few rows in our dataset have issues wrt date entries. These include:
    - Some entries are invalid or missing (eg. `loss_date` = `3120-04-03` or `Nan`)
    - There are also entries where the date logged for `loss_date` is post the date logged for `date_received` (which is infeasible)
    - Similarly their are entries where the date logged for `start_date` is post either `date_received` or `loss_date` (which is infeasible).
    - WE CANNOT DROP SUCH ENTRIES. We have limited business context. But it is possible that such errors, correspond with errors on the actual claim that was submitted, which could potentially have an impact on both the amount and the timing of claim payouts (see the section on feature engineering to see how we create features to indicate such errors to our model during training).

- As best we can tell, the data in account_revenue_distribution is lagging behind the raw data in claims_data by 5 months(based on lag in derived columns). As such we won't include any entries from the most recent 5 months (between 2024-02-01 and 2024-07-01) when we engineer our features per account. 

- The existing derived columns [`months_since_joined`, `last_year_claim_count`, `months_since_last_claim`, `two_months_dry`,  `six_months_dry`, `one_year_dry`, `two_years_dry`] are not consistent with the raw data (5 months behind). After investigating the `claim_count` column in `account_revenue_distribution` we can see that the same thing is true for this dataset. It is also lagging (assuming by the same period of 5 months - no way to verify). If we were to recalculate the derived columns in `claims_data` the datasets would no longer be in sync (we have no way of updating `account_revenue_distributions`). If we do this, we will be warping the relationship between the datasets and end up skewing our results. As a result, instead of updating the derived columns, we will limit the data we use for creating any features to those having a `date_received` of < 5 months prior to the `reference_date`. By doing this we keep our data in sync, and do not introduce bias into our data.
    - We will however recalculate the derived columns based on this period as we don't have access to the logic used to compute them and we can't check each entry.
    - we will also add a `four_years_dry` column.
    - Features from pre_sync_data will be use during traing and from complete data during testing

- The `is_closed` column indicating whether 18 months have passed since the `received_date` month is also outdated and will be recalculated.
- We mentioned above that there often issues wrt the date columns in our data. We will handle these as follows in order to indicate such issues on an account level:
    - Incorrect date values (`3120-04-03`) and missing date values (`Nan`) will be treated in the same way. Motivation is for the same of simplicity and because it is likely that the effect of such errors on the payout distribution is very similar.
    - We will create a feature named: date_entry_issues that indicates the proportion of entries per account that has such date issues.
- The second type of issue we see wrt date issues is that there is at times a significant gap between the specified loss_date and the date_received. This likely contributes to the payout distribution per account. As sush:
    - We will engineer a feature indicating the average time (in days) between these two dates.
    - We will exclude entries that have any of the above issues in our calculations.
    - If all entries for a paricular account have loss_date as NaN we will set the diff to -1 so that the algorithms can handle it (NaN not supported for all models).



#### arpc_data notes
    
    This dataset provides historic data wrt the typical amounts received and success rate of claims on both an account and an industry level.

Here follows a list of bullets details points to note wrt preparting this dataset:
- We assume that the mean and median are only determined wrt claims where a payout actually occured.

- The `industry_segment` entry against each `account_id` in `claims_data` does not correspond with the `industry_segment` entry against each `account_id` in `arpc_values`. This is a problem for joining and using the industry level information in `arpc_data`. 
    - We are going to assume that the relationship between the `industry_segment` column and other industry level columns (eg. [`mean_industry_arpc`, `industry_hit_success_rate`, ...]) is correct. And that the error lies between the account_number and the `industry_segment` assigned. For that we will use `claims_data` as the source of truth.
    - This is another step away from a trustworthy outcome.

- We can also see that across different `commercial_subtype` entries wrt the `commercial` `industry_segment`, the industry level values stay consistent. So we need not account for `commercial_subtype` in the join.
- Our data has two columns providing information on the typical amounts paid out wrt account claims: `Mean`, and `Median`. Which should we use as the target variable of our prediction tasks wrt payout amounts per account? 
    - The primary difference between mean and median wrt our goal, is that mean is sensitive to outliears, while median is more resistent to the influence of outliers. We were asked to provide business with an estimate of the amount of profit that can be expected. In this case, it is better to be conservative. So that we don't over estimate profits based on the influence of rare outliers. If we had rather been asked to provide insight into potential costs, it have been better to be less conservative in our predictions. As we would not want to be surprised by outlier costs. As such we will make use of median as our target variable.
- Almost all of our datapoints have an median_account_arpc of less then 5000. Then there are a handful of entries that range between [5000, 25000]. Such outliers will most likely make it difficult for any model to be performant given our limited data quality. We are further motivated to drop such values given the same reasoning as discussed in the `Predicting mean vs median` section. However, what we will do is make an informed decision and test model performance when including these datapoints and then removing them. 
    - After experimenting the model was able to perform significantly better on the test set after removing these outliers. E.g RMSE went from 5000+ to plus minus 700. R2 went from < 0 which is worse then predicting mean to Test R^2: 0.1960. As such we proceed without them. 



#### account_revenue_distribution notes

    This dataset provides for each account_id the portion of a claim that can be expected for each of the 18 month period. Based on the prior claims wrt the account in question, this is assumed to simply represent the average distribution of past claim receipts. 

Here follows a list of bullets details points to note wrt preparting this dataset:
- The `claim_count` column in `account_revenue_distribution` is not consistent with the `claims_data` set. For example for `account_id = 0010G00002CbubXQAR` we see a `claim_count` entry of `543`, however, there are 1968 claim entries in `claims_data` wrt this account.
    - This again raises question about how up to date/trustworth the `account_revenue_distribution` dataset is, however we make due for now and simply will not make use of the claim_count column in `account_revenue_distribution` but rather recalculate it from the `claims_data` set.

- Although the sum of payment proportions over 18 months wrt a particular account always sum to 1, there are entries where the proportion for a particular month is indeed greater than 1 or even less than zero (correcting for prior month of > 1). 
    - Note that entries like this represent administration errors. It could be that business wants to include such occurences in predictions, but without further information, we deem that adding them to our training data simply introduces noise at no additional value. There are 21 accounts that have this type of error. As there will remain more than enough accounts for our training set if we remove them (364) we won't use such rows for our training set. You can essentially views these rows as outliers that are being removed.
    - As some of these accounts include accounts for which revenue is still pending, we will maintain there distributions in the data, as for these accounts we will use the detailed distribution instead of rellying on a predicted one. 

#### industry_revenue_distribution notes

    This dataset provides for each industry_segment the portion of a claim that can be expected for each of the 18 month period. Based on the prior claims wrt the industry_segment in question, this is assumed to simply represent the average distribution of past claim receipts per month. 

Here follows a list of bullets details points to note wrt preparting this dataset:
- Given the very questionable quality of the `claim_count` in `account_revenue_distribution` we won't be making use of the `claim_count` in `industry_revenue_distribution` either. Doing so will likely introduce false information into our training set.
- Unlike `account_revenue_distribution` this dataset does not have any invalid (not in `[0, 1]`) or `NaN` entries in the `revenue_proportion` column.



### Joining Datasets notes
- Note that within arpc_values, we do not have industry level information for industry_segment: `Insurance Company'. As such when we calculate industry level values for accounts represented across multiple industries according to the proportion of their entries represented by each, we will ignore this category. It is unfortunately the best we can do.
- Additionally the `Insurance Company` `industry_segment` is also not present in the `industry_revenue_distribution` dataset. 
    - We will again weight the `revenue_proportion` per month assigned to each account by the industries that represent them. During this exercise we will once again have to ignore the  `Insurance Company` category.

### Feature Validation
- We use the same set of features across our three prediction tasks. All features have a logical relationship to each of the target variables.
- We can see in our pearson correlation plots that none of the features have an absolute correlation of more than 0.4, which is not ideal. However, we have engineered a proper set of features. The issue points back to the concerns we've raised about data quality. We will hope that combinations between features give the algorithms additional predictive power.

---

# Making Predictions

#### Initial model selection
For our prediction tasks, we distinguish between scalar targets and vector targets representing distributions over time.

For scalar predictions:

    median_account_arpc, account_hit_success_rate
We experimented with
- ElasticNet
- XGBoost

These two models provide complementary approaches. ElasticNet linear modelling, and XGBoost for non-linear modeling. Including both allows us to compare model performance and choose the best.

For vector prediction:

    account_rev_month_x

We wanted to select a model that could enforce a sumation of 1 across proportions during training, so that it could potentially learn the relationship between month proportions. As such we decided to use:
-  Centered Log-Ratio (CLR) transformation of target with XGBoost

If there had been more time and data worth investing the time in we could also have experimented with
- softmax neural net with cross entropy loss function
- Dirichlet Regression



#### Feature scaling
One of the models we used greatly benefits from feature scaling:
- ElasticNet is based on the assumption that features are scaled to [0, 1]. This is so that its internal regularisation mechanism are sensible and the model can converge. 

As such, we are going to scale our data as part of our preprossing even though Xgboost is not sensitive to scaling. This means that we will lose a measure of feature interpretability wrt our results. 


#### Including targets as features?
Should we include the predicted median_account_arpc values when predicting account_hit_success_rate. It is logical to expect there to be a relationship between these metrics. However, given that we already provide the models with industry level information wrt median, success rate and proportion. We decided to relly on these instead of stacking models. If there was time we would test both to see which performs best.


#### Hyperparameter tuning
For each model we've used grid search to tune the models hyperparameters. In addition we plot each models performance on the validation set as eac of the parameters are varied in order to get a better picture of whether the ranges we've set for the parameters to explore are sufficient. However the score on the final test set is used to govern the selection. Sometimes adding additional ranges to explore per perameter results in better performance on the validation set (so gridsearch selects those) but worse performance on the final test set. We tune these ranges by hand with trail and error to get the best result. 


# Final model selection
After experimenting with our models, the only task for which a model could outperform simply predicting the mean was the `account_median_arpc` prediction. Our Xgboost model achieved an R2 of 0.25, which is not great, but is better than simply predicting the mean. As such we will use the following approach for our prediction tasks. We will use:
    - the xgboost model for predicting median_account_arpc
    - mean across account for predicting account_hit_success_rate
    - per month mean for predicting account_revenue_proportions

----
