# Data Preparation


## Claims_data notes:

#### What does this dataset contain?
    This dataset provides for each account_id the portion of a claim that can be expected for each of the 18 month period. Based on the prior claims wrt the account in question, this is assumed to simply represent the average distribution of past claim receipts. 

General:
- Based on the data and the assignment description payments never take longer than 18 months. That means that only claims with `date_received` of `2023-02-01` and later can effect the revenue received post `2024-06-30`.
- The particular day of a month a claim is received is negligent. Payment is assumed to start during the same month regardless.

Filtering out noise
- data will be limited to entries with `date_received` prior to `2024-07-01`
- `is_deleted` is `False` for all accounts (can be ignored)
- `type` is `Full Service` for all accounts (can be ignored)
- We cannot do anything with data for which both `account_id` and `account_name` is `Unknown Account`- cannot be linked to `account_revenue_distribution` dataset (these rows will be dropped)
- `case_numbers` adds no value and will be dropped. 

Imputing or dropping rows with missing/incorrect values
- `industry_segment` is `NaN` for a select number of entries. 
    - In the data you can see that some `account_id`, maps to multiple `industry_segment`s. As such, we cannot impute `industry_segment` from other entries for a particular `account_id`
    - We will thus encode the `industry_segment` column by created columns for each category containing the proportion of an accounts entries represented by the category.
    
- `commercial_subtype` has a many to one relationship with `industry_segment` so its worth initially adding as a feature. It also has a one to one mapping with account_id (so it exists on an account level). This means that we can use it directly for training (one hot encoded). 
    - In theory it would be possible to impute `commercial_subtype` when missing if another entry against the same `account_id` has it filled out. Unfortunaly, for accounts where it is missing it is missing for all entries.
    - There are six accounts for which `commercial_subtype` is NaN after our cleaning steps. As none of these account effect the revenue to be collected, or represent unique behaviour to be learned, and as we have sufficient data remaining, they will be dropped (accounts in question also not present in `arpc_values`).
- Quite a few rows in our dataset have issues wrt date entries. These include:
    - Some entries are invalid or missing (eg. `loss_date` = `3120-04-03` or `Nan`)
    - There are also entries where the date logged for `loss_date` is post the date logged for `date_received` (which is infeasible)
    - Similarly their are entries where the date logged for `start_date` is post either `date_received` or `loss_date` (which is infeasible).
    - WE CANNOT DROP SUCH ENTRIES. We have limited business context. But it is possible that such errors, correspond with errors on the actual claim that was submitted, which could potentially have an impact on both the amount and the timing of claim payouts (see the section on feature engineering to see how we create features to indicate such errors to our model during training).


Initial set of engineered features
As best we can tell, the data in account_revenue_distribution is lagging behind the raw data in claims_data by 5 months(based on lag in derived columns). As such we won't include any entries from the most recent 5 months (between 2024-02-01 and 2024-07-01) when we engineer our features per account. 

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

Special note wrt the effect of payout amount on distribution
- It is highly likely that the amount to be paid out has an effect on the payout distribution. Larger amounts take longer to process. But unfortunately we have not been given payout amount information (`mean_account_arpc`, `median_account_arpc`) for all the account we need to predict. As such we need to create a seperate prediction model to determine those amounts. Should we use the output of that model as input to this model?
    - In general you need to be careful when creating stacked models as it leaves room for compounding erros. We'll keep this in mind as we go.
    - 

**For starters we will include/engineer the following features in our training dataset and make use of pearson correlation :**
- industry_segment (one hot encoded)



## arpc_data notes

- We assume that the mean and median are only determined wrt claims where a payout actually occured.

## account_revenue_distribution notes
- The `claim_count` column in `account_revenue_distribution` is not consistent with the `claims_data` set. For example for `account_id = 0010G00002CbubXQAR` we see a `claim_count` entry of `543`, however, there are 1968 claim entries in `claims_data` wrt this account.
    - This again raises question about how up to date/trustworth the `account_revenue_distribution` dataset is, however we make due for now and simply will not make use of the claim_count column in `account_revenue_distribution` but rather recalculate it from the `claims_data` set.
- Although the sum of payment proportions over 18 months wrt a particular account always sum to 1, there are entries where the proportion for a particular month is indeed greater than 1 or even less than zero (correcting for prior month of > 1). 
    - Note that entries like this represent administration errors. It could be that business wants to include such occurences in predictions, but without further information, we deem that adding them to our training data simply introduces noise at no additional value. There are 21 accounts that have this type of error. As there will remain more than enough accounts for our training set if we remove them (364) we won't use such rows for our training set. 
    - As some of these accounts include accounts for which revenue is still pending, we will maintain there distributions in the data, as for these accounts we will use the detailed distribution instead of rellying on a predicted one. 

## industry_revenue_distribution notes
- Given the very questionable quality of the `claim_count` in `account_revenue_distribution` we won't be making use of the `claim_count` in `industry_revenue_distribution` either. Doing so will likely introduce false information into our training set. 







# Train Test Split
As best we can tell, the data in account_revenue_distribution is lagging behind the raw data in claims_data by 5 months(based on lag in derived columns). As such we won't include any entries from the most recent 5 months (between 2024-02-01 and 2024-07-01) when we engineer our features per account. If we do this, we will be warping the relationship between the datasets and skew our results. 

you need to change the above to fit this section