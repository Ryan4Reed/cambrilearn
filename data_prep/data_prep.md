## Data Preparation for account_revenue_distribution Prediction Task


### Claims_data notes:

General:
- Based on the data and the assignment description payments never take longer than 18 months. That means that only claims with `date_received` of `2023-02-01` and later can effect the revenue received post `2024-06-30`.

Filtering out noise
- data will be limited to entries with `date_received` prior to `2024-07-01`
- `is_deleted` is `False` for all accounts (can be ignored)
- `type` is `Full Service` for all accounts (can be ignored)
- We cannot do anything with data for which both `account_id` and `account_name` is `Unknown Account`- cannot be linked to `account_revenue_distribution` dataset (these rows will be dropped)

Imputing or dropping rows with missing/incorrect values
- `industry_segment` is `NaN` for a select number of entries. However, in the data you can see that every `account_id`, maps to a single `industry_segment`. As such, where possible we will impute `industry_segment` from other entries for a particular `account_id`
    - None of the accounts that impact the revenue to be received has `industry_segment` entries of Unassigned/NaN. If we will not see such entries in our predictions, the algorith need not map such instances (we need not include it in our training set). As such, if there are any that remain after the above, they will be dropped.
- Quite a few rows in our dataset have issues wrt date entries. These include:
    - Some entries are invalid or missing (eg. `loss_date` = `3120-04-03` or `Nan`)
    - There are also entries where the date logged for `loss_date` is post the date logged for `date_received` (which is infeasible)
    - Similarly their are entries where the date logged for `start_date` is post either `date_received` or `loss_date` (which is infeasible).
    - WE CANNOT DROP SUCH ENTRIES. We have limited business context. But it is possible that such errors, correspond with errors on the actual claim that was submitted, which could potentially have an impact on both the amount and the timing of claim payouts (see the section on feature engineering to see how we create features to indicate such errors to our model during training).


Initial set of engineered features
- `commercial_subtype_proportion`: `account_id` has a one to many relationship with `commercial_subtype` (doesn't exist on an account level). This means that we cant use it for training. We need to convert it to an account level feature: A feature that indicates the proportion of account entries represented by each `commercial_subtype`.
- The existing derived columns [`months_since_joined`, `last_year_claim_count`, `months_since_last_claim`, `two_months_dry`,  `six_months_dry`, `one_year_dry`, `two_years_dry`] is not consistent with the raw data (seems to be 5 months behind). They need to be recomputed.
<!-- as they could include data that sits outside the period of interest (entries with a `date_received` after to `2024-06-30`) -->
    - we will also add a `four_years_dry` column
- We mentioned above that there often issues wrt the date columns in our data. We will handle these as follows in order to indicate such issues on an account level:
    - Incorrect date values (`3120-04-03`) and missing date values (`Nan`) will be treated in the same way. Motivation is for the same of simplicity and because it is likely that the effect of such errors on the payout distribution is very similar.
    - We will create a feature named: date_entry_issues that indicates the proportion of entries per account that has such date issues.
- The second type of issue we see wrt date issues is that there is at times a significant gap between the specified loss_date and the date_received. This likely contributes to the payout distribution per account. As sush:
    - We will engineer a feature indicating the average time (in days) between these two dates.

Special note wrt the effect of payout amount on distribution
- It is highly likely that the amount to be paid out has an effect on the payout distribution. Larger amounts take longer to process. But unfortunately we have not been given payout amount information (`mean_account_arpc`, `median_account_arpc`) for all the account we need to predict. As such we need to create a seperate prediction model to determine those amounts. Should we use the output of that model as input to this model?
    - In general you need to be careful when creating stacked models as it leaves room for compounding erros. We'll keep this in mind as we go.
    - 

**For starters we will include/engineer the following features in our training dataset and make use of pearson correlation :**
- industry_segment (one hot encoded)
