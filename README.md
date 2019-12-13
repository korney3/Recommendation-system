# Recommendation-system
Hybrid recommendation system based on collaborative-filtering and user-based features for ["Data-like" ](https://vc.ru/data-like) hackathon dataset made with LightFM Python library. 

The project aims to solve several tasks:

1.  Transform given datasets into format acceptable by LightFM 
2.  Train LightFM model
3.  Predict recommendation of shopping category suitable for ceratin customer_ids


## Dataset
The raw data-like dataset has been preprocessed and saved in two csv-files, containing encoded customers' features and information about customers' transactions.

**customer.csv**
Contain personal information about customer in numerical form.

| customer_id | gender_cd | marital_status_cd | children_cnt | job_position_cd | job_title | first_session_year | first_session_month | first_session_day | first_session_hour |
| ------ | ------ | ----h-- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| ... | ... |  ... | ... | ... | ... | ... | ... | ... | ... |

**transactions.csv**
Weight - number of transactions made by customer in appropriate MCC category. 

| customer_id | merchant_mcc | weight |
| ------ | ------ | ------ |
| ... | ... | ... |




