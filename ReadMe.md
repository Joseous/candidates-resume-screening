# Candidate Selection Prediction

## Project Overview

This project develops a machine learning classification model to predict whether a candidate will be **Selected** or **Rejected** during a recruitment process.

The project follows an end-to-end data science workflow:

* Data inspection and cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Statistical hypothesis testing
* Categorical encoding and numerical scaling
* Model comparison using cross-validation
* Hyperparameter tuning
* Final evaluation on a held-out test set

The final model selected for this project was **Gradient Boosting**, which achieved the best performance after hyperparameter tuning.

---

## 1. Dataset Overview

The dataset contains approximately **100,000 candidate records** and includes information about candidates' education, experience, skills, assessments, resume characteristics, and other recruitment-related attributes.

The target variable is:

* `selected` — whether the candidate was selected or rejected.

The target contains two classes:

* `Selected`
* `Rejected`

### Main types of features

The dataset contains:

* Numerical features such as:

  * `experience_years`
  * `interview_score`
  * `technical_test_score`
  * `ats_score`
  * `keyword_match_percentage`
  * `cgpa`
  * `projects_completed`
  * `publications`
  * `expected_salary`
  * `resume_quality_score`
  * `communication_score`
  * `problem_solving_score`
  * `aptitude_score`

* Categorical features such as:

  * `gender`
  * `highest_education`
  * `employment_type`
  * `remote_preference`
  * `availability`
  * `job_role`
  * `country`
  * `university`
  * `field_of_study`

* Comma-separated/list-like columns such as:

  * `technical_skills`
  * `programming_languages`
  * `frameworks`
  * `databases`
  * `cloud_platform`
  * `certifications`
  * `previous_companies`

---

# 2. Handling Comma-Separated Columns

Several columns contained multiple values stored in a single cell.

For example:

```text
Scikit-learn, HTML, Git
```

Rather than one-hot encoding these raw strings, I transformed them into numerical count features.

For example:

```text
Scikit-learn, HTML, Git → 3
```

The value:

```text
Not Applicable
```

was treated as zero.

The following columns were converted into count-based features:

* `technical_skills`
* `programming_languages`
* `frameworks`
* `databases`
* `cloud_platform`
* `certifications`
* `previous_companies`

The resulting features included:

```text
technical_skills_count
programming_languages_count
frameworks_count
databases_count
cloud_platform_count
certifications_count
previous_companies_count
```

This approach converted the list-like text into information that machine learning models could use while avoiding extremely high-dimensional one-hot encoding.

### High-cardinality columns

Some columns had very high numbers of unique values.

For example:

* `city` had approximately 37,950 unique values.
* `preferred_location` had approximately 15,831 unique values.
* `previous_companies` contained approximately 34,104 unique values.
* `technical_skills` contained approximately 97,372 unique values.

Because of their high cardinality and free-text/list-like nature, `city` and `preferred_location` were excluded from the model, while useful information from `previous_companies` and `technical_skills` was captured through engineered count features.

---

# 3. Exploratory Data Analysis

EDA was performed to understand the distribution of the data and identify relationships between candidate characteristics and selection outcomes.

### Key EDA findings

### Candidate scores

The candidate assessment and resume-related scores showed important differences between selected and rejected candidates.

In particular, features such as:

* `interview_score`
* `technical_test_score`
* `communication_score`
* `problem_solving_score`
* `aptitude_score`
* `resume_quality_score`

were examined to determine whether stronger candidate scores were associated with selection.

### Education

Selection rates were compared across different `highest_education` categories.

The analysis showed that selection rates is rougly equal across education levels, suggesting that educational background may not be associated with candidate selection.

### Internship and leadership experience

Selection rates were also compared for:

* `internship_experience`
* `leadership_experience`

These variables were investigated to determine whether candidates with such experience had higher selection rates.
- __Leadership experience:__ Candidates with leadership experience recoerded higher selection rate than those without.
With p-value of below 0.05, it means There is a statistically significant association with selection.

### Technical skills

The engineered `technical_skills_count` feature was analyzed against the target.

This allowed the project to evaluate whether candidates with a larger number of listed technical skills tended to have different selection outcomes.
- The selected candidates have higher mean compared to rejected in relation to technical skills count

### Correlation between candidate scores

A correlation heatmap was created for:

* `communication_score`
* `problem_solving_score`
* `technical_test_score`
* `interview_score`
* `aptitude_score`
* `resume_quality_score`

The heatmap was used to identify groups of scores that moved together and to assess potential redundancy between variables. 

**Key correlation finding:**

> There was no correlation among the columns.

---

# 4. Hypothesis Testing

Statistical hypothesis testing was used to investigate whether candidate features were significantly associated with the selection outcome.

The significance level was:

```text
α = 0.05
```

## Numerical Features — Independent t-test

Because the target contained two groups (`Selected` and `Rejected`), independent-samples t-tests were used for numerical variables.

### Hypotheses

**H₀:** The mean value of the feature is the same for selected and rejected candidates.

**H₁:** The mean value of the feature differs between selected and rejected candidates.

Welch's independent t-test was used because it does not require the two groups to have equal variances.

Features with:

```text
p-value < 0.05
```

were considered statistically significant at the 5% significance level.

### Main findings

The hypothesis tests identified several numerical features with statistically significant differences between selected and rejected candidates.

Some of Numerical features strongest findings included:

| Feature     | Test                  |   p-value    | Significant? |
| --------------- | ----------------- | -----------: | ------------ |
| age             | independent ttest | 0.000000e+00 | Yes          |
| cgpa	          | independent ttest | 0.000000e+00 | Yes          |
| interview_score | independent ttest | 0.000000e+00 | Yes          |
| aptitude_score  | independent ttest | 9.423158e-42 | Yes          |
| expected_salary | independent ttest | 0.000000e+00 | Yes          |

These results indicated that candidate assessment and/or profile characteristics were associated with selection outcomes.

---

## Categorical Features — Chi-square Test

For categorical variables, the Chi-square test of independence was used.

### Hypotheses

**H₀:** The categorical feature and candidate selection are independent.

**H₁:** The categorical feature and candidate selection are associated.

A p-value below 0.05 resulted in rejection of H₀.

### Main findings

The Chi-square analysis identified the following categorical variables as statistically associated with selection:

| Feature             | Chi-square statistic |   p-value   | Significant? |
| ------------------- | -------------------: | ----------: | ------------ |
| current_job_title   |  6026.502683         |0.000000e+00 | Yes          |
|internship_experience|  824.617409          |2.398245e-181| Yes          |
|leadership_experience|  1715.917172         |0.000000e+00 | Yes          |

These statistical tests were used as part of feature analysis, but statistical significance was not treated as the only criterion for determining whether a feature should be used in machine learning.

---

# 5. Feature Preprocessing

Before model training, the data was divided into training and test sets.

Categorical variables were encoded using **One-Hot Encoding**, while numerical variables were standardized using **StandardScaler**.

The preprocessing was implemented inside a scikit-learn `Pipeline` and `ColumnTransformer`.

This ensured that preprocessing was performed correctly within the cross-validation process and helped prevent data leakage.

The target variable was encoded as:

```text
Rejected → 0
Selected → 1
```

Therefore, the F1-score focused on the `Selected` class.

---

# 6. Model Comparison

Several classification algorithms were initially evaluated using stratified cross-validation.

The models included:

* Logistic Regression
* Decision Tree
* Random Forest
* K-Nearest Neighbors
* Gradient Boosting
* XGBoost

The models were compared using:

* Cross-validation accuracy
* Cross-validation F1-score

F1-score was particularly useful because it balances precision and recall for the positive class (`Selected`).

### Initial model comparison

| Model               | Mean CV Accuracy | Mean CV F1 |
| ------------------- | ---------------: | ---------: |
| Logistic Regression |  0.813762        | 0.814890   |
| Decision Tree       |  0.812813        | 0.828404   |
| Random Forest       |  0.833100        | 0.844955   |
| KNN                 |  0.801775        | 0.807155   |
| Gradient Boosting   |  0.844750        | 0.849496   |
| XGBoost             |  0.838862        |  0.843612  |

Based on cross-validation performance, **Random Forest and Gradient Boosting** were selected as the top two models for further hyperparameter tuning.

---

# 7. Hyperparameter Tuning

The top two models were tuned using `RandomizedSearchCV`.

Randomized search was chosen instead of an exhaustive grid search because the dataset contains approximately 100,000 rows, making a large GridSearchCV computationally expensive.

The tuning process used cross-validation and optimized for F1-score.

The key hyperparameters considered included:

### Random Forest

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

### Gradient Boosting

* `n_estimators`
* `learning_rate`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`


After tuning, **Gradient Boosting produced the best overall validation performance**.

---

# 8. Final Model

The final selected model was:

## Gradient Boosting Classifier

The best hyperparameters identified during randomized search were:

```text
Best Parameters
{'model__n_estimators': 250, 'model__min_samples_split': 2, 'model__min_samples_leaf': 2, 'model__max_depth': 3, 'model__learning_rate': 0.05}

Best cv_f1
0.8512812047489643
```

For example:

```python
{
    "n_estimators": 250,
    "learning_rate": 0.05,
    "max_depth": 3,
    ...
}
```

The best estimator was refitted using the complete training dataset and evaluated on the held-out test set.

The test set was kept separate during model selection and hyperparameter tuning to provide an unbiased final evaluation.

---

# 9. Final Test-Set Performance

The final Gradient Boosting model achieved:

| Metric        |                 Score |
| ------------- | --------------------: |
| Test Accuracy | **0.84**              |
| Test F1-score | **0.85**              |

### Confusion Matrix

The final confusion matrix was:

```text
                 Predicted
                 Rejected  Selected
Actual Rejected     [7994]    [2006]
Actual Selected     [1153]    [8847]
```

### Interpretation

* **True Negatives (TN):** [7994]
  Rejected candidates correctly predicted as Rejected.

* **False Positives (FP):** [2006]
  Rejected candidates incorrectly predicted as Selected.

* **False Negatives (FN):** [1153]
  Selected candidates incorrectly predicted as Rejected.

* **True Positives (TP):** [8847]
  Selected candidates correctly predicted as Selected.

The F1-score provides a useful summary of the model's ability to correctly identify selected candidates while balancing precision and recall.

---

# 10. Key Conclusions

Several conclusions were drawn from the analysis:

1. Candidate assessment and profile-related variables showed meaningful relationships with selection outcomes.
2. Several numerical features showed statistically significant differences between selected and rejected candidates.
3. Some categorical variables were significantly associated with selection according to Chi-square tests.
4. The comma-separated skill and experience fields were successfully converted into numerical count features rather than being directly one-hot encoded.
5. Cross-validation provided a more reliable comparison of the candidate classification models.
6. Randomized hyperparameter search was more computationally practical than an exhaustive grid search for a dataset of approximately 100,000 rows.
7. **Gradient Boosting was the best-performing model after hyperparameter tuning.**
8. The final model was evaluated only once on the held-out test set after model selection and tuning.

---

# 11. Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* SciPy
* Scikit-learn
* XGBoost
* Jupyter Notebook
* Kaggle

---

## Final Outcome

This project demonstrates an end-to-end machine learning workflow for a binary classification problem, from raw candidate data and exploratory analysis through statistical testing, feature engineering, model comparison, hyperparameter tuning, and final test-set evaluation.

The final selected model was **Gradient Boosting**, based on its performance after hyperparameter tuning.
