# Kaggle Competition: TalkingData AdTracking Fraud Detection Challenge

Authors: **Kevin Liao**

### Objective
This is where the best practice happens in data science. This place is responsible for in-sample data mining. We knew that Kaggle only allows maximum of five submission per day, which is good. Limiting iteration frequency in out-of-sample socring can help reduce chance of overfitting. However, we need a hold-out set for in-sample feature selection and model selection. And this repository is for in-sample data mining purpose.

### File Structure

The main directories of this repository are:
* `data`, which stores the original data set `train_raw.hdf`, `test_raw.hdf`, `train.hdf`, and `test.hdf`
* `scripts`, which holds feature engineering, model training, and feature selection scripts

The complete file-structure for the project is as follows:

```
insample_iterations/
    README.md
    data/
        train_raw.hdf
        test_raw.hdf
        train.hdf
        test.hdf
    scripts/
        dump_in_sample_data.py
        feature_engineering.py
        feature_univariate_selection.py
        feature_forward_selection.py
        feature_backward_selection.py
        feature_permutation_selection.py
        train_model.py
```
