# Kaggle Competition: TalkingData AdTracking Fraud Detection Challenge

Authors: **Kevin Liao**

### Objective 
This repository is mainly about my learning experience in Kaggle Competition. It consists of python scripts (for model training and model selection) and jupyter notebook (for EDA). I hope beginers can find something useful in this repo. For me, I think I might re-use some of the tools from this repo as well.

Competition Website: [TalkingData](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)

### File Structure

The main directories of this repository are:
* `data`, which stores the original data set `train.csv`, `test.csv`, and `test_supplement.csv`
* `scripts`, which holds the meat for the competition. It includes feature engineering and model trainning/prediction
* `eda_nb`, which stores jupyter notebooks and HTML for some EDA process and output
* `insample_iterations`, which is reponsible for in-sample model selection, tuning and evaluations
* `images`, which stores the graphic output for EDA
* `reference`, which contains other top kagglers' scripts and tutorials

The complete file-structure for the project is as follows:

```
TalkingData/
    README.md
    LICENSE
    data/
        README.md
        train.csv
        test.csv
        test_supplement.csv
        train_v1.hdf
        test_v1.hdf
        train_v2.hdf
        test_v2.hdf
        train_v3.hdf
        test_v3.hdf
    scripts/
        feature_eng-v1.py
        train_xgb-v1.py
        feature_eng-v2.py
        train_lightgbm-v2.py
        feature_eng-v3.py
        train_lightgbm-v3.py
    eda_nb/
        basic_EDA.ipynb
        basic_EDA.html
        better_EDA.ipynb (jtter plot)
        better_EDA.html
        SHAP_toy_example.ipynb
        SHAP_toy_example.html
        BayesOpt_toy_example.ipynb
        BayesOpt_toy_example.html
        Boruta_algo_toy_example.ipynb
        Boruta_algo_toy_example.html
    insample_iterations/
        README.md
        data/
            train_raw.hdf
            test_raw.hdf
            train.hdf
            test.hdf
        scripts/
            dump_in_sample_data.py
            feature_eng.py
            train_model.py
    images/
        ...too many
    reference/
        ...good stuff
```

### REFERENCES

[IP address encoding issues](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/52374)

[EDA](https://www.kaggle.com/pranav84/talkingdata-eda-to-model-evaluation-lb-0-9683)

[FM_FTRL](https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9752/code)
