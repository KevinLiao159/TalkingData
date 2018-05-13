# Kaggle Competition: TalkingData AdTracking Fraud Detection Challenge

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/KevinLiao159/klearn/blob/master/LICENSE)

Authors: **Kevin Liao**

## Objective 
This repository is mainly about my learning experience in Kaggle Competition. It consists of python scripts (for feature engineering, model training and model selection) and jupyter notebook (for EDA). I hope beginers can find something useful in this repo. For me, I think I might recycle some of the tools in this repo as well.

Competition Website: [TalkingData](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection)


## File structure

The main directories of this repository are:
* `data`, which stores the original data set `train.csv`, `test.csv`, and `test_supplement.csv`
* `scripts`, which holds the meat for the competition. It includes feature engineering and model trainning/prediction
* `models`, which stores trained models (trained object)
* `eda_nb`, which stores jupyter notebooks and HTML for some EDA process and output
* `insample_iterations`, which is reponsible for in-sample model selection, tuning and evaluations
* `images`, which stores the graphic output for EDA
* `reference`, which contains other top kagglers' scripts and tutorials

The complete file-structure for the project is as follows:

```
TalkingData/
    README.md
    LICENSE
    requirements.txt
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
    models/
        model_lgbm.txt
    eda_nb/
        basic_EDA.ipynb
        basic_EDA.html
        better_EDA.ipynb
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
            feature_engineering.py
            feature_univariate_selection.py
            feature_forward_selection.py
            feature_backward_selection.py
            feature_permutation_selection.py
            train_model.py
    images/
        ...too many random plots
    reference/
        ...good stuff
```

## Some thoughts

This is my first Kaggle competition. I got myself ranked **top 3%** in *public leaderboard* just a week before the final submission deadline. My final submission is ranked **top 15%** in *private leaderboard*. It's so far one of the most competitive competion in which roughly 4000 kagglers participated. 

It's a two month duration competition but I spent only about 80 hours in it because I have a full-time job and I can only kaggle during the weekend. Because time resource is too limited, I decided to implement just a single model (no blending/stacking) and my final submission is from that single model. I knew myself was at disadvantage and many other kagglers implements muliple models and blend them to form final submission. But I value this experience and I have learned more than what I expected before going into the competition.

__Benefits of kaggling.__
1. Adjust my bias (prior belief)
2. Knowing where I am at (weakness and strength)
3. Opens up my mind and learn a totally different approach to tackle the problem
4. help me not stuck in local minimum (better benchmarking)
5. learn to be efficient and leverage others' results

__What works in this competition.__
1. a good CV strategy (insample iterations)
2. EDA and decide to remove 'ip' from data
2. feature engineeing (next_click)
3. lightgbm algo
4. use last date data to speed up iteration speed

__What works in this competition.__
1. don't trap in local minimum
2. slow iteration in hyper-param tuning
3. slow iteration in feature engineering



## Reference

[IP address encoding issues](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/52374)

[EDA](https://www.kaggle.com/pranav84/talkingdata-eda-to-model-evaluation-lb-0-9683)

[FM_FTRL](https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9752/code)
