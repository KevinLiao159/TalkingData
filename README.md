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

Big thanks to sponsor *TalkingData* and *Kaggle* for providing such an interesting competition. Congradulations to those top teams and appreciations of kernal contributions from @Pranav Pandya and @anttip. This is my first Kaggle competition and I can tell you that I had so much fun being a part of it. I had a fulltime job and I knew that I can only commit my weekend free time to the competition. As a newby on kaggle, I did not anticipate a good LB score at all before going into the competition. Just one week right before the final submission deadline, I was so pumped up that I got myself solo ranked **top 3%** in *public LB*. However, that didn't last long, and my final submission is ranked **top 15%** in *private LB*, which I think it is a reasonable rank for me. Overall, I think this is one of the most competitive competition and it's very hard to get in **top 5%** without a team.

My results are shown below, I won't share too much my strategy because it's not a winning strategy anyway and most of my stuff is taken from public kernals. However, I will share what I have learned and what makes winning strategies.

## My Model and LB score (AUC-ROC)
model definition can be found in [scripts/train_lightgbm-v3.py](https://github.com/KevinLiao159/TalkingData/blob/master/scripts/train_lightgbm-v3.py)

feature engineering can be found in [scripts/feature_eng-v3.py](https://github.com/KevinLiao159/TalkingData/blob/master/scripts/feature_eng-v3.py)

  - **model1** LGBM with 42 (36 numerical, 6 categorical) features.

|model|private score|public score|
|---|---|---|
| model V3 |0.9806721|0.9811112|


## What I have learned kaggle competition winners?

__We have to understand the game before wasting time.__

In this compeition, the data set is huge but we only have six features. This means that 1). we need a lot of time in feature engineering 2). feature engineering and model validation cycle would take long time (because data is huge). Unless we have a good team, time resource allocation is crucial in this particular competition. A suggested time table will be like following:

* 80% feature engineering
* 10% making local validation as fast as possible
* 5% hyper parameter tuning
* 5% ensembling

__Establishing a high speed research cycle is the key to win.__

This competition is about training model in past historical data and predicting future fraudulant clicks. For future classification problem, using tradititonal five-fold cross-validation may not be a good strategy (or you have to be really careful about the timing and future information leakage).

1. So a good practice research framework for this kind would be like following:

    * Understanding that training data starts from day 7 and ends at day 9. Testing data is day 10, in hours of 4, 5, 9, 10, 13, 14.

    * Introducing a insample hold-out set bright line. So we can enforce a bright line between day 8 and day 9 for insample research cycle.

    * Training on day <= 8, and validating on both day 9 - hour 4 (mirror public LP), and day-9, hours 5, 9, 10, 13, 14 (mirror private LP).

    * For out-of-sample (public LB score) iteration, we retrain on all data using 1.2 times the number of trees found by early stopping in insample validation

2. 



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



learn that what makes a winning strategy
__Cross Validation strat for future prediction.__

__Feature engineering strategy.__
1. for categorical -- factors, latent signlas, 
2. brute force
3. feature selection
4. kaggle peers post

__Run-time memory management strategy.__
1. drop

__negative down sample strategy.__
1. model performance check

__modeling strategy.__
1. lightgbm
2. bagging



## Reference

[IP address encoding issues](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/52374)

[EDA](https://www.kaggle.com/pranav84/talkingdata-eda-to-model-evaluation-lb-0-9683)

[FM_FTRL](https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9752/code)
