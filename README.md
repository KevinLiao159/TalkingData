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

Big thanks to sponsor *TalkingData* and *Kaggle* for providing such an interesting competition. Congradulations to those top teams and appreciations of kernal contributions from @Pranav Pandya and @anttip. This is my first Kaggle competition and I can't tell you how much fun for me being a part of it. I had a fulltime job and I knew that I can only commit my weekend free time to the competition. As a newby on kaggle, I did not anticipate a good LB score at all before going into the competition. Just one week right before the final submission deadline, I was so pumped up that I got myself solo ranked **top 3%** in *public LB*. However, that didn't last long, and my final submission is ranked **top 15%** in *private LB*, which I think it is a reasonable rank for me. Overall, I think this is one of the most competitive competition and it's very hard to get in **top 5%** without a team.

My results are shown below, I won't share too much about my strategy because it's not a winning strategy anyway and most of my stuff is taken from public kernels. However, I will share what I have learned and what makes a winning strategy.

## My Model and LB score (AUC-ROC)
model definition can be found in [scripts/train_lightgbm-v3.py](https://github.com/KevinLiao159/TalkingData/blob/master/scripts/train_lightgbm-v3.py)

feature engineering can be found in [scripts/feature_eng-v3.py](https://github.com/KevinLiao159/TalkingData/blob/master/scripts/feature_eng-v3.py)

  - **model** LGBM with 42 (36 numerical, 6 categorical) features.

|model|private score|public score|final rank| 
|---|---|---|---|
| model V3 |0.9806721|0.9811112| 585th (*top15%*)|


## What I have learned from kaggle competition winners?

__We have to understand the game before wasting time.__

In this compeition, the data set is huge but we only have six features. This means that 1). we need to spend a lot of time in feature engineering 2). feature engineering and model validation cycle would take long time (because data is huge). Unless we have a good team, time resource allocation is crucial in this particular competition. A suggested time table will be like following [6th place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56283):

* 80% feature engineering
* 10% making local validation as fast as possible
* 5% hyper parameter tuning
* 5% ensembling

__Establishing a high speed research cycle is the key to win__

This competition is about training model in past historical data and predicting future fraudulant clicks (which is a big-time imbalanced classification). For imbalanced future classification problem, using tradititonal five-fold cross-validation may not be a good strategy (or you have to be really careful about sampling ratio, the timing and future information leakage).

1. Basic strategy: a good practice research framework for this kind would be like following [6th place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56283):

    * Understanding that training data starts from day 7 and ends at day 9. Testing data is day 10, in hours of 4, 5, 9, 10, 13, 14.

    * Introducing a insample hold-out set bright line. So we can enforce a bright line between day 8 and day 9 for insample research cycle.

    * Training on day <= 8, and validating on both day 9 - hour 4 (mirror public LP), and day-9, hours 5, 9, 10, 13, 14 (mirror private LP).

    * For out-of-sample (public LB score) iteration, we retrain on all data using 1.2 times the number of trees found by early stopping in insample validation


2. Advanced strategy: a fast run-time and light weight memory usage iteration would be [1st place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475):

    * Understanding that there are 99.85% of negative examples in the data and dropping out tons of negative example DOES NOT deteriorate out-of-sample performance.

    * Using negative down-sampling strategy, which means that we use all positive examples (i.e., is_attributed == 1) and down-sampled negative examples on model training. We down-sampled negative examples such that their size becomes equal to the number of positive ones. It discards about 99.8% of negative examples.

    * Using sample bagging technique, which means we bag five predictors trained on five sampled datasets created from different random seeds.

    * This technique allows us to use hundreds of features while keeping LGB training time less than 30 minutes.

    * Or use [memory trick](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56105) in numpy 


3. Good principle: keep your insample hold-out score align with LB score:

    * Do not rely solely on either pulic LB score or insample hold-out score. If you do that, you will end up overfitting to one of them eventually 

    * Discard features that increase the gap between pulic LB score and insample hold-out score even though it increases your insample hold-out score


__Feature engineering the winning secret sauce__

We have five original categorical features and one timestamp feature in the data set. Unless you have some crazy NN models with proper data preprocessing ([3rd place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262)), you definitely need some magic features to separate youself from the crowd. If you have no idea about how to engineer some new features, please see this [good feature engineering guidance](https://github.com/h2oai/h2o-meetups/blob/master/2017_11_29_Feature_Engineering/Feature%20Engineering.pdf).

Here are some general ideas taken from top winners:

* dropping original worse-than-noise features [*ip*, maybe *device*]

* encode timestamp into day and hour

* user concepts: ip, device, os triplets

* (require brute-force) aggregates on various feature groups (click series-based feature sets (i.e., each feature set consists of 31 (=(2^5) - 1) features)) [1st place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475)
    * count features, unique count features, cumcount features
    * time delta with previous value, delta with next value
    * mean and variance with respect to hour
    * standard target encoding
    * [Weights of Evidence target encoding](https://github.com/h2oai/h2o-meetups/blob/master/2017_11_29_Feature_Engineering/Feature%20Engineering.pdf)

* ratios features
    * number of clicks per ip, app to number of click per app
    * nunique_counts_ratio
    * top_counts_ratio

* magic additions:
    * feature extraction (topic models): categorical feature embedding by using LDA/NMF/LSA [1st place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56475)

    * matrix factorization: truncated svd from sklearn and FM-like embedding

* [data leakage in test set](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56268) 


__Appropriate models for categorical features with large data__

* LightGBM is crowned over XGBoost in this competition in terms of memory usage and run-time optimization

* Do NOT spend too much time on hyper-param tuning (not too much juice from hyper-params)

* Some NN models for me to learn 
    * [2nd place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56328)

    * [3rd place solution](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/56262)

    * [4th place solution](https://github.com/CuteChibiko/TalkingData/blob/master/model.png)

    * [libFM in Keras](https://www.ibm.com/developerworks/community/blogs/jfp/entry/Implementing_Libfm_in_Keras?lang=en_us)


__Extra slight boost from ensembling__

* most people ensemble their predictions based on LB score

* good practice in blending - average the logit of the predictions (aka raw predictions)

* [restacking](https://github.com/kaz-Anova/StackNet#restacking-mode) barely helps in this competition


## Some baseline benchmark from my observations (this is meant for ranking roughly estimates)

1. To be in *top 30%*, use solely LightGBM and trained (without too much tuning) it on some good features from public kernels 

2. To be in *top 20%*, use solely LightGBM and trained (with some proper tuning) it on at least top 40 features from public kernels (must include time delta, count, unique count types aggregates with various feature groups)

3. To be in *top 10%*, must have beast machine and train models with minimum of 100 proven-to-be-useful features or use NN models based on 20+ aggregate level features 

4. To be in *top 5%*, all above + feature extraction (categorical feature embedding) or FM-like algos

5. To be in *top 1%*, this is really hard. Not sure how to do it.


## Reference

[1][IP address encoding issues](https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/discussion/52374)

[2][EDA by @Pranav Pandya](https://www.kaggle.com/pranav84/talkingdata-eda-to-model-evaluation-lb-0-9683)

[3][FM_FTRL by @anttip](https://www.kaggle.com/anttip/talkingdata-wordbatch-fm-ftrl-lb-0-9752/code)

[4][Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)

[5][Ad Click Prediction: a View from the Trenches](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/41159.pdf)
