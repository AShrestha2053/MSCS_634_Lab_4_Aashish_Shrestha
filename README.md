# MSCS 634 - Lab 4: Regression Analysis and Regularization

In this lab I work with regression. Regression is used to predict a number, not a group. I use the
Diabetes dataset from scikit-learn. It has 442 patients and 10 health measurements for each one. The
target is a number that shows how much the disease moved forward after one year. As part of this Lab 4 to build
different regression models, measure them with the same scores, and see which one works best.

## What I did

I built the models step by step:

- **Simple Linear Regression** using only one feature (bmi).
- **Multiple Regression** using all 10 features.
- **Polynomial Regression** to show how a high degree causes overfitting.
- **Ridge and Lasso Regression** that add a penalty to control overfitting.

I checked every model with four scores: MAE, MSE, RMSE (lower is better) and R2 (higher is better,
from 0 to 1).

## Files in this repo

- `MSCS_634_Lab_4.ipynb` - the notebook with all the code, the scores, the plots, and my analysis.
- `README.md` - this file.

## How to run

```bash
pip install scikit-learn pandas numpy matplotlib jupyter
jupyter notebook MSCS_634_Lab_4.ipynb
```

Run the cells from top to bottom. I used a fixed random seed (42), so the scores are the same every
time you run it.

## The scores I got

| Model                    | MAE   | RMSE  | R2     |
| ------------------------ | ----- | ----- | ------ |
| Simple Linear (bmi only) | 52.26 | 63.73 | 0.2334 |
| Multiple Linear (all 10) | 42.79 | 53.85 | 0.4526 |
| Polynomial (degree 2)    | 44.82 | 54.71 | 0.4156 |
| Ridge (alpha = 0.1)      | 43.00 | 53.45 | 0.4609 |
| Lasso (alpha = 0.1)      | 42.85 | 52.90 | 0.4719 |

## What I learned

**More features helped a lot.** The model with only bmi was weak (R2 about 0.23). When I used all 10
features, the R2 almost doubled to about 0.45 and the RMSE dropped from about 64 to 54. So one
feature alone can not explain the disease score, but all of them together do much better.

**High-degree polynomials overfit badly.** This was the clearest lesson. At degree 2 the test score
already started to fall. At degree 3 and 4 the model fit the training data almost perfectly (degree
4 reached train R2 = 1.0), but on the test data the R2 became very negative (-5 and then -27). This
means the model only memorized the training points and was useless on new data. The number of
features also grew from 10 to 1000 at degree 4. So a bigger degree is not better after some point.

**Regularization gave the best model.** Ridge and Lasso both improved the result with a small alpha.
Lasso with alpha = 0.1 was the best model of all (R2 about 0.47), and it did this while keeping only
7 features out of 10. So Lasso removed 3 weak features by itself and still got a better score. When
alpha was too big, the penalty was too strong and the score dropped, and for Lasso with alpha = 10
every weight became zero.

**About the dataset.** The Diabetes data is hard to predict. Even the best model explained only
about 47% of the target, so the disease score depends on many things that these 10 measurements do
not fully cover. The two most useful features are bmi and s5.

## Decisions I made and problems I faced

**I did not standardize the data.** This dataset is already scaled by scikit-learn. Every feature
has a mean near 0 and small values, so scaling again was not needed. I checked this in the notebook.

**No cleaning was needed.** I checked for missing values and there were none, so there was no data
cleaning to do. The main work was choosing the right model, not fixing the data.

**I chose bmi for the simple model.** For the one-feature model I had to pick a feature. I looked at
the correlation of each feature with the target and bmi was the strongest, so I used it.

**I tested many alpha values.** For Ridge and Lasso the result depends a lot on alpha. So I tried
several values (from very small to large) and showed the full trend, instead of using only one
value. This made it clear that a small alpha works best for this data.

**I fixed the random seed.** The train/test split uses a random part, so I set the seed to 42. This
makes the scores stable and the same in every run.
