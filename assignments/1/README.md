# Assignment 1 Report

## KNN

### Data Visualization

```
df.info() after dropping NaN values:

<class 'pandas.core.frame.DataFrame'>
Index: 113999 entries, 0 to 113999
Data columns (total 21 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   Unnamed: 0        113999 non-null  int64  
 1   track_id          113999 non-null  object 
 2   artists           113999 non-null  object 
 3   album_name        113999 non-null  object 
 4   track_name        113999 non-null  object 
 5   popularity        113999 non-null  int64  
 6   duration_ms       113999 non-null  int64  
 7   explicit          113999 non-null  bool   
 8   danceability      113999 non-null  float64
 9   energy            113999 non-null  float64
 10  key               113999 non-null  int64  
 11  loudness          113999 non-null  float64
 12  mode              113999 non-null  int64  
 13  speechiness       113999 non-null  float64
 14  acousticness      113999 non-null  float64
 15  instrumentalness  113999 non-null  float64
 16  liveness          113999 non-null  float64
 17  valence           113999 non-null  float64
 18  tempo             113999 non-null  float64
 19  time_signature    113999 non-null  int64  
 20  track_genre       113999 non-null  object 
dtypes: bool(1), float64(9), int64(6), object(5)
memory usage: 18.4+ MB
```

1. Popularity

```
Highly skewed distribution
    - large number of instances having "popularity" value close to 0.
    - many items in dataset have popularity 0
Long Tail towards end
Multiple peaks:
    - at around 22 and 45
    - presence of clusters
Significant outliers:
    - large spike at 0
    - could not correlate with the rest of the plot
High kurtosis (indicates heavy tails)
```

2. Duration (ms):

```
Long Tail towards end as compared to start
Sinlge peaks:
    - at around 250000 approx.
    - presence of a single cluster
Almost no outliers
Low kurtosis (indicates light tail)
```

3. Explicit:

```
Imbalanced distribution, high relative height
    Mode: False
    Biased towards Falses
    True instances very low compared to False instaces
    Extremely high variablity
```

4. Danceability

```
Negative skewness
        - tail of the distribution is longer to the left side
    Sinlge peak:
        - at around 250000 approx.
        - presence of a single cluster
    Few outliers near 0.8 from the normal-like distribution
    Range: 0-1
```

5. Energy

```
Linear increasing distribution
    Few outliers near 0, 0.55 and 0.95 from linear-like distribution
    Range: 0-1
```

6. Loudness

```
Negatively skewed distribution
    High kurtosis (heavy tails)
    No outliers
    Range: both positive and negative
    Peak: near -6.000
```

7. Speechiness

```
Positive skewness with sharp dip at beginning
        - tail of the distribution is longer to the right side
    Sinlge peak:
    Outliers: near 0.0 (sharp decrease) and some instances b/w 0.8-1.0
    Range: 0-1
```

8. Acousticness

```
Outlier: near 0.0 and 1.0 (sharp increase)
    Positive skewed with peak at 0.0 and Negative skewed with peak at 1.0
    High Kurtosis (heavy tails)
```

9. Instrumentalness

```
Outlier: 0.0 (heavily outlied from rest) (very high frequency)
    High Kurtosis (heavy tails)
    Two peaks: near 0.0 and near 0.9
    range: 0-1
```

10. Liveness

```
Range: 0-1
    Almost uniform distribution from 0.45
    Two peaks: (less kurtosis (light tails))
        - near 0.1
        - near 0.35
```

11. Valence

```
Range: 0-1
    Outliers: 0.00 (sudden peak), 0.00-0.05 (less freq), 0.05 (sudden peak), 0.96 (sudden peak)
    Inverted-U curve distribution
```

12. tempo

```
Range: 0-250
    Outliers: Too-many small outliers
```

13. time_signature

```
Distribution: Categorical: 0, 1, 2, 3, 4, 5 (labels)
    Non-uniform distributn with outlier 5 (high frequency)
    Frequency of labels 0 and 2 are negligible, while 1, 3 are significant
```

### K-Nearest Neighbours

Hyperparameter-Tuning

```
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k metric
0.218333 0.223391 0.219633   0.22025 0.218333 0.218333  0.218333 15 cosine
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.218333 0.223391 0.219633  0.220250 0.218333 0.218333  0.218333 15    cosine
0.218333 0.221584 0.215688  0.217224 0.218333 0.218333  0.218333 17    cosine
0.217719 0.223267 0.216039  0.218553 0.217719 0.217719  0.217719 13    cosine
0.217193 0.224700 0.217697  0.219864 0.217193 0.217193  0.217193 19 euclidean
0.216667 0.222985 0.215870  0.218321 0.216667 0.216667  0.216667 13 euclidean
0.216228 0.221727 0.217087  0.218131 0.216228 0.216228  0.216228 11 euclidean
0.216140 0.223077 0.217332  0.218778 0.216140 0.216140  0.216140  1    cosine
0.215702 0.223801 0.217386  0.218817 0.215702 0.215702  0.215702 11 manhattan
0.215614 0.221461 0.215571  0.217128 0.215614 0.215614  0.215614 11    cosine
0.215526 0.221954 0.217552  0.218788 0.215526 0.215526  0.215526  7 euclidean
```

Specific observations after Hyperparameter Tuning:

```
1. The performance metrics (accuracy, precision, recall, F1) are all clustered closely together, with little variation across different hyperparameter settings.
2. This suggests that the model's performance is relatively insensitive to the choice of k (ranging from 1 to 19) and the distance metric (cosine, euclidean, manhattan).
3. The cosine distance metric provides the best result, but only marginally better than the euclidean and manhattan metrics
4. This marginal difference suggests that the choice of distance metric has a minimal impact on performance.
5. The low accuracy and F1 scores indicate that the KNN model might not be generalizing well to the data, possibly due to high class imbalance, poor feature selection, or insufficient data quality.

```


Dropping columns: popularity, instrumentalness, valence:

```
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.159123 0.160975 0.158682  0.158527 0.159123 0.159123  0.159123  3 manhattan
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.159123 0.160975 0.158682  0.158527 0.159123 0.159123  0.159123  3 manhattan
0.158772 0.162418 0.159282  0.159651 0.158772 0.158772  0.158772  1 manhattan
0.157719 0.161375 0.158363  0.158769 0.157719 0.157719  0.157719  1 euclidean
0.155088 0.158364 0.153734  0.154725 0.155088 0.155088  0.155088  3 euclidean
0.153070 0.156727 0.153840  0.154356 0.153070 0.153070  0.153070  3    cosine
0.151491 0.156137 0.153547  0.153742 0.151491 0.151491  0.151491  1    cosine
```

```
Observations:
The accuracy is significantly lower compared to the previous tuning result (0.218333) when all features were included. This suggests that these dropped features might have been contributing positively to the model's performance.

```

Combination: 'acousticness', 'energy'
Reason: Extreme negative correlation coefficient

```
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.106228 0.104928 0.106188  0.104861 0.106228 0.106228  0.106228  3 manhattan
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.106228 0.104928 0.106188  0.104861 0.106228 0.106228  0.106228  3 manhattan
0.100702 0.100796 0.100093  0.099764 0.100702 0.100702  0.100702  3    cosine
0.098772 0.098860 0.098826  0.098064 0.098772 0.098772  0.098772  3 euclidean
```

```
The model struggles with these two features, likely because the extreme negative correlation between them makes it difficult for the KNN model to effectively classify the data based on these features alone.
```


Combination: 'speechiness', 'tempo'
Reason: pair-plot

```
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k metric
0.105088 0.104312 0.103943  0.103314 0.105088 0.105088  0.105088  3 cosine
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.105088 0.104312 0.103943  0.103314 0.105088 0.105088  0.105088  3    cosine
0.104123 0.102875 0.104275  0.102647 0.104123 0.104123  0.104123  3 manhattan
0.104123 0.101472 0.104117  0.102021 0.104123 0.104123  0.104123  3 euclidean
```

```
Similar to the previous combination, this pair of features does not seem to provide sufficient information for the KNN model to perform well, leading to poor classification metrics.
```

Combination: 'popularity', 'speechiness'
Reason: pair-plot   

```
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.069912 0.069292 0.069799  0.069109 0.069912 0.069912  0.069912  3 manhattan
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.069912 0.069292 0.069799  0.069109 0.069912 0.069912  0.069912  3 manhattan
0.066930 0.067960 0.067544  0.067465 0.066930 0.066930  0.066930  3    cosine
0.066140 0.066442 0.067129  0.066318 0.066140 0.066140  0.066140  3 euclidean
```

```
This drastic drop in performance suggests that popularity and speechiness are not effective features for distinguishing between classes in this dataset when used in isolation.
```


Combination: 'speechiness', 'instrumentalness'
Reason: pair-plot

```
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.070614 0.071262 0.071313  0.070377 0.070614 0.070614  0.070614  3 manhattan
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.070614 0.071262 0.071313  0.070377 0.070614 0.070614  0.070614  3 manhattan
0.067456 0.066836 0.067268  0.066485 0.067456 0.067456  0.067456  3    cosine
0.065965 0.066283 0.065664  0.065467 0.065965 0.065965  0.065965  3 euclidean
```

```
This pair of features also fails to provide enough discriminatory power for the KNN model.
```


### SECOND DATASET:

```
Observations:

The accuracy of KNN model trained on second dataset is slightly lower than the first dataset.
Although, it is not significantly lower, but this shows that the first dataset is slightly better for predicting genre than second dataset.
```

## Linear Regression

```
Checking for best learning rate for Polynomial Regression

    Best lr: 0.1
    Minimum MSE: 0.04868740596031792

    Best learning rate in case of Polynomial Regression is higher than simple linear regression
```

```
Finding best k (degree) value in range 1 to 20 which minimizes MSE on test set

Best k: 15

Minimum MSE: 0.008329993315670487
```

```
Making GIF

Reference: https://www.kaggle.com/discussions/general/501746
```

```
Few point(s) about random seed vs no seed:
 - Potential of overfitting: slightly lower validation MSE with seed initialization might suggest a marginally better fit (although almost similar)
```

### Regularization

ChatGPT:

```
General form of the cost function in Linear regression:

J(θ) = (1/2m) * Σ(h_θ(x_i) - y_i)^2

Where:
1. J(θ) is the cost function
2. m is the number of training examples
3. h_θ(x_i) is the hypothesis (predicted value) for the i-th example
4. y_i is the actual value for the i-th example

Based on the type of regularization, we update thist cost function.

1. No Regularization:
    Cost function: J(θ) = (1/2m) * Σ(h_θ(x_i) - y_i)^2
    Gradient: ∇J(θ) = (1/m) * X^T * (h_θ(X) - y)
2. L2 Regularization (Ridge):
    Cost function: J(θ) = (1/2m) * Σ(h_θ(x_i) - y_i)^2 + (λ/2m) * Σθ_j^2
    Gradient: ∇J(θ) = (1/m) * X^T * (h_θ(X) - y) + (λ/m) * θ
3. L1 Regularization (Lasso):
    Cost function: J(θ) = (1/2m) * Σ(h_θ(x_i) - y_i)^2 + (λ/m) * Σ|θ_j|
    Gradient: ∇J(θ) = (1/m) * X^T * (h_θ(X) - y) + (λ/m) * sign(θ)

The effects of these regularizations:

L2 Regularization (Ridge):

1. Adds a penalty term proportional to the square of the magnitude of coefficients.
2. Encourages the weights to be small, but doesn't make them exactly zero.
3. Useful when you want to prevent overfitting, but don't want to eliminate any features entirely.


L1 Regularization (Lasso):

1. Adds a penalty term proportional to the absolute value of the magnitude of coefficients.
2. Encourages sparsity: it tends to make some of the feature weights exactly zero.
3. Useful for feature selection, as it can completely eliminate the least important features.
```

Without Regularization:

```
 k  train_mse  valid_mse  test_mse  train_var  valid_var  test_var  train_sd  valid_sd  test_sd
 1   0.222916   0.183023  0.147485   0.029189   0.024691  0.029899  0.170849  0.157133 0.172912
 2   0.019772   0.015982  0.013570   0.226130   0.177977  0.201452  0.475531  0.421873 0.448834
 3   0.018170   0.013046  0.012814   0.228396   0.164853  0.196296  0.477908  0.406021 0.443053
 4   0.011403   0.011676  0.007601   0.239508   0.157659  0.185130  0.489395  0.397063 0.430267
 5   0.011413   0.011513  0.007609   0.238081   0.154217  0.182733  0.487935  0.392704 0.427473
 6   0.010680   0.013562  0.007107   0.238233   0.148572  0.172233  0.488091  0.385451 0.415010
 7   0.010534   0.013771  0.007040   0.238212   0.151317  0.173917  0.488069  0.388995 0.417034
 8   0.010611   0.015011  0.007198   0.237463   0.149908  0.169394  0.487302  0.387179 0.411576
 9   0.010515   0.015174  0.007050   0.238267   0.153910  0.172431  0.488126  0.392313 0.415248
10   0.010550   0.015458  0.007071   0.238013   0.154311  0.171743  0.487866  0.392825 0.414419
11   0.010528   0.015570  0.006964   0.238877   0.157507  0.174542  0.488751  0.396871 0.417782
12   0.010517   0.015423  0.006941   0.239004   0.158080  0.175394  0.488880  0.397593 0.418801
13   0.010556   0.015527  0.006938   0.239578   0.159857  0.177226  0.489467  0.399822 0.420982
14   0.010547   0.015292  0.006959   0.239751   0.159902  0.178145  0.489643  0.399877 0.422072
15   0.010613   0.015417  0.007044   0.239946   0.160394  0.178859  0.489843  0.400492 0.422917
16   0.010612   0.015244  0.007111   0.239978   0.159893  0.179251  0.489876  0.399866 0.423380
17   0.010672   0.015398  0.007235   0.239866   0.159575  0.179086  0.489761  0.399469 0.423186
18   0.010669   0.015324  0.007316   0.239747   0.158832  0.178952  0.489640  0.398537 0.423027
19   0.010706   0.015491  0.007431   0.239471   0.158217  0.178311  0.489358  0.397765 0.422269
20   0.010695   0.015494  0.007493   0.239285   0.157557  0.177904  0.489168  0.396935 0.421786
21   0.010707   0.015645  0.007568   0.238988   0.157019  0.177166  0.488864  0.396256 0.420911
22   0.010690   0.015683  0.007592   0.238827   0.156643  0.176755  0.488700  0.395781 0.420422
23   0.010684   0.015793  0.007615   0.238609   0.156368  0.176185  0.488476  0.395434 0.419744
24   0.010666   0.015831  0.007600   0.238532   0.156329  0.175948  0.488397  0.395385 0.419462
25   0.010652   0.015886  0.007579   0.238434   0.156348  0.175673  0.488297  0.395408 0.419134
26   0.010639   0.015903  0.007538   0.238456   0.156586  0.175682  0.488319  0.395710 0.419144
27   0.010627   0.015905  0.007494   0.238473   0.156823  0.175712  0.488337  0.396009 0.419180
28   0.010622   0.015895  0.007448   0.238573   0.157216  0.175942  0.488440  0.396505 0.419455
29   0.010617   0.015857  0.007404   0.238670   0.157547  0.176203  0.488538  0.396922 0.419766
```

```
Observations:

1. As k increases, `train_mse` decreases almost monotonically.
2. However, the validation loss increases.
3. This shows Overfitting since model is learning more and more noises in training set. Hence, it's performance on validation set is degrading.
```

With L1 Regularisation:

```
 k  train_mse  valid_mse  test_mse  train_var  valid_var  test_var  train_sd  valid_sd  test_sd
 1   0.222929   0.182337  0.147448   0.027980   0.023668  0.028660  0.167273  0.153845 0.169294
 2   0.019931   0.015871  0.013189   0.221258   0.173379  0.197487  0.470381  0.416389 0.444395
 3   0.018344   0.013256  0.012247   0.220826   0.161055  0.191393  0.469921  0.401317 0.437484
 4   0.011309   0.011730  0.007385   0.233836   0.154250  0.182029  0.483566  0.392747 0.426648
 5   0.011492   0.011490  0.007413   0.231372   0.150799  0.179395  0.481011  0.388328 0.423551
 6   0.011035   0.012814  0.007171   0.232048   0.141388  0.167959  0.481714  0.376016 0.409827
 7   0.010953   0.012772  0.007119   0.232239   0.142799  0.169110  0.481912  0.377888 0.411230
 8   0.011057   0.013479  0.007293   0.230316   0.139766  0.164320  0.479912  0.373853 0.405364
 9   0.010982   0.013413  0.007198   0.230863   0.141595  0.165927  0.480482  0.376291 0.407342
10   0.010976   0.013253  0.007171   0.230142   0.141903  0.166354  0.479731  0.376700 0.407865
11   0.010962   0.013203  0.007127   0.230635   0.142978  0.167470  0.480245  0.378125 0.409231
12   0.010964   0.013081  0.007121   0.230758   0.143098  0.167907  0.480373  0.378284 0.409764
13   0.010972   0.012988  0.007124   0.231106   0.143688  0.168956  0.480735  0.379062 0.411042
14   0.010976   0.012960  0.007126   0.231078   0.143595  0.168955  0.480706  0.378940 0.411042
15   0.010996   0.012850  0.007186   0.231232   0.143669  0.169816  0.480865  0.379037 0.412088
16   0.011000   0.012872  0.007202   0.231244   0.143262  0.169441  0.480878  0.378500 0.411633
17   0.011007   0.012806  0.007275   0.231106   0.143381  0.170188  0.480735  0.378656 0.412538
18   0.011005   0.012824  0.007297   0.231060   0.142876  0.169790  0.480687  0.377989 0.412055
19   0.011013   0.012812  0.007369   0.230893   0.142806  0.170102  0.480513  0.377897 0.412434
20   0.011013   0.012830  0.007394   0.230718   0.142175  0.169583  0.480332  0.377061 0.411804
21   0.011016   0.012859  0.007448   0.230303   0.142041  0.169567  0.479899  0.376884 0.411785
22   0.011012   0.012872  0.007468   0.230086   0.141563  0.169177  0.479673  0.376248 0.411312
23   0.011012   0.012898  0.007514   0.230093   0.141508  0.169200  0.479680  0.376175 0.411339
24   0.011004   0.012892  0.007526   0.230047   0.141300  0.169096  0.479632  0.375899 0.411212
25   0.011001   0.012917  0.007553   0.230021   0.141216  0.169008  0.479605  0.375788 0.411106
26   0.010992   0.012909  0.007556   0.230000   0.141129  0.168979  0.479583  0.375672 0.411071
27   0.010989   0.012923  0.007571   0.229976   0.141068  0.168909  0.479558  0.375591 0.410985
28   0.010983   0.012910  0.007570   0.229925   0.141029  0.168915  0.479505  0.375538 0.410993
29   0.010980   0.012918  0.007577   0.229879   0.140983  0.168853  0.479456  0.375477 0.410918
```

```
Observations:

1. As k increases, `train_mse` doesn't decrease that drastically as earlier with no regularization.
2. Also, `valid_loss` doesn't increase that drastically here in L1 regularization.
3. This shows that penalizing Loss function with L1 norm, helps model to overcome Overfitting issue.
```

With L2 Regularization:

```
 k  train_mse  valid_mse  test_mse  train_var  valid_var  test_var  train_sd  valid_sd  test_sd
 1   0.222917   0.182821  0.147472   0.028834   0.024391  0.029535  0.169807  0.156175 0.171858
 2   0.019989   0.015691  0.012927   0.218062   0.171170  0.194487  0.466971  0.413727 0.441007
 3   0.018368   0.013018  0.012286   0.220858   0.158748  0.189965  0.469956  0.398432 0.435849
 4   0.011474   0.011555  0.007464   0.234849   0.153909  0.181622  0.484612  0.392313 0.426172
 5   0.011531   0.011391  0.007508   0.233683   0.150147  0.179153  0.483407  0.387489 0.423265
 6   0.010808   0.013399  0.007118   0.234877   0.144973  0.169325  0.484642  0.380754 0.411491
 7   0.010659   0.013539  0.007042   0.234791   0.147185  0.170668  0.484552  0.383647 0.413119
 8   0.010754   0.014802  0.007250   0.234372   0.145719  0.166103  0.484119  0.381731 0.407558
 9   0.010629   0.014898  0.007081   0.235040   0.149354  0.168860  0.484809  0.386464 0.410926
10   0.010670   0.015220  0.007116   0.234836   0.149623  0.167955  0.484599  0.386811 0.409823
11   0.010621   0.015284  0.006989   0.235609   0.152674  0.170630  0.485396  0.390735 0.413074
12   0.010607   0.015167  0.006959   0.235708   0.153185  0.171314  0.485497  0.391388 0.413901
13   0.010629   0.015244  0.006944   0.236259   0.154992  0.173176  0.486065  0.393690 0.416144
14   0.010616   0.015029  0.006956   0.236412   0.155061  0.174043  0.486222  0.393778 0.417185
15   0.010678   0.015142  0.007040   0.236642   0.155676  0.174892  0.486458  0.394558 0.418201
16   0.010678   0.014980  0.007104   0.236683   0.155247  0.175330  0.486501  0.394014 0.418724
17   0.010743   0.015128  0.007238   0.236636   0.155070  0.175342  0.486452  0.393790 0.418738
18   0.010745   0.015056  0.007323   0.236546   0.154396  0.175299  0.486360  0.392933 0.418687
19   0.010791   0.015219  0.007454   0.236338   0.153890  0.174821  0.486146  0.392288 0.418115
20   0.010786   0.015218  0.007526   0.236183   0.153265  0.174500  0.485987  0.391491 0.417732
21   0.010807   0.015367  0.007617   0.235940   0.152781  0.173878  0.485736  0.390873 0.416987
22   0.010794   0.015400  0.007652   0.235799   0.152397  0.173520  0.485591  0.390380 0.416557
23   0.010793   0.015508  0.007689   0.235610   0.152126  0.173006  0.485397  0.390033 0.415940
24   0.010776   0.015540  0.007683   0.235537   0.152044  0.172781  0.485321  0.389928 0.415669
25   0.010764   0.015596  0.007673   0.235445   0.152029  0.172510  0.485227  0.389909 0.415343
26   0.010749   0.015608  0.007637   0.235456   0.152207  0.172494  0.485238  0.390137 0.415324
27   0.010735   0.015612  0.007598   0.235463   0.152396  0.172491  0.485246  0.390379 0.415320
28   0.010726   0.015600  0.007553   0.235545   0.152729  0.172677  0.485330  0.390806 0.415544
29   0.010718   0.015565  0.007510   0.235623   0.153016  0.172889  0.485410  0.391172 0.415799
30   0.010717   0.015534  0.007475   0.235742   0.153382  0.173209  0.485533  0.391640 0.416184
```

```
Observations:

1. As k increases, `train_mse` doesn't decrease that drastically as earlier with no regularization.
2. Also, `valid_loss` doesn't increase that drastically here in L1 regularization.
3. This shows that penalizing Loss function with L2 norm , also, helps model to overcome Overfitting issue. But this doesn't work as well as L1 regularization.
4. This could be becoz we only have one feature, and L1 penalty is proportional to absolute value of the magnitude of coefficients.
```














