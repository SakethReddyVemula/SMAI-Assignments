# Assignment-2 Report

Saketh Reddy Vemula | 2022114014 | saketh.vemula@research.iiit.ac.in | @SakethReddyVemula

## 3. K-Means Clustering

### 3.2 Determine the Optimal Number of Clusters for 512 dimensions

Elbow Plots:
![Elbow plot for kmeans](./figures/WCSS_vs_k_1to200.png?raw=true"Title")

![Elbow plot for kmeans](./figures/WCSS_vs_k_1to30.png?raw=true"Title")

Based on the Elbow method applied to this k vs WCSS (Within-Cluster Sum of Squares) plot:

1. The graph shows a steep decrease in WCSS as k increases from 1 to about 7.

2. After k=7, the rate of decrease in WCSS slows down significantly, forming an "elbow" shape.

3. The elbow point represents the optimal trade-off between cluster compactness and number of clusters.

4. At k=7, we see the last significant drop before the curve starts to level off.

5. Beyond k=7, adding more clusters provides diminishing returns in reducing WCSS.

Therefore, using the Elbow method, `kkmeans = 7` is identified as the optimal number of clusters, balancing cluster quality with model complexity.

K-Means clustering using kkmeans:

![kkmeans = 7](./figures/3_2.png?raw=true"kkmeans=7_plot")

Observations:

`kkmeans1 = 7`

## 4 Gaussian Mixture Models

### 4.2.1
Yes, the GMM class works well. Likelihood of both sklearn and my class decreases and converges
This can be confirmed using AIC, BIC scores also by comparing two methods.

"Lower the AIC and BIC scores are better the clustering"

The AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) plots for both the custom GMM implementation and sklearn's GMM show interesting characteristics:

1. Linear trend:
   Both implementations show a nearly linear increase in AIC and BIC scores as the number of clusters increases. This is unusual, as typically these scores would show an elbow or minimum point.

2. No clear optimal point:
   Neither plot shows a clear minimum or elbow point, which would normally indicate the optimal number of clusters.

3. Consistent behavior:
   The behavior is consistent across both custom and sklearn implementations, suggesting this is a characteristic of the data rather than an implementation issue.

4. High-dimensional data effect:
   The linear trend is likely due to the high dimensionality (512 dimensions) of the dataset. In high-dimensional spaces, the curse of dimensionality can make it difficult for clustering algorithms to find meaningful structures.

5. Overfitting potential:
   The continuous decrease in scores suggests that adding more clusters always improves the model fit, potentially leading to overfitting.

6. Scale differences:
   The custom implementation shows slightly higher AIC/BIC values compared to sklearn's, but the overall trend is the same.

7. Model complexity vs. data fit:
   The linear increase indicates that the improvement in data fit (log-likelihood) is consistently outweighing the penalty for model complexity as clusters are added.

8. Lack of natural clusters:
   This behavior suggests that the data might not have well-defined, natural clusters in the 512-dimensional space.

These results indicate that traditional methods for determining the optimal number of clusters (like AIC/BIC) may not be reliable for this high-dimensional dataset. Alternative approaches, such as dimensionality reduction before clustering or using other cluster quality metrics, might be more appropriate for this data.

Therefore, optimal number of clusters for the 512-dimensional dataset is `kgmm1 = 1`

## Dimensionality Reduction and Visualization

## 5.2 Data Analysis

```
X.shape:  (200, 512)
self.components.shape:  (512, 2)
transformed data shape:  (200, 2)
Shape check: True
Original shape: (200, 512), Transformed shape: (200, 2)
Variance check: True
Original variance: 22.639683183348104, Transformed variance: 2.99956203592834
Explained variance ratio: [0.09169602 0.04079532]
Sum of explained variance ratio: 0.132491343259369
True
X.shape:  (200, 512)
self.components.shape:  (512, 2)
transformed data shape:  (200, 2)
X.shape:  (200, 512)
self.components.shape:  (512, 3)
transformed data shape:  (200, 3)
Shape check: True
Original shape: (200, 512), Transformed shape: (200, 3)
Variance check: True
Original variance: 22.639683183348104, Transformed variance: 3.7334539937306666
Explained variance ratio: [0.09169602 0.04079532 0.03241618]
Sum of explained variance ratio: 0.16490751939835827
True
X.shape:  (200, 512)
self.components.shape:  (512, 3)
transformed data shape:  (200, 3)
Total explained variance (2D): 0.1325
Total explained variance (3D): 0.1649
```

Upon examining:

For 2D-PCA:

1. Axis 1 (PC1): This axis could represent `abstract vs concrete concepts`. Words: love, happy, angry are abstract, while words: table, pencil, car... are concrete.
2. Axis 2 (PC2): `animate vs inanimate concepts`. Words: deer, bear, human -> animate; Words: laptop, chair, door -> inanimate words.

For 3D-PCA:

1. Axis 1 (PC1): This axis could represent `abstract vs concrete concepts`. Words: love, happy, angry are abstract, while words: table, pencil, car... are concrete.
2. Axis 2 (PC2): `animate vs inanimate concepts`. Words: deer, bear, human -> animate; Words: laptop, chair, door -> inanimate words.
3. Axis 3 (PC3): may represent `physical vs non-physical entities`. Words such as happy, sad, angry -> non-physical, whereas words such as car, tree, fish are physical objects.

![2D](./figures/5_2_3_words_labeled.jpeg?raw=true"Title")

![3D](./figures/5_2_3_words_labeled_3d.jpeg?raw=true"Title")

approximately we can estimate 4-5 clusters. To be safe 5 clusters.

Therefore,
`k2 = 5`


# 6 PCA + Clustering

## 6.1 K-means Clustering Based on 2D Visualization

![6.1.1](./figures/6_1_1.png?raw=true"Title")

1.  K-means Clustering Results:
    Number of Clusters: 5
    Cost (WCSS): 3977.2409
    Silhouette Score: 0.0364
    Davies Bouldin Score: 4.3173
    Fit time: 0.1435 seconds
    Predict time: 0.0014 seconds
    Cluster sizes:
        Cluster 1: 29 points
        Cluster 2: 39 points
        Cluster 3: 29 points
        Cluster 4: 56 points
        Cluster 5: 47 points

## 6.2 PCA + K-Means Clustering

![6.2.1](./figures/6_2_1.png?raw=true"Title")

1. Clearly, we see that the optimal number of dimensions for reduction is `k = 5`. `Reduced dataset has 5 components`
The choice of 5 as the optimal number of dimensions for reduction in this scree plot can be justified as follows:

2. Elbow point: There's a clear "elbow" or bend in the curve around 5 components, where the rate of decrease in eigenvalues slows down significantly.

3. Explained variance: The first 5 components likely explain a substantial portion of the total variance in the data.

4. Balance: 5 components strike a balance between dimensionality reduction and retaining important information.

5. Computational efficiency: Reducing to 5 dimensions significantly decreases computational complexity compared to the original high-dimensional space.

This choice aims to capture the most important features of the data while effectively reducing noise and computational complexity.

![6.2.2](./figures/6_2_2.png?raw=true"Title")

2. Optimal number of clusters: `kkmeans3 = 6`

K-means Clustering Results:
Number of Clusters: 6
Cost (WCSS): 352.5643
Silhouette Score: 0.2725
Davies Bouldin Score: 1.1381
Fit time: 0.0995 seconds
Predict time: 0.0002 seconds
Cluster sizes:
	Cluster 1: 28 points
	Cluster 2: 27 points
	Cluster 3: 25 points
	Cluster 4: 57 points
	Cluster 5: 34 points
	Cluster 6: 29 points

![6.2.3](./figures/6_2_3.png?raw=true"Title")

Observations:

Much less WCSS as compared to non-PCA dataset

## 6.3 GMM Clustering Based on 2D Visualization

![6.3.1](./figures/6_3_1.png?raw=true"Title")

Keeping the model simple, lets keep `kgmm3 = 3`

## 6.4 PCA + GMM

![6.4.1](./figures/6_4_1.png?raw=true"Title")

![6.4.2](./figures/6_4_1_sklearn.png?raw=true"Title")

Custom GMM (Plot 1):

- `Stability`: The AIC and BIC scores show more fluctuations or "noise" compared to the sklearn GMM, indicating a less smooth convergence. Both curves fluctuate more heavily, particularly as the number of components increases beyond ~50.
- `Lowest AIC and BIC`: The AIC reaches its lowest value at 89 components, while BIC reaches its lowest at 3 components. The discrepancy between the lowest AIC and BIC components suggests the model's uncertainty in balancing complexity and fit.
- `AIC and BIC Behavior`: 
  - AIC initially decreases before starting to rise sharply after ~75 components.
  - BIC shows a more gradual decrease, then stabilizes after 50 components, before it starts increasing more sharply.

Sklearn GMM (Plot 2)

- `Stability`: The sklearn version shows a much smoother curve for both AIC and BIC, with far fewer fluctuations. The scores change more gradually, which indicates better convergence behavior.
- `Lowest AIC and BIC`: The lowest AIC occurs at 122 components, while BIC, similar to the custom GMM, reaches its minimum at 3 components. However, in this case, the curves leading to these points are smoother, indicating more consistent model performance.
- `AIC and BIC Behavior`:
  - AIC follows a more gradual decline and reaches a stable point before starting to rise again after 122 components.
  - BIC decreases until about 3 components and then steadily increases without the high degree of noise seen in the custom model.

Comparison

1. `Fluctuations`: The custom GMM shows far more fluctuations and noise in both AIC and BIC, especially at higher numbers of components. This suggests the custom implementation might have some instability or sensitivity to randomness, which could come from either the PCA or GMM components.
   
2. `Convergence`: The sklearn GMM converges more smoothly and consistently, with fewer perturbations in the curves. This indicates that sklearn's implementation likely handles numerical precision and optimization better, leading to more stable results.
   
3. `Model Choice`: Both models suggest a low number of components (BIC minimum at 3), but the custom GMM shows far less confidence in the results due to noisy behavior, whereas sklearn’s GMM presents a clearer distinction between good and bad fits.

Custom GMM implementation may need adjustments in optimization techniques or random initialization control to reduce the noise and achieve more reliable convergence, similar to the stability observed in sklearn's GMM.

Therefore `kgmm3 = 3`:

![6.4.2](./figures/6_4_2.png?raw=true"Title")


# 7 Cluster Analysis

## 7.1 K-Means Cluster Analysis

`kkmeans1 = 7` and `dataset=original`

K-means Clustering Results:
Number of Clusters: 7
Cost (WCSS): 3843.1084
Silhouette Score: 0.0373
Davies Bouldin Score: 3.8670
Fit time: 0.1497 seconds
Predict time: 0.0024 seconds
Cluster sizes:
	Cluster 1: 49 points
	Cluster 2: 29 points
	Cluster 3: 23 points
	Cluster 4: 22 points
	Cluster 5: 17 points
	Cluster 6: 37 points
	Cluster 7: 23 points

`kkmeans1 = 7` and `dataset=reduced`

K-means Clustering Results:
Number of Clusters: 7
Cost (WCSS): 327.6721
Silhouette Score: 0.2385
Davies Bouldin Score: 1.2722
Fit time: 0.1199 seconds
Predict time: 0.0001 seconds
Cluster sizes:
	Cluster 1: 36 points
	Cluster 2: 24 points
	Cluster 3: 30 points
	Cluster 4: 28 points
	Cluster 5: 24 points
	Cluster 6: 33 points
	Cluster 7: 25 points

`k2 = 5` and `dataset=original`

K-means Clustering Results:
Number of Clusters: 5
Cost (WCSS): 3990.5359
Silhouette Score: 0.0413
Davies Bouldin Score: 4.0205
Fit time: 0.1403 seconds
Predict time: 0.0011 seconds
Cluster sizes:
	Cluster 1: 19 points
	Cluster 2: 61 points
	Cluster 3: 33 points
	Cluster 4: 69 points
	Cluster 5: 18 points

`k2 = 5` and `dataset=reduced`

K-means Clustering Results:
Number of Clusters: 5
Cost (WCSS): 421.6117
Silhouette Score: 0.2480
Davies Bouldin Score: 1.2938
Fit time: 0.1020 seconds
Predict time: 0.0001 seconds
Cluster sizes:
	Cluster 1: 28 points
	Cluster 2: 58 points
	Cluster 3: 40 points
	Cluster 4: 35 points
	Cluster 5: 39 points

`kkmeans3 = 6` and `dataset=original`

K-means Clustering Results:
Number of Clusters: 6
Cost (WCSS): 3965.1347
Silhouette Score: 0.0343
Davies Bouldin Score: 4.1253
Fit time: 0.1420 seconds
Predict time: 0.0020 seconds
Cluster sizes:
	Cluster 1: 35 points
	Cluster 2: 25 points
	Cluster 3: 10 points
	Cluster 4: 12 points
	Cluster 5: 62 points
	Cluster 6: 56 points

`kkmeans3 = 6` and `dataset=reduced`

K-means Clustering Results:
Number of Clusters: 6
Cost (WCSS): 394.8850
Silhouette Score: 0.2092
Davies Bouldin Score: 1.3768
Fit time: 0.1130 seconds
Predict time: 0.0002 seconds
Cluster sizes:
	Cluster 1: 25 points
	Cluster 2: 34 points
	Cluster 3: 37 points
	Cluster 4: 33 points
	Cluster 5: 36 points
	Cluster 6: 35 points


Silhoette scores for original dataset:
1. `kkmeans1 = 7`: 0.0373
2. `k2 = 5`: 0.0413
3. `kkmeans3 = 6`: 0.0343

Davies-Bouldin scores for original dataset:
1. `kkmeans1 = 7`: 3.8670
2. `k2 = 5`: 4.0205
3. `kkmeans3 = 6`: 4.1253

Silhouette scores for reduced dataset:
1. `kkmeans1 = 7`: 0.2385
2. `k2 = 5`: 0.2480
3. `kkmeans3 = 6`: 0.2092

Davies-Bouldin scores for reduced dataset
1. `kkmeans1 = 7`: 1.2722
2. `k2 = 5`: 1.2938
3. `kkmeans3 = 6`: 1.3768

Therefore, `kkmeans = 5`

### Observations:

1. Performance on original vs. reduced dataset:
   - K-means clustering consistently performs better on the PCA-reduced dataset compared to the original dataset. This is evident from the higher Silhouette scores and lower Davies-Bouldin scores for the reduced dataset across all configurations.
   - The improvement is substantial: Silhouette scores increase by a factor of 5-6, while Davies-Bouldin scores decrease by a factor of about 3 when moving from the original to the reduced dataset.

2. Impact of number of clusters:
   - For the original dataset, the differences in performance across different numbers of clusters (5, 6, or 7) are relatively small.
   - For the reduced dataset, there's more variation in performance with the number of clusters, suggesting that the dimensionality reduction has made the cluster structure more apparent.

3. Optimal number of clusters:
   - For the original dataset, 5 clusters (k2 = 5) show slightly better performance with the highest Silhouette score (0.0413) and a mid-range Davies-Bouldin score.
   - For the reduced dataset, 5 clusters also perform best, with the highest Silhouette score (0.2480) and a competitive Davies-Bouldin score (1.2938).

4. Clustering quality:
   - The clustering quality on the original dataset is poor, with very low Silhouette scores (all below 0.05) and high Davies-Bouldin scores (all above 3.8).
   - The reduced dataset shows much better clustering quality, with Silhouette scores around 0.2-0.25 and Davies-Bouldin scores around 1.2-1.4.
   - However, even the best Silhouette score (0.2480) indicates that the clusters are not very well-separated or compact.

5. Cluster sizes:
   - In the original dataset, cluster sizes are often imbalanced, especially for 5 and 6 clusters.
   - The reduced dataset tends to produce more balanced cluster sizes across all configurations.

6. Computational efficiency:
   - The fit and predict times are generally faster for the reduced dataset, which is expected due to the lower dimensionality.
   - The difference in computational time between the original and reduced datasets is more pronounced for the predict step than the fit step.

7. Cost (WCSS - Within-Cluster Sum of Squares):
   - The WCSS is much lower for the reduced dataset, which is expected due to the lower dimensionality.
   - However, WCSS values can't be directly compared between the original and reduced datasets due to the difference in dimensionality.

8. Data complexity:
   - The substantial improvement in clustering quality after PCA suggests that the original dataset has high dimensionality that complicates the clustering task.
   - Even after dimensionality reduction, the moderate Silhouette scores indicate that the data may not have a very strong inherent cluster structure.

In conclusion, K-means clustering shows significantly better results on the PCA-reduced dataset, with 5 clusters appearing to be the optimal choice.

## 7.2 GMM Cluster Analysis

For `kgmm1 = 1` and `dataset = original`

Fitting GMM with data shape: (200, 512)
Prediction statistics:
Unique predictions: [0]
Prediction counts: [200]
Number of unique labels: 1
Unique labels: [0]
Label counts: [200]
Warning: All samples were assigned to the same cluster.
Prediction statistics:
Unique predictions: [0]
Prediction counts: [200]
Number of Unique labels: 1
Unique labels: [0]
Label counts: [200]
Warning: All samples were assigned to the same cluster.
Silhoette Score: -1.0000
Davies Bouldin Score: -1.0000

For `kgmm1 = 1` and `dataset = reduced`

Fitting GMM with data shape: (200, 5)
Prediction statistics:
Unique predictions: [0]
Prediction counts: [200]
Number of unique labels: 1
Unique labels: [0]
Label counts: [200]
Warning: All samples were assigned to the same cluster.
Prediction statistics:
Unique predictions: [0]
Prediction counts: [200]
Number of Unique labels: 1
Unique labels: [0]
Label counts: [200]
Warning: All samples were assigned to the same cluster.
Silhoette Score: -1.0000
Davies Bouldin Score: -1.0000

For `k2 = 1` and `dataset = original`

Fitting GMM with data shape: (200, 512)
Prediction statistics:
Unique predictions: [0 1 2 3 4]
Prediction counts: [29 83  8 27 53]
Number of unique labels: 5
Unique labels: [0 1 2 3 4]
Label counts: [29, 83, 8, 27, 53]
Prediction statistics:
Unique predictions: [0 1 2 3 4]
Prediction counts: [29 83  8 27 53]
Number of Unique labels: 5
Unique labels: [0 1 2 3 4]
Label counts: [29, 83, 8, 27, 53]
Silhoette Score: -0.0107
Davies Bouldin Score: 7.3164

For `k2 = 1` and `dataset = reduced`

Fitting GMM with data shape: (200, 5)
Prediction statistics:
Unique predictions: [0 1 2 3 4]
Prediction counts: [83 11 21 37 48]
Number of unique labels: 5
Unique labels: [0 1 2 3 4]
Label counts: [83, 11, 21, 37, 48]
Prediction statistics:
Unique predictions: [0 1 2 3 4]
Prediction counts: [83 11 21 37 48]
Number of Unique labels: 5
Unique labels: [0 1 2 3 4]
Label counts: [83, 11, 21, 37, 48]
Silhoette Score: 0.0395
Davies Bouldin Score: 1.9508

For `kgmm3 = 3` and `dataset = original`

Fitting GMM with data shape: (200, 512)
Prediction statistics:
Unique predictions: [0 1 2]
Prediction counts: [ 6 96 98]
Number of unique labels: 3
Unique labels: [0 1 2]
Label counts: [6, 96, 98]
Prediction statistics:
Unique predictions: [0 1 2]
Prediction counts: [ 6 96 98]
Number of Unique labels: 3
Unique labels: [0 1 2]
Label counts: [6, 96, 98]
Silhoette Score: 0.0029
Davies Bouldin Score: 7.6730

For `kgmm3 = 3` and `dataset = reduced`

Fitting GMM with data shape: (200, 5)
Prediction statistics:
Unique predictions: [0 1 2]
Prediction counts: [52 64 84]
Number of unique labels: 3
Unique labels: [0 1 2]
Label counts: [52, 64, 84]
Prediction statistics:
Unique predictions: [0 1 2]
Prediction counts: [52 64 84]
Number of Unique labels: 3
Unique labels: [0 1 2]
Label counts: [52, 64, 84]
Silhoette Score: 0.1527
Davies Bouldin Score: 1.9640


On original non-PCA dataset:
1. `kgmm1 = 1`: silhouette_score = -1; davies-bouldin score = -1
2. `k2 = 5`: silhouette_score = -0.0107; davies-bouldin score = 7.3164
3. `kgmm3 = 3`: silhouette_score = 0.0029; davies-bouldin score = 7.6730

On PCAed reduced_dataset:
1. `kgmm1 = 1`: silhouette_score = -1; davies-bouldin score = -1
2. `k2 = 5`: silhouette_score = 0.0395; davies-bouldin score = 1.9508
3. `kgmm3 = 3`: silhouette_score = 0.1527; davies-bouldin score = 1.9640

Higher silhouette_score and lower davies-bouldin score means better clustering. Therefore, `kgmm = 3`

Sklearn's GMM (performs better):

Silhouette Score: 0.0369

Davies Bouldin Score: 4.7551

Silhouette Score (reduced dataset): 0.1936

Davies Bouldin Score (reduced dataset): 1.7574

### Observations:


1. Performance on original vs. reduced dataset:
   - GMM consistently performs better on the PCA-reduced dataset compared to the original dataset. This is evident from the higher Silhouette scores and lower Davies-Bouldin scores for the reduced dataset across all configurations.
   - This suggests that dimensionality reduction through PCA has helped in improving the clustering quality by potentially reducing noise and focusing on the most important features.

2. Impact of number of components:
   - For k=1 (single component), both datasets show the worst possible scores (Silhouette = -1, Davies-Bouldin = -1). This indicates that a single cluster is entirely inadequate for representing the data structure.
   - Increasing the number of components from 1 to 3 or 5 significantly improves the clustering quality, especially for the reduced dataset.

3. Optimal number of components:
   - For the original dataset, the difference in performance between 3 and 5 components is minimal, with a slight edge to 3 components (Silhouette score of 0.0029 vs -0.0107).
   - For the reduced dataset, 3 components (kgmm3) show the best performance with the highest Silhouette score (0.1527) and a comparable Davies-Bouldin score to the 5-component model.

4. Sklearn's GMM performance:
   - Sklearn's GMM implementation shows better performance than the custom implementation, particularly on the reduced dataset.
   - On the reduced dataset, Sklearn's GMM achieves the highest Silhouette score (0.1936) and the lowest Davies-Bouldin score (1.7574) among all configurations.

5. Clustering quality:
   - While the clustering improves with more components and on the reduced dataset, the overall Silhouette scores are still relatively low (best is 0.1936). This suggests that the clusters are not very well-separated or compact.
   - The Davies-Bouldin scores for the original dataset are quite high, indicating poor cluster separation. The scores improve significantly for the reduced dataset, but still suggest room for improvement in cluster definition.

6. Data complexity:
   - The substantial improvement in clustering quality after PCA suggests that the original dataset may have high dimensionality that complicates the clustering task.
   - The difficulty in achieving high-quality clusters even after dimensionality reduction indicates that the underlying data structure might not naturally separate into distinct, well-defined clusters.

In conclusion, while GMM clustering shows better results on the PCA-reduced dataset with 3 components, the overall clustering quality suggests that the data may not have a strong inherent cluster structure. The Sklearn implementation of GMM appears to handle this data more effectively than the my implementation.




## 7.3 Compare K-means and GMMs

`kkmeans = 5`

For original dataset (better than sklearn)

Silhouette scores: `0.0413`
Davies-bouldin scores: `4.0205`

For reduced dataset (slightly lesser than sklearn)

Silhouette scores: `0.2480`
Davies-bouldin scores: `1.2938`

`kgmm = 3`

For original dataset (lesser than sklearn)

Silhouette scores: `0.0029`
Davies-bouldin scores: `7.6730`

For reduced dataset (lesser than sklearn)

Silhouette scores: `0.1527`
Davies-bouldin scores: `1.9640`

### Observations:

1. Performance on original vs. reduced dataset:
   - Both K-means and GMM show significant improvement when applied to the reduced (PCA) dataset compared to the original dataset.
   - This consistent improvement across both methods suggests that dimensionality reduction is crucial for revealing the underlying structure in this data.

2. K-means vs. GMM performance:
   - K-means consistently outperforms GMM across both datasets and evaluation metrics.
   - On the original dataset, K-means achieves a higher Silhouette score (0.0413 vs 0.0029) and a lower Davies-Bouldin score (4.0205 vs 7.6730) compared to GMM.
   - On the reduced dataset, K-means again shows better performance with a higher Silhouette score (0.2480 vs 0.1527) and a lower Davies-Bouldin score (1.2938 vs 1.9640).

3. Optimal number of clusters:
   - The best performance for K-means is achieved with 5 clusters, while GMM performs best with 3 components.
   - This difference suggests that K-means might be capturing finer granularity in the data structure compared to GMM.

4. Clustering quality:
   - While both methods show improvement on the reduced dataset, the overall clustering quality is moderate at best.
   - The highest Silhouette score (0.2480 for K-means on the reduced dataset) indicates that while there is some cluster structure, the clusters are not very well-separated or compact.

5. Relative improvement:
   - K-means shows a larger relative improvement from the original to the reduced dataset compared to GMM.
   - This suggests that K-means benefits more from the dimensionality reduction process in this case.

6. Comparison to sklearn implementations:
   - For K-means, the custom implementation performs better than sklearn on the original dataset, but slightly worse on the reduced dataset.
   - For GMM, the sklearn implementation consistently outperforms the custom implementation.
   - This difference in relative performance to sklearn implementations might indicate areas for potential improvement in the custom GMM implementation.

7. Robustness to high dimensionality:
   - K-means appears more robust to high dimensionality, performing relatively better on the original high-dimensional dataset compared to GMM.

8. Appropriateness of clustering assumptions:
   - The better performance of K-means might suggest that the underlying data structure aligns more closely with the assumptions of K-means (roughly spherical clusters) than with those of GMM (Gaussian-distributed clusters).

In conclusion, for this particular dataset, K-means clustering with 5 clusters on the PCA-reduced data provides the best clustering results among the tested configurations.

# Hierarchial Clustering

1. Cluster Distribution:
   - Most methods (6 out of 8) resulted in highly imbalanced clusters, with 198 points in one cluster and 1 point each in the other two clusters.
   - Only the complete linkage with Euclidean distance produced a more balanced distribution (75, 120, and 5 points).

2. Linkage Method Impact:
   - Single, average, and centroid linkages consistently produced the same highly imbalanced clustering across different distance metrics.
   - Complete linkage showed the most variation in results between Euclidean and cosine distances.

3. Distance Metric Influence:
   - For most linkage methods, changing the distance metric from Euclidean to cosine did not significantly alter the clustering results.
   - The complete linkage method was the most sensitive to the choice of distance metric, showing substantial differences in both cluster distribution and evaluation scores.

4. Silhouette Scores:
   - Generally low across all methods (range: 0.0288 to 0.1062), indicating poor cluster separation overall.
   - The highest silhouette score (0.1062) was achieved by multiple methods, all resulting in the highly imbalanced 198-1-1 distribution.

5. Davies-Bouldin Scores:
   - Varied more widely than silhouette scores (range: 0.7734 to 3.6922).
   - Lower scores (better) were associated with the imbalanced 198-1-1 clustering.
   - The highest (worst) score was for complete linkage with Euclidean distance, which had the most balanced cluster distribution.

6. Computation Time:
   - Generally fast, with fit times ranging from 0.0027 to 0.0043 seconds.
   - Cosine distance typically resulted in slightly faster computation times compared to Euclidean distance.

7. Outlier Sensitivity:
   - The prevalence of the 198-1-1 split suggests that most methods are identifying two points as significant outliers in the dataset.

8. Euclidean vs. Cosine:
   - For complete linkage, Euclidean distance led to more balanced clusters but worse evaluation scores compared to cosine distance.

Comparision Results:

Comparison with K-means (k=5):
Adjusted Rand Index: 0.1914
Adjusted Mutual Information: 0.2564

Hierarchical Clustering (k=5) cluster sizes:
Cluster 1: 7 points
Cluster 2: 68 points
Cluster 3: 120 points
Cluster 4: 4 points
Cluster 5: 1 points

Comparison with GMM (k=3):
Adjusted Rand Index: 0.0126
Adjusted Mutual Information: 0.0261

Hierarchical Clustering (k=3) cluster sizes:
Cluster 1: 75 points
Cluster 2: 120 points
Cluster 3: 5 points

1. Comparison with K-means (k=5):
   - The Adjusted Rand Index (ARI) of 0.1914 and Adjusted Mutual Information (AMI) of 0.2564 indicate a low to moderate agreement between hierarchical clustering and K-means.
   - This suggests that while there's some similarity in how the two methods group the data, there are also significant differences.

2. Comparison with GMM (k=3):
   - The very low ARI (0.0126) and AMI (0.0261) scores indicate that the hierarchical clustering results are very different from the GMM clustering.
   - This suggests that these two methods are identifying quite different structures in the data.

3. Cluster Sizes:
   - For k=5, hierarchical clustering produces highly imbalanced clusters, with one very small cluster (1 point) and one very large cluster (120 points).
   - For k=3, the clustering is less imbalanced, but still has one dominant cluster (120 points) and one small cluster (5 points).

4. Outliers:
   - The presence of very small clusters (4 and 1 points for k=5, 5 points for k=3) suggests the existence of potential outliers or small, distinct groups in the data.

5. Method Differences:
   - The hierarchical clustering results align more closely with K-means than with GMM, though the agreement is still relatively low.


# Nearest Neighbour Search

## PCA + KNN

![9.1.1](./figures/9_1_1.png?raw=true"Title")

Optimal Number of dimensions would be `3`

X.shape:  (114000, 10)

self.components.shape:  (10, 3)

transformed data shape:  (114000, 3)

[[-0.25140584 -0.18159672 -0.08531468]
 [ 0.74047394 -0.3647264  -0.29580146]
 [ 0.15282965  0.0378516  -0.46217271]
 ...
 [ 0.43100201 -0.52817273  0.2352823 ]
 [ 0.07874689 -0.14768998 -0.14548096]
 [ 0.25548291 -0.39453796  0.16611707]]

`Applying KNN on reduced dataset`:

Predictions made: 1000/11400
Predictions made: 2000/11400
Predictions made: 3000/11400
Predictions made: 4000/11400
Predictions made: 5000/11400
Predictions made: 6000/11400
Predictions made: 7000/11400
Predictions made: 8000/11400
Predictions made: 9000/11400
Predictions made: 10000/11400
Predictions made: 11000/11400

Validation Set Results:
accuracy: 0.0875
macro_p: 0.0838
macro_r: 0.0882
macro_f1: 0.0810
micro_p: 0.0875
micro_r: 0.0875
micro_f1: 0.0875
avg_time: 0.0163

## Evaluation

     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k metric
0.087544 0.083822 0.088198  0.080998 0.087544 0.087544  0.087544 15 cosine

### Comparision with A1 scores:

Before applying PCA:

acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k metric

0.218333 0.223391 0.219633   0.22025 0.218333 0.218333  0.218333 15 cosine

After appyling PCA:

acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k metric

0.087544 0.083822 0.088198  0.080998 0.087544 0.087544  0.087544 15 cosine

### Analysis:

1. applying PCA has degraded the performance of KNN significantly.
2. The accuracy dropped by nearly 13%, which indicates that PCA might have reduced the dimensionality too much, causing the model to lose critical information that was helping in classification.

Possible Reasons:
1. `Loss of variance`: PCA selects components that maximize variance, but these might not always align with the features important for classification. If the variance is reduced too much, the KNN model can struggle to classify correctly.
2. `Number of Components`: It's possible that PCA reduced the data to too few components, leading to a loss of information that was crucial for the KNN classifier.
3. `KNN Sensitivity`: KNN can be sensitive to the shape of the feature space. Reducing dimensions might have distorted the neighborhoods, making it harder for KNN to find the correct neighbors.
4. `Imbalanced Dataset`: The results before PCA suggest that the dataset might be imbalanced, which could also explain why the macro metrics are higher than the micro metrics (macro metrics are more sensitive to class imbalance).

### 9.2.3 Inference Time

`Overall Trend`:

Inference time increases almost linearly with the dataset size for both the original and PCA-reduced datasets.

`Original Dataset`:

The inference time increases more rapidly as the dataset size grows, particularly noticeable for larger datasets.
For the largest dataset size (~100,000), the inference time approaches 160 seconds.

`Reduced Dataset` (PCA Applied):

The PCA-reduced dataset consistently has a lower inference time compared to the original dataset across all sizes.
This reduction in inference time becomes more prominent as the dataset size increases.
For the largest dataset size, the inference time is slightly under 150 seconds, showing a roughly 10-second improvement over the original dataset.

`Inference Time Gap`:

The gap between the original and reduced datasets' inference times is minimal for smaller datasets but becomes more pronounced with larger datasets, indicating that the PCA reduction significantly helps with computational efficiency as the dataset size increases.





