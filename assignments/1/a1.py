# K-NN
# Data Visualization
# For help, I prompted GPT for how to analyze different plots like Histogram, Bar graphs based on some standard characteristics
"""
Prompt: I want to make observations on the distribution of a single feature in a dataset. Provide me terms to describe the distribution along with its definition. Categorize these terms based on the graph type on which they are used.

Here’s a list of common terms used to describe the distribution of a single feature in a dataset, categorized based on the type of graph they are often associated with:

### **1. Terms Used with Histograms**
- **Skewness**: Describes the asymmetry of the distribution. A distribution can be **positively skewed** (right-skewed, where the tail is longer on the right) or **negatively skewed** (left-skewed, where the tail is longer on the left).
- **Kurtosis**: Describes the "tailedness" of the distribution. **High kurtosis** indicates heavy tails, while **low kurtosis** indicates light tails.
- **Bimodal/Multimodal**: Indicates the presence of two (bimodal) or more (multimodal) peaks in the distribution.
- **Uniform Distribution**: Indicates that all values are equally likely, resulting in a flat histogram.
- **Outliers**: Data points that are significantly different from the rest of the data. These appear as isolated bars far from the main distribution.
- **Range**: The difference between the minimum and maximum values in the dataset, affecting the spread of the histogram.
- **Normal Distribution**: A bell-shaped, symmetrical distribution where the mean, median, and mode are all equal.

### **2. Terms Used with Box Plots**
- **Median**: The middle value of the dataset, represented by the line inside the box.
- **Interquartile Range (IQR)**: The range between the first quartile (Q1) and the third quartile (Q3), indicating the spread of the middle 50% of the data.
- **Outliers**: Data points that fall outside 1.5 times the IQR above Q3 or below Q1. They are usually plotted as individual points.
- **Whiskers**: Lines extending from the box to the smallest and largest values within 1.5 times the IQR from Q1 and Q3, respectively.
- **Symmetry**: A box plot is symmetrical if the median is centered in the box and the whiskers are of equal length.

### **3. Terms Used with Density Plots**
- **Peaks (Modes)**: The high points in the curve represent the most frequent data values. A density plot can have one (unimodal) or more (multimodal) peaks.
- **Bandwidth**: Controls the smoothness of the density estimate. A small bandwidth may show more detail (but could also highlight noise), while a larger bandwidth may oversmooth the data.
- **Tails**: The ends of the distribution curve, indicating the frequency of extreme values.
- **Bimodal/Multimodal**: Similar to histograms, these terms describe the presence of multiple peaks in the density plot.
- **Skewness**: Asymmetry in the curve, where one tail is longer than the other.
- **Kurtosis**: The steepness of the peak and the weight of the tails.

### **4. Terms Used with Q-Q (Quantile-Quantile) Plots**
- **Linearity**: A Q-Q plot shows how well the data follows a particular distribution. If the points lie along the reference line, the data follows that distribution.
- **Heavy Tails**: Points deviating from the reference line at the ends of the plot indicate heavy tails (high kurtosis).
- **Light Tails**: Points deviating towards the middle of the plot suggest light tails (low kurtosis).
- **Outliers**: Points that significantly deviate from the reference line, indicating outliers in the data.
- **S-Shape**: Indicates a distribution with a lower-than-normal kurtosis (platykurtic).

### **5. Terms Used with Bar Plots**
- **Frequency**: The height of each bar represents the count or frequency of each category or value in the dataset.
- **Proportion/Percentage**: Bars can represent proportions or percentages, showing the relative frequency of each category.
- **Categorical Distribution**: The overall pattern of how categories are distributed. This term is used when the bar plot shows different categories (e.g., different types of fruits).
- **Mode**: The category with the highest bar, indicating the most frequent value.
- **Uniformity**: Indicates if the bars are of similar height, suggesting that all categories have similar frequencies.
- **Skewness**: Though less common in bar plots compared to histograms, skewness can still describe an asymmetric distribution of categorical data.
- **Outliers**: Categories with an unusually high or low frequency compared to others. These are evident as particularly tall or short bars relative to the rest.
- **Clustering**: Refers to groups of bars with similar heights, indicating that certain categories have similar frequencies.
- **Variability**: The differences in the heights of the bars, indicating how much the frequencies or proportions vary across categories.
- **Relative Height**: Comparison of bar heights to discuss relative frequencies between different categories.

These terms will help you effectively describe and analyze the distribution of a single feature in your dataset.

"""


# Import Necessary libraries for Data Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
2.2.1 Task-1

Generate suitable plots that show distributions of various features in the dataset.

Write comments about what you observe (for example, outliers, skewed data,
etc)

Try to visualize different combinations of features to try and understand
the correlation of the features with the target variable.

Try coming up with a hierarchy of which columns are more necessary than
others for the classification task based on the data exploration. Mention all
these results in the report.
"""

# Load the Data using pandas
df = pd.read_csv("../../data/external/spotify.csv") # read spotify.csv

"""
df.info():
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 114000 entries, 0 to 113999
Data columns (total 21 columns):
 #   Column            Non-Null Count   Dtype  
---  ------            --------------   -----  
 0   Unnamed: 0        114000 non-null  int64  
 1   track_id          114000 non-null  object 
 2   artists           113999 non-null  object 
 3   album_name        113999 non-null  object 
 4   track_name        113999 non-null  object 
 5   popularity        114000 non-null  int64  
 6   duration_ms       114000 non-null  int64  
 7   explicit          114000 non-null  bool   
 8   danceability      114000 non-null  float64
 9   energy            114000 non-null  float64
 10  key               114000 non-null  int64  
 11  loudness          114000 non-null  float64
 12  mode              114000 non-null  int64  
 13  speechiness       114000 non-null  float64
 14  acousticness      114000 non-null  float64
 15  instrumentalness  114000 non-null  float64
 16  liveness          114000 non-null  float64
 17  valence           114000 non-null  float64
 18  tempo             114000 non-null  float64
 19  time_signature    114000 non-null  int64  
 20  track_genre       114000 non-null  object 
dtypes: bool(1), float64(9), int64(6), object(5)
memory usage: 17.5+ MB
"""
# Remove the rows that contain empty cells (here: artists, album_name, track_name contain null values/empty values)
df.dropna(inplace=True) # since we have lots of data, we simply remove the row

"""
df.info()
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
"""

# Popularity-Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['popularity'], bins=100)
plt.title("Popularity - Histogram")
plt.xlabel("popularity")
plt.ylabel("frequency")
plt.savefig("figures/popularity.png")

# Observations:
"""
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
"""

# Duration_ms-Distribution (Binned)
# Histogram. Bins = 1000
plt.hist(df['duration_ms'], bins=1000)
plt.title("Duration (ms) - Histogram")
plt.xlabel("duration_ms")
plt.ylabel("frequency")
plt.savefig("figures/duration_ms.png")

# Observations:
"""
    Long Tail towards end as compared to start
    Sinlge peaks:
        - at around 250000 approx.
        - presence of a single cluster
    Almost no outliers
    Low kurtosis (indicates light tail)
"""


# Explicit-Distribution (Basic)
# Bar Graph E
true_count = (df['explicit'] == True).sum()
false_count = (df['explicit'] == False).sum()
true_percentage = (true_count) / (true_count + false_count)
false_percentage = (false_count) / (true_count + false_count)

plt.figure(figsize=(6, 4))
plt.bar(['False', 'True'], [false_percentage * 100, true_percentage * 100], color=['blue', 'red'])
plt.title("Explicit - Bar Graph")
plt.xlabel("explicit")
plt.ylabel("percentage")
plt.savefig("figures/explicit.png")

# Observations:
'''
    Imbalanced distribution, high relative height
    Mode: False
    Biased towards Falses
    True instances very low compared to False instaces
    Extremely high variablity
'''

# Danceability - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['danceability'], bins=100)
plt.title("Danceability - Histogram")
plt.xlabel("danceability")
plt.ylabel("frequency")
plt.savefig("figures/danceability.png")

# Observations:
'''
    Negative skewness
        - tail of the distribution is longer to the left side
    Sinlge peak:
        - at around 250000 approx.
        - presence of a single cluster
    Few outliers near 0.8 from the normal-like distribution
    Range: 0-1
'''
# Energy - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['energy'], bins=100)
plt.title("Energy - Histogram")
plt.xlabel("energy")
plt.ylabel("frequency")
plt.savefig("figures/energy.png")

# Observations:
'''
    Linear increasing distribution
    Few outliers near 0, 0.55 and 0.95 from linear-like distribution
    Range: 0-1
'''

# Loudness - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['loudness'], bins=100)
plt.title("Loudness - Histogram")
plt.xlabel("loudness")
plt.ylabel("frequency")
plt.savefig("figures/loudness.png")

# Observations:
'''
    Negatively skewed distribution
    High kurtosis (heavy tails)
    No outliers
    Range: both positive and negative
    Peak: near -6.000
'''

# Speechiness - Distribution (Binned)
# Histogram. Bins = 200
plt.hist(df['speechiness'], bins=200)
plt.title("Speechiness - Histogram")
plt.xlabel("speechiness")
plt.ylabel("frequency")
plt.savefig("figures/speechiness.png")

# Observations:
'''
    Positive skewness with sharp dip at beginning
        - tail of the distribution is longer to the right side
    Sinlge peak:
    Outliers: near 0.0 (sharp decrease) and some instances b/w 0.8-1.0
    Range: 0-1
'''

# Acousticness - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['acousticness'], bins=100)
plt.title("Acousticness - Histogram")
plt.xlabel("acousticness")
plt.ylabel("frequency")
plt.savefig("figures/acousticness.png")

# Observations:
'''
    Outlier: near 0.0 and 1.0 (sharp increase)
    Positive skewed with peak at 0.0 and Negative skewed with peak at 1.0
    High Kurtosis (heavy tails)
'''

# Instrumentalness - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['instrumentalness'], bins=100)
plt.title("Instrumentalness - Histogram")
plt.xlabel("instrumentalness")
plt.ylabel("frequency")
plt.savefig("figures/instrumentalness.png")

# Observations:
'''
    Outlier: 0.0 (heavily outlied from rest) (very high frequency)
    High Kurtosis (heavy tails)
    Two peaks: near 0.0 and near 0.9
    range: 0-1
'''

# Liveness - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['liveness'], bins=100)
plt.title("Liveness - Histogram")
plt.xlabel("liveness")
plt.ylabel("frequency")
plt.savefig("figures/liveness.png")

# Observations:
'''
    Range: 0-1
    Almost uniform distribution from 0.45
    Two peaks: (less kurtosis (light tails))
        - near 0.1
        - near 0.35
'''

# valence - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['valence'], bins=100)
plt.title("valence - Histogram")
plt.xlabel("valence")
plt.ylabel("frequency")

# ChatGPT: How to increase the number of labels on x-axis
# Increase the number of labels on the x-axis
ticks = np.linspace(0, 1, num=11)  # Adjust 'num' to set the number of ticks
plt.xticks(ticks)

plt.savefig("figures/valence.png")

# Observations:
'''
    Range: 0-1
    Outliers: 0.00 (sudden peak), 0.00-0.05 (less freq), 0.05 (sudden peak), 0.96 (sudden peak)
    Inverted-U curve distribution
'''

# tempo - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['tempo'], bins=100)
plt.title("tempo - Histogram")
plt.xlabel("tempo")
plt.ylabel("frequency")
plt.savefig("figures/tempo.png")

# Observations:
'''
    Range: 0-250
    Outliers: Too-many small outliers
'''

# time_signature - Distribution (Binned)
# Histogram. Bins = 100
plt.hist(df['time_signature'], bins=5)
plt.title("time_signature - Histogram")
plt.xlabel("time_signature")
plt.ylabel("frequency")
plt.savefig("figures/time_signature.png")

# Observations:
'''
    Distribution: Categorical: 0, 1, 2, 3, 4, 5 (labels)
    Non-uniform distributn with outlier 5 (high frequency)
    Frequency of labels 0 and 2 are negligible, while 1, 3 are significant
'''

# check the distribution of target_feature column
target_feature = 'track_genre'
category_counts = df[target_feature].value_counts()

plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar', color='skyblue')
plt.title('Frequency of Categories')
plt.xlabel('Category')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.savefig("figures/track_genre.png")


# Pair-plots using Seaborn library on selected feautures based on hieuristics
target_feature = 'track_genre'

# MinMax Scaling / 0-1 scaling
list_of_features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
df_minmax = (df[list_of_features] - df[list_of_features].min()) / (df[list_of_features].max() - df[list_of_features].min())

df_minmax[target_feature] = df[target_feature]

plt.figure(figsize=(12, 12))
sns.pairplot(data=df_minmax, hue=target_feature)

# Place the legend below the plot with horizontal orientation
plt.legend(
    loc='upper center',      # Position the legend at the top center of the plot
    bbox_to_anchor=(0.5, -0.1),  # Move the legend below the plot (x, y)
    ncol=5,                 # Number of columns in the legend
    fontsize=7,            # Font size of the legend labels
    title_fontsize=12       # Font size of the legend title
)

plt.grid(True)
plt.tight_layout()

plt.savefig("figures/correlation_pairplot.png")


# Pair plot between two features seperately (for better scale)

feature_1 = "acousticness"
feature_2 = "speechiness"
target_feature = "track_genre"

# MinMax Scaling / 0-1 scaling
df_minmax = (df[list_of_features] - df[list_of_features].min()) / (df[list_of_features].max() - df[list_of_features].min())

# Add the target_feature column back to the scaled DataFrame
df_minmax[target_feature] = df[target_feature]

# Standardization / Unit-Normal scaling
mean_feature_1 = df[feature_1].mean()
std_feature_1 = df[feature_1].std()

mean_feature_2 = df[feature_2].mean()
std_feature_2 = df[feature_2].std()
df_std = pd.DataFrame()
df_std[feature_1] = (df[feature_1] - mean_feature_1) / std_feature_1
df_std[feature_2] = (df[feature_2] - mean_feature_2) / std_feature_2


# Add the target_feature column back to the standardized DataFrame
df_std[target_feature] = df[target_feature]

# Plotting
plt.figure(figsize=(12, 12))

# Create the scatter plot for normal data
# sns.scatterplot(data=df, x=feature_1, y=feature_2, hue=target_feature, alpha=0.5)

# Create the scatter plot for standardized data
# sns.scatterplot(data=df_std, x=feature_1, y=feature_2, hue=target_feature, alpha=0.5)

# Create the scatter plot for min-max scaled data (Optional: Uncomment if needed)
sns.scatterplot(data=df_minmax, x=feature_1, y=feature_2, hue=target_feature, alpha=0.5)

# Place the legend below the plot with horizontal orientation
plt.legend(
    loc='upper center',      # Position the legend at the top center of the plot
    bbox_to_anchor=(0.5, -0.1),  # Move the legend below the plot (x, y)
    ncol=5,                 # Number of columns in the legend
    fontsize=7,            # Font size of the legend labels
    title_fontsize=12       # Font size of the legend title
)

plt.title(f'{feature_1} vs {feature_2} by {target_feature}')
plt.xlabel(f'{feature_1}')
plt.ylabel(f'{feature_2}')
plt.grid(True)
plt.tight_layout()

plt.savefig("figures/acousticness_vs_speechiness.png")

# Correlation Heat map for getting better picture
target_feature = 'track_genre'

# MinMax Scaling / 0-1 scaling
list_of_features = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
df_minmax = (df[list_of_features] - df[list_of_features].min()) / (df[list_of_features].max() - df[list_of_features].min())

df_minmax[target_feature] = df[target_feature]

# Calculate the correlation matrix
corr_matrix = df_minmax[list_of_features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Draw the heatmap
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar_kws={'label': 'Correlation Coefficient'}, linewidths=0.5)

# Customize the plot
plt.title('Correlation Heatmap of Features', fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.yticks(rotation=0)  # Keep y-axis labels horizontal
plt.tight_layout()

plt.savefig("figures/correlation_heatmap.png")

#-------------------------------------------------K-NEAREST-NEIGHBOURS-----------------------------------------
# Imports necessary for K-NN
import pandas as pd
import numpy as np
from numpy.linalg import norm # Uses: cosine similarity, ...
from collections import defaultdict
import time
import matplotlib.pyplot as plt
from numpy import linalg


"""
    n: Number of training samples (rows in the dataset)
    m: Number of features used for distance computatio)
    k: Number of nearest neighbors considered (k-value)
    f: Number of samples to predict (size of the test set)
    d: Distance metric used (manhattan, euclidean, cosine)
    L: Number of unique labels for storing the count of each label
"""
class KNN_model:
    # TC: O(1); SC: O(m) -> to store the list of features and other attributes
    def __init__(self, k: int, distance_metrics: str, features: list):
        # Initialize the KNN model with the number of neighbors (k),
        # the distance metric to use ('manhattan', 'euclidean', or 'cosine'),
        # and the list of features to consider for distance calculation.
        self.k = k
        self.distance_metrics = distance_metrics
        self.features = features
        self.train_data = None # Placeholder for the training dataset
        self.prediction_count = 0 # Counter for the number of predictions made
        self.total_time_taken = 0 # Accumulator for total time taken to make predictions

    # TC: O(1); SC: O(1) 
    def get_k(self):
        # Getter method for k (number of neighbors)
        return self.k
    
    # TC: O(1); SC: O(1)
    def set_k(self, k):
        # Setter method for k (allows updating the number of neighbors)
        self.k = k

    # TC: O(1); SC: O(1)
    def get_distance_metrics(self):
        # Getter method for the distance metric
        return self.distance_metrics
    
    # TC: O(1); SC: O(1)
    def set_distance_metrics(self, distance_metrics):
        # Setter method for the distance metric (allows updating the metric)
        self.distance_metrics = distance_metrics

    # TC: O(m); SC: O(1)
    def get_distances(self, row1, row2):
        # Compute the distance between two rows based on the selected distance metric
        if self.distance_metrics == 'manhattan':
            # Manhattan distance (sum of absolute differences)
            distance = 0.0
            for feature in self.features:
                diff = abs(row1[feature] - row2[feature])
                distance += diff
            return distance
        elif self.distance_metrics == 'euclidean':
            # Euclidean distance (sum of squared differences)
            distance = 0.0
            for feature in self.features:
                diff = row1[feature] - row2[feature]
                distance += diff
            return distance
        elif self.distance_metrics == 'cosine':
            # Cosine distance (1 - cosine similarity)
            dot_product = 0.0
            norm1 = 0.0
            norm2 = 0.0
            for feature in self.features:
                dot_product += row1[feature] * row2[feature]
                norm1 += row1[feature] * row1[feature]
                norm2 += row2[feature] * row2[feature]
            if norm1 == 0 and norm2 == 0:
                return 1.0 # Handle the case where both norms are zero
            cosine_similarity = dot_product / ((norm1 ** 0.5) * (norm2 ** 0.5))
            return (1.0 - cosine_similarity)  # Return cosine distance
        else:
            # Raise an error if an invalid metric is provided
            return ValueError("invalid metric")

    # TC: O(1); SC: O(nm)
    def train(self, df):
        # Store the training data
        self.train_data = df

    # TC: O(n); SC: O(n)
    def split_data(self, validation_split=0.1, test_split=0.1):
        # Split the dataset into training, validation, and test sets
        self.validation_split = validation_split
        self.test_split = test_split

        # Shuffle the dataset and reset indices
        shuffled_df = self.train_data.sample(frac = 1).reset_index(drop=True) # Shuffle total df with dropping na
        
        # Calculate the sizes of the validation and test sets
        self.validation_size = int(len(shuffled_df) * validation_split)
        self.test_size = int(len(shuffled_df) * test_split)
        print(f"validation size: {self.validation_size}\ttest_size: {self.test_size}\n")
        
        # Extract the test, validation, and training sets from the shuffled data
        self.test_set = shuffled_df.iloc[: self.test_size]
        self.valid_set = shuffled_df.iloc[self.test_size : self.test_size + self.validation_size]
        self.train_set = shuffled_df.iloc[self.test_size + self.validation_size : ]

    # TC: O(k); SC: O(L)
    def get_majority(self, nearest_neighbours):
        # Determine the majority label among the nearest neighbors
        label_counts = {}

        # Count the occurrences of each label in the nearest neighbors
        for i, _ in nearest_neighbours:
            label = self.train_data.iloc[i]['track_genre']  # Use the index to get the label
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Return the label with the highest count (majority vote)
        predicted_label = max(label_counts, key=label_counts.get)
        return predicted_label

    # TC: (nm + nlogn + k) (computing distances + sorting the distances + determining majority label)
    # SC: O(n) -> store the distances for all 'n' training samples
    def predict_a_sample(self, test_row):
        # Predict the label for a single test sample
        start_time = time.time()
        distances = []

        # Calculate distances between the test sample and all training samples
        for i, train_row in self.train_data.iterrows():
            distance = self.get_distances(test_row, train_row)
            distances.append((i, distance))

        # Sort distances and select the k-nearest neighbors
        distances.sort(key=lambda x: x[1])
        nearest_neighbours = distances[:self.k]

        # Get the majority label from the k-nearest neighbors
        prediction = self.get_majority(nearest_neighbours)

        # Calculate time taken for prediction and update totals
        time_taken = time.time() - start_time
        self.total_time_taken += time_taken
        self.prediction_count += 1

        # Optionally print progress every 10 predictions
        if self.prediction_count % 10 == 0:
            print(f"Predictions made: {self.prediction_count}/{self.validation_size}")
        return prediction
    
    # TC: O(f (nm + n log n + k)) -> make prediction for each of the 'f' test samples
    # SC: O(f n) -> same
    def predict(self, X_test):
        # Predict labels for all samples in the test set
        self.prediction_count = 0 # Reset prediction counter 
        self.total_time_taken = 0 # Reset time counter

        # Generate predictions for each sample in the test set
        predictions = [self.predict_a_sample(row) for _, row in X_test.iterrows()]
        return predictions
    

class KNN_evaluate:
    def __init__(self, KNN):
        # Initialize the evaluation object with a KNN model
        self.KNN: object = KNN
        self.validation_split = self.KNN.validation_split
        self.test_split = self.KNN.test_split
    
    # calculate the evaluation scores manually using numpy
    def calculate_metrics(self, true_y, pred_y):
        # Get the unique classes in the true labels
        unique_classes = np.unique(true_y) # O(n), Space: O(C) where C is the number of unique classes
        
        # Initialize dictionaries for macro scores
        precision_dict = defaultdict(float) # O(1), Space: O(C)
        recall_dict = defaultdict(float) # O(1), Space: O(C)
        F1_dict = defaultdict(float) # O(1), Space: O(C)
        
        # Initialize variables for micro scores
        tp_micro = 0 # O(1), Space: O(1)
        fp_micro = 0 # O(1), Space: O(1)
        fn_micro = 0 # O(1), Space: O(1)
        
        # Iterate over each unique class to calculate precision, recall, and F1-score
        for cls in unique_classes:
            # Calculate true positives, false positives, and false negatives for each class
            tp = np.sum((true_y == cls) & (pred_y == cls)) # O(n), Space: O(1)
            fp = np.sum((true_y != cls) & (pred_y == cls)) # O(n), Space: O(1)
            fn = np.sum((true_y == cls) & (pred_y != cls)) # O(n), Space: O(1)
            
            # Calculate precision, recall, and F1-score for the current class
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0 # O(1), Space: O(1)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0 # O(1), Space: O(1)
            F1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0 # O(1), Space: O(1)   
            
            # Store the calculated metrics in the corresponding dictionaries
            precision_dict[cls] = precision # O(1), Space: O(1)
            recall_dict[cls] = recall # O(1), Space: O(1)
            F1_dict[cls] = F1_score # O(1), Space: O(1)
            
            # for micro scores, accumulate tp, fp, fn
            tp_micro += tp # O(1), Space: O(1)
            fp_micro += fp # O(1), Space: O(1)
            fn_micro += fn # O(1), Space: O(1)
        
        # Calculate macro scores
        macro_precision = np.mean(list(precision_dict.values())) # O(C), Space: O(C)
        macro_recall = np.mean(list(recall_dict.values())) # O(C), Space: O(C)
        macro_F1_score = np.mean(list(F1_dict.values())) # O(C), Space: O(C)
        
        # Calculate micro-average metrics by using accumulated true positives, false positives, and false negatives
        micro_precision = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
        micro_recall = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
        micro_F1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        accuracy = np.mean(true_y == pred_y)
        
        return accuracy, macro_precision, macro_recall, macro_F1_score, micro_precision, micro_recall, micro_F1_score
    
    def evaluate(self, X_test):
        # Extract true labels from the test set
        true_y = X_test['track_genre'].values  # O(f), Space: O(f) where f is the number of samples in X_test
        pred_y = self.KNN.predict(X_test) # O(f * (n * m + n log n + k)), Space: O(f * n)
        
        true_y = np.array(true_y) # O(f), Space: O(f)
        pred_y = np.array(pred_y) # O(f), Space: O(f)

        acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1 = self.calculate_metrics(true_y, pred_y)

        avg_time_taken_per_prediction = self.KNN.total_time_taken / self.KNN.prediction_count

        return {
            'accuracy': acc,
            'macro_p': macro_p,
            'macro_r': macro_r,
            'macro_f1': macro_f1,
            'micro_p': micro_p,
            'micro_r': micro_r,
            'micro_f1': micro_f1,
            'avg_time': avg_time_taken_per_prediction
        }

    # Print evaluation metrics for a given set (validation or test)
    def print_metrics(self, metrics, set_name):
        print(f"\n{set_name} Set Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")



class Vectorized_KNN_model:
    # TC: O(1) - Simply assigns values to variables.
    # SC: O(d) - Space used to store the features list.
    def __init__(self, k: int, distance_metrics: str, features: list):
        self.k = k
        self.distance_metrics = distance_metrics
        self.features = features
        self.train_embeddings = None  # Numpy array for training embeddings
        self.train_labels = None  # Numpy array for training labels
        self.prediction_count = 0
        self.total_time_taken = 0

    # TC: O(1) - Simple getter function
    def get_k(self):
        return self.k
    
    # TC: O(1) - Simple setter function.
    def set_k(self, k):
        self.k = k

    # TC: O(1) - Simple getter function.
    def get_distance_metrics(self):
        return self.distance_metrics
    
    # TC: O(1) - Simple setter function.
    def set_distance_metrics(self, distance_metrics):
        self.distance_metrics = distance_metrics

    # TC:
    # - Manhattan: O(n * d) - Computes absolute differences and sums them across all features.
    # - Euclidean: O(n * d) - Computes squared differences, sums them, and takes the square root.
    # - Cosine: O(n * d) - Computes dot product and norms, then performs division.
    # SC: O(n) - Space to store the distances for each training sample.
    def calculate_distances(self, test_embedding):
        if self.distance_metrics == 'manhattan':
            distances = np.sum(np.abs(self.train_embeddings - test_embedding), axis=1)
        elif self.distance_metrics == 'euclidean':
            distances = np.sqrt(np.sum((self.train_embeddings - test_embedding) ** 2, axis=1))
        elif self.distance_metrics == 'cosine':
            dot_product = np.dot(self.train_embeddings, test_embedding)
            norms = np.linalg.norm(self.train_embeddings, axis=1) * np.linalg.norm(test_embedding)
            distances = 1 - (dot_product / norms)
        else:
            raise ValueError("Invalid distance metric")
        return distances
    
    # TC: O(n) - Shuffling and splitting data.
    # SC: O(n * d) - Space to store embeddings and labels after splitting.
    def split_data(self, validation_split=0.1, test_split=0.1):
        self.validation_split = validation_split
        self.test_split = test_split
        shuffled_df = self.train_data.sample(frac = 1).reset_index(drop=True) # shuffle total df with dropping na
        self.validation_size = int(len(shuffled_df) * validation_split)
        self.test_size = int(len(shuffled_df) * test_split)
        print(f"validation size: {self.validation_size}\ttest_size: {self.test_size}\n")
        
        self.test_set = shuffled_df.iloc[: self.test_size]
        self.valid_set = shuffled_df.iloc[self.test_size : self.test_size + self.validation_size]
        self.train_set = shuffled_df.iloc[self.test_size + self.validation_size : ]

        self.train_embeddings = self.train_set[self.features].values
        self.train_labels = self.train_set['track_genre']

    # TC: O(1) - Simply assigns the dataframe.
    # SC: O(n * d) - Space to store the training data embeddings and labels.
    def train(self, df):
        self.train_data = df

    # TC: O(k) - Determines the majority label among the k nearest neighbors.
    # SC: O(k) - Space to store the nearest labels.
    def get_majority(self, nearest_indices):
        nearest_labels = self.train_labels.iloc[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    # TC: O(n * d + k) - Calculate distances (O(n * d)) + get majority label (O(k)).
    # SC: O(n) - Space for storing distances.
    def predict_a_sample(self, row):
        start_time = time.time()
        row_embedding = row[self.features].values
        distances = self.calculate_distances(row_embedding)
        nearest_indices = np.argpartition(distances, self.k)[:self.k] # argpartition is more optimized
        prediction = self.get_majority(nearest_indices)
        time_taken = time.time() - start_time
        self.total_time_taken += time_taken
        self.prediction_count += 1
        if self.prediction_count % 10 == 0:
            print(f"Predictions made: {self.prediction_count}/{self.validation_size}")
        return prediction
    
    # TC: O(m * (n * d + k)) - Predicting m samples.
    # SC: O(m * n) - Space to store distances for all m test samples.
    def predict(self, X_test):
        self.prediction_count = 0
        self.total_time_taken = 0
        predictions = [self.predict_a_sample(row) for _, row in X_test.iterrows()]
        return predictions
    
"""
    n -> number of training samples
    d -> number of features
    k -> hyperparameter
    m -> number of test samples
    c -> number of metrics
"""
class Best_KNN_model:
    # TC: O(1) => initializes the class variables
    # SC: O(1) => doesn't depend on the size of the input data
    def __init__(self, k: int, distance_metrics: str, features: list):
        self.k = k
        self.distance_metrics = distance_metrics
        self.features = features
        self.train_embeddings = None  # Numpy array for training embeddings
        self.train_labels = None  # Numpy array for training labels
        self.prediction_count = 0
        self.total_time_taken = 0

    # TC: O(1) => Simple retrieval of k.
    # SC: O(1) => No additional space is used.
    def get_k(self):
        return self.k
    
    # TC: O(1) => Simple assignment of k.
    # SC: O(1) => No additional space is used.
    def set_k(self, k):
        self.k = k

    # TC: O(1) => Simple retrieval of distance metrics.
    # SC: O(1) => No additional space is used.
    def get_distance_metrics(self):
        return self.distance_metrics
    
    # TC: O(1) => Simple assignment of distance metrics.
    # SC: O(1) => No additional space is used.
    def set_distance_metrics(self, distance_metrics):
        self.distance_metrics = distance_metrics

    # TC: O(nd) => Computes distances between a test sample and all training samples.
    # SC: O(n) => Stores the distances for all training samples.
    def calculate_distances(self, test_embedding):
        if self.distance_metrics == 'manhattan':
            distances = np.sum(np.abs(self.train_embeddings - test_embedding), axis=1)
        elif self.distance_metrics == 'euclidean':
            distances = np.sqrt(np.sum((self.train_embeddings - test_embedding) ** 2, axis=1))
        elif self.distance_metrics == 'cosine':
            dot_product = np.dot(self.train_embeddings, test_embedding)
            norms = np.linalg.norm(self.train_embeddings, axis=1) * np.linalg.norm(test_embedding)
            distances = 1 - (dot_product / norms)
        else:
            raise ValueError("Invalid distance metric")
        return distances
    
    # TC: O(n) => Shuffles and splits the dataset into training, validation, and test sets.
    # SC: O(n) => Stores the embeddings and labels for each split.
    def split_data(self, validation_split=0.1, test_split=0.1):
        self.validation_split = validation_split
        self.test_split = test_split
        total_samples = self.X.shape[0]
        indices = np.random.permutation(total_samples)

        test_size = int(total_samples * test_split)
        valid_size = int(total_samples * validation_split)
        self.test_size = test_size
        self.validation_size = valid_size
        test_indices = indices[:test_size]
        valid_indices = indices[test_size:test_size+valid_size]
        train_indices = indices[test_size+valid_size:]
        
        self.X_test, self.y_test = self.X[test_indices], self.y[test_indices]
        self.X_valid, self.y_valid = self.X[valid_indices], self.y[valid_indices]
        self.X_train, self.y_train = self.X[train_indices], self.y[train_indices]
        
        self.train_embeddings = self.X_train
        self.train_labels = self.y_train

    # TC: O(n) => Converts the input DataFrame into numpy arrays.
    # SC: O(nd) => Stores the features and labels from the DataFrame.
    def train(self, df):
        self.df = df
        self.X = np.array(self.df[self.features].values)
        self.y = np.array(self.df['track_genre'].values)

    # TC: O(k) => Finds the majority label among k nearest neighbors.
    # SC: O(k) => Stores labels of the k nearest neighbors.
    def get_majority(self, nearest_indices):
        nearest_labels = self.train_labels[nearest_indices]
        unique_labels, counts = np.unique(nearest_labels, return_counts=True)
        return unique_labels[np.argmax(counts)]

    # TC: O(nd + k log k) => Computes distances and sorts to find k nearest neighbors.
    # SC: O(n) 
    def predict_a_sample(self, row_embedding):
        start_time = time.time()
        distances = self.calculate_distances(row_embedding)
        nearest_indices = np.argsort(distances)[:self.k] # argpartition is more optimized
        prediction = self.get_majority(nearest_indices)
        time_taken = time.time() - start_time
        self.total_time_taken += time_taken
        self.prediction_count += 1
        if self.prediction_count % 1000 == 0:
            print(f"Predictions made: {self.prediction_count}/{self.validation_size}")
        return prediction
    
    # TC: O(m(nd + k log k))
    # SC: O(mn)
    def predict(self, X_test):
        self.prediction_count = 0
        self.total_time_taken = 0
        predictions = [self.predict_a_sample(row) for row in X_test]
        return predictions
    

class Best_KNN_evaluate:
    # TC: O(1)
    def __init__(self, KNN):
        self.KNN: object = KNN
        self.validation_split = self.KNN.validation_split
        self.test_split = self.KNN.test_split
    
    # TC: O(mc)
    # SC: O(c + m)
    # calculate the evaluation scores manually using numpy
    def calculate_metrics(self, true_y, pred_y):
        unique_classes = np.unique(true_y)
        
        # Initialize dictionaries for macro scores
        precision_dict = defaultdict(float)
        recall_dict = defaultdict(float)
        F1_dict = defaultdict(float)
        
        # Initialize variables for micro scores
        tp_micro = 0
        fp_micro = 0
        fn_micro = 0
        
        for cls in unique_classes:
            tp = np.sum((true_y == cls) & (pred_y == cls))
            fp = np.sum((true_y != cls) & (pred_y == cls))
            fn = np.sum((true_y == cls) & (pred_y != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            F1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_dict[cls] = precision
            recall_dict[cls] = recall
            F1_dict[cls] = F1_score
            
            # for micro scores, accumulate tp, fp, fn
            tp_micro += tp
            fp_micro += fp
            fn_micro += fn
        
        # Calculate macro scores
        macro_precision = np.mean(list(precision_dict.values()))
        macro_recall = np.mean(list(recall_dict.values()))
        macro_F1_score = np.mean(list(F1_dict.values()))
        
        # Calculate micro scores
        micro_precision = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0
        micro_recall = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0
        micro_F1_score = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        accuracy = np.mean(true_y == pred_y)
        
        return accuracy, macro_precision, macro_recall, macro_F1_score, micro_precision, micro_recall, micro_F1_score
    
    # TC: O(m(nd + k log k) + mc)
    # SC: O(mn + c + m)
    def evaluate(self, X_test, y_test):
        # true_y = X_test['track_genre'].values
        true_y = np.array(y_test)
        pred_y = np.array(self.KNN.predict(X_test))
        
        true_y = np.array(true_y)
        pred_y = np.array(pred_y)

        acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1 = self.calculate_metrics(true_y, pred_y)

        avg_time_taken_per_prediction = self.KNN.total_time_taken / self.KNN.prediction_count

        return {
            'accuracy': acc,
            'macro_p': macro_p,
            'macro_r': macro_r,
            'macro_f1': macro_f1,
            'micro_p': micro_p,
            'micro_r': micro_r,
            'micro_f1': micro_f1,
            'avg_time': avg_time_taken_per_prediction
        }
    
    # TC; O(c)
    def print_metrics(self, metrics, set_name):
        print(f"\n{set_name} Set Results:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

df = pd.read_csv("../../data/external/spotify.csv", nrows=2000)
df.dropna()
features_to_normalize = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

knn_model = KNN_model(3, "manhattan", features_to_normalize)
knn_model.train(normalized_df)
knn_model.split_data(validation_split=0.1, test_split=0.1)
evaluator = KNN_evaluate(knn_model)
validation_metrics = evaluator.evaluate(knn_model.valid_set)
evaluator.print_metrics(validation_metrics, "Validation")

"""
Validation Set Results:
accuracy: 0.8900
macro_p: 0.8895
macro_r: 0.8895
macro_f1: 0.8895
micro_p: 0.8900
micro_r: 0.8900
micro_f1: 0.8900
avg_time: 0.0920
"""

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

knn_model = Vectorized_KNN_model(3, "manhattan", features_to_normalize)
knn_model.train(normalized_df)
knn_model.split_data(validation_split=0.1, test_split=0.1)
evaluator = KNN_evaluate(knn_model)
validation_metrics = evaluator.evaluate(knn_model.valid_set)
evaluator.print_metrics(validation_metrics, "Validation")

"""
Validation Set Results:
accuracy: 0.1546
macro_p: 0.1740
macro_r: 0.1528
macro_f1: 0.1487
micro_p: 0.1546
micro_r: 0.1546
micro_f1: 0.1546
avg_time: 0.0620
"""

df = pd.read_csv("../../data/external/spotify.csv", nrows=2000)
df.dropna()
features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

knn_model = Best_KNN_model(3, "manhattan", features_to_normalize)
knn_model.train(normalized_df)
knn_model.split_data(validation_split=0.1, test_split=0.1)
evaluator = Best_KNN_evaluate(knn_model)
validation_metrics = evaluator.evaluate(knn_model.X_valid, knn_model.y_valid)
evaluator.print_metrics(validation_metrics, "Validation")

"""
Validation Set Results:
accuracy: 0.8650
macro_p: 0.8626
macro_r: 0.8703
macro_f1: 0.8638
micro_p: 0.8650
micro_r: 0.8650
micro_f1: 0.8650
avg_time: 0.0001
"""

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

knn_model = Best_KNN_model(3, "manhattan", features_to_normalize)
knn_model.train(normalized_df)
knn_model.split_data(validation_split=0.1, test_split=0.1)
evaluator = Best_KNN_evaluate(knn_model)
validation_metrics = evaluator.evaluate(knn_model.X_valid, knn_model.y_valid)
evaluator.print_metrics(validation_metrics, "Validation")
"""
Validation Set Results:
accuracy: 0.1975
macro_p: 0.2250
macro_r: 0.1969
macro_f1: 0.1941
micro_p: 0.1975
micro_r: 0.1975
micro_f1: 0.1975
avg_time: 0.0091
"""

### HYPERPARAMETER TUNING

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

k_values = []
for k in range(1, 20, 2):
    k_values.append(k)

metrics = ['manhattan', 'euclidean', 'cosine']

results = []

for k in k_values:
    for metric in metrics:
        print(f"\nEvaluating for k: {k}, distance metrics: {metric}")
        model = Best_KNN_model(1, 'manhattan', features_to_normalize)
        model.train(normalized_df)
        model.split_data(validation_split=0.1, test_split=0.1)
        model.set_k = k
        model.set_distance_metrics = metric
        evaluator = Best_KNN_evaluate(model)
        validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid)
        evaluator.print_metrics(validation_metrics, "Validation")
        acc = validation_metrics['accuracy']
        macro_p = validation_metrics['macro_p']
        macro_r = validation_metrics['macro_r']
        macro_f1 = validation_metrics['macro_f1']
        micro_p = validation_metrics['micro_p']
        micro_r = validation_metrics['micro_r']
        micro_f1 = validation_metrics['micro_f1']
        results.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
best_result = sorted_results[0]
print("==================================================================")
print("Best Accuracy hyperparameters:")
df = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))
print("==================================================================")
df = pd.DataFrame(sorted_results[:10], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))

"""
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
"""

# K vs Accuracy
custom_metric = 'euclidean'

results_custom = []

for (acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric) in results:
    if metric == custom_metric:
        results_custom.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

df_results_custom = pd.DataFrame(results_custom, columns=['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])

print("==================================================================")
plt.figure(figsize=(12, 12))
plt.plot(df_results_custom['k'], df_results_custom['acc'], marker='o')
plt.title(f"k vs acc for {custom_metric} distance")
plt.xlabel('k')
plt.ylabel('acc')
plt.grid(True)
# plt.show()
plt.savefig("figures/kVSacc_euclidean.png")

custom_metric = 'manhattan'

results_custom = []

for (acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric) in results:
    if metric == custom_metric:
        results_custom.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

df_results_custom = pd.DataFrame(results_custom, columns=['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])

print("==================================================================")
plt.figure(figsize=(12, 12))
plt.plot(df_results_custom['k'], df_results_custom['acc'], marker='o')
plt.title(f"k vs acc for {custom_metric} distance")
plt.xlabel('k')
plt.ylabel('acc')
plt.grid(True)
# plt.show()
plt.savefig("figures/kVSacc_manhattan.png")


custom_metric = 'cosine'

results_custom = []

for (acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric) in results:
    if metric == custom_metric:
        results_custom.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

df_results_custom = pd.DataFrame(results_custom, columns=['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])

print("==================================================================")
plt.figure(figsize=(12, 12))
plt.plot(df_results_custom['k'], df_results_custom['acc'], marker='o')
plt.title(f"k vs acc for {custom_metric} distance")
plt.xlabel('k')
plt.ylabel('acc')
plt.grid(True)
# plt.show()
plt.savefig("kVSacc_cosine.png")

### BONUS: More data doen't mean more accuracy

"""
    Dropping columns: popularity, instrumentalness, valence
"""

import json

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'liveness', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

k_values = []
for k in range(1, 5, 2):
    k_values.append(k)

metrics = ['manhattan', 'euclidean', 'cosine']

results = []

for k in k_values:
    for metric in metrics:
        print(f"\nEvaluating for k: {k}, distance metrics: {metric}")
        model = Best_KNN_model(1, 'manhattan', features_to_normalize)
        model.train(normalized_df)
        model.split_data(validation_split=0.1, test_split=0.1)
        model.set_k = k
        model.set_distance_metrics = metric
        evaluator = Best_KNN_evaluate(model)
        validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid)
        evaluator.print_metrics(validation_metrics, "Validation")
        acc = validation_metrics['accuracy']
        macro_p = validation_metrics['macro_p']
        macro_r = validation_metrics['macro_r']
        macro_f1 = validation_metrics['macro_f1']
        micro_p = validation_metrics['micro_p']
        micro_r = validation_metrics['micro_r']
        micro_f1 = validation_metrics['micro_f1']
        results.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
best_result = sorted_results[0]
print("==================================================================")
print("Best Accuracy hyperparameters:")
df = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))
print("==================================================================")
df = pd.DataFrame(sorted_results[:9], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))


"""
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
"""

# """
#     Combination: 'acousticness', 'energy'
#     Reason: Extreme negative correlation coefficient
# """

import json

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['acousticness', 'energy']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

k_values = []
for k in range(3, 4, 2):
    k_values.append(k)

metrics = ['manhattan', 'euclidean', 'cosine']

results = []

for k in k_values:
    for metric in metrics:
        print(f"\nEvaluating for k: {k}, distance metrics: {metric}")
        model = Best_KNN_model(1, 'manhattan', features_to_normalize)
        model.train(normalized_df)
        model.split_data(validation_split=0.1, test_split=0.1)
        model.set_k = k
        model.set_distance_metrics = metric
        evaluator = Best_KNN_evaluate(model)
        validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid)
        evaluator.print_metrics(validation_metrics, "Validation")
        acc = validation_metrics['accuracy']
        macro_p = validation_metrics['macro_p']
        macro_r = validation_metrics['macro_r']
        macro_f1 = validation_metrics['macro_f1']
        micro_p = validation_metrics['micro_p']
        micro_r = validation_metrics['micro_r']
        micro_f1 = validation_metrics['micro_f1']
        results.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
best_result = sorted_results[0]
print("==================================================================")
print("Best Accuracy hyperparameters:")
df = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))
print("==================================================================")
df = pd.DataFrame(sorted_results[:9], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))

"""
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.106228 0.104928 0.106188  0.104861 0.106228 0.106228  0.106228  3 manhattan
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.106228 0.104928 0.106188  0.104861 0.106228 0.106228  0.106228  3 manhattan
0.100702 0.100796 0.100093  0.099764 0.100702 0.100702  0.100702  3    cosine
0.098772 0.098860 0.098826  0.098064 0.098772 0.098772  0.098772  3 euclidean
"""

"""
    Combination: 'speechiness', 'tempo'
    Reason: pair-plot
"""

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['speechiness', 'tempo']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

k_values = []
for k in range(3, 4, 2):
    k_values.append(k)

metrics = ['manhattan', 'euclidean', 'cosine']

results = []

for k in k_values:
    for metric in metrics:
        print(f"\nEvaluating for k: {k}, distance metrics: {metric}")
        model = Best_KNN_model(1, 'manhattan', features_to_normalize)
        model.train(normalized_df)
        model.split_data(validation_split=0.1, test_split=0.1)
        model.set_k = k
        model.set_distance_metrics = metric
        evaluator = Best_KNN_evaluate(model)
        validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid)
        evaluator.print_metrics(validation_metrics, "Validation")
        acc = validation_metrics['accuracy']
        macro_p = validation_metrics['macro_p']
        macro_r = validation_metrics['macro_r']
        macro_f1 = validation_metrics['macro_f1']
        micro_p = validation_metrics['micro_p']
        micro_r = validation_metrics['micro_r']
        micro_f1 = validation_metrics['micro_f1']
        results.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
best_result = sorted_results[0]
print("==================================================================")
print("Best Accuracy hyperparameters:")
df = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))
print("==================================================================")
df = pd.DataFrame(sorted_results[:9], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))

"""
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k metric
0.105088 0.104312 0.103943  0.103314 0.105088 0.105088  0.105088  3 cosine
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.105088 0.104312 0.103943  0.103314 0.105088 0.105088  0.105088  3    cosine
0.104123 0.102875 0.104275  0.102647 0.104123 0.104123  0.104123  3 manhattan
0.104123 0.101472 0.104117  0.102021 0.104123 0.104123  0.104123  3 euclidean
"""

"""
    Combination: 'popularity', 'speechiness'
    Reason: pair-plot
"""

import json

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['popularity', 'speechiness']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

k_values = []
for k in range(3, 4, 2):
    k_values.append(k)

metrics = ['manhattan', 'euclidean', 'cosine']

results = []

for k in k_values:
    for metric in metrics:
        print(f"\nEvaluating for k: {k}, distance metrics: {metric}")
        model = Best_KNN_model(1, 'manhattan', features_to_normalize)
        model.train(normalized_df)
        model.split_data(validation_split=0.1, test_split=0.1)
        model.set_k = k
        model.set_distance_metrics = metric
        evaluator = Best_KNN_evaluate(model)
        validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid)
        evaluator.print_metrics(validation_metrics, "Validation")
        acc = validation_metrics['accuracy']
        macro_p = validation_metrics['macro_p']
        macro_r = validation_metrics['macro_r']
        macro_f1 = validation_metrics['macro_f1']
        micro_p = validation_metrics['micro_p']
        micro_r = validation_metrics['micro_r']
        micro_f1 = validation_metrics['micro_f1']
        results.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
best_result = sorted_results[0]
print("==================================================================")
print("Best Accuracy hyperparameters:")
df = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))
print("==================================================================")
df = pd.DataFrame(sorted_results[:9], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))

"""
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.069912 0.069292 0.069799  0.069109 0.069912 0.069912  0.069912  3 manhattan
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.069912 0.069292 0.069799  0.069109 0.069912 0.069912  0.069912  3 manhattan
0.066930 0.067960 0.067544  0.067465 0.066930 0.066930  0.066930  3    cosine
0.066140 0.066442 0.067129  0.066318 0.066140 0.066140  0.066140  3 euclidean
"""

"""
    Combination: 'speechiness', 'instrumentalness'
    Reason: pair-plot
"""

import json

df = pd.read_csv("../../data/external/spotify.csv")
df.dropna()
features_to_normalize = ['speechiness', 'instrumentalness']

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

k_values = []
for k in range(3, 4, 2):
    k_values.append(k)

metrics = ['manhattan', 'euclidean', 'cosine']

results = []

for k in k_values:
    for metric in metrics:
        print(f"\nEvaluating for k: {k}, distance metrics: {metric}")
        model = Best_KNN_model(1, 'manhattan', features_to_normalize)
        model.train(normalized_df)
        model.split_data(validation_split=0.1, test_split=0.1)
        model.set_k = k
        model.set_distance_metrics = metric
        evaluator = Best_KNN_evaluate(model)
        validation_metrics = evaluator.evaluate(model.X_valid, model.y_valid)
        evaluator.print_metrics(validation_metrics, "Validation")
        acc = validation_metrics['accuracy']
        macro_p = validation_metrics['macro_p']
        macro_r = validation_metrics['macro_r']
        macro_f1 = validation_metrics['macro_f1']
        micro_p = validation_metrics['micro_p']
        micro_r = validation_metrics['micro_r']
        micro_f1 = validation_metrics['micro_f1']
        results.append((acc, macro_p, macro_r, macro_f1, micro_p, micro_r, micro_f1, k, metric))

sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
best_result = sorted_results[0]
print("==================================================================")
print("Best Accuracy hyperparameters:")
df = pd.DataFrame(sorted_results[:1], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))
print("==================================================================")
df = pd.DataFrame(sorted_results[:9], columns = ['acc', 'macro_p', 'macro_r', 'macro_f1', 'micro_p', 'micro_r', 'micro_f1', 'k', 'metric'])
print(df.to_string(index=False))

"""
==================================================================
Best Accuracy hyperparameters:
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.070614 0.071262 0.071313  0.070377 0.070614 0.070614  0.070614  3 manhattan
==================================================================
     acc  macro_p  macro_r  macro_f1  micro_p  micro_r  micro_f1  k    metric
0.070614 0.071262 0.071313  0.070377 0.070614 0.070614  0.070614  3 manhattan
0.067456 0.066836 0.067268  0.066485 0.067456 0.067456  0.067456  3    cosine
0.065965 0.066283 0.065664  0.065467 0.065965 0.065965  0.065965  3 euclidean
"""

####################################################INFERENCE TIME########################################################

# Inference time vs Models

from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("../../data/external/spotify.csv", nrows=1000)
df.dropna()

normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
normalized_df['track_genre'] = df['track_genre']

features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

sklearn_knn = KNeighborsClassifier(n_neighbors = 3, metric = 'manhattan')
inference_times = []

models = [KNN_model, Vectorized_KNN_model, Best_KNN_model]
evaluators = [KNN_evaluate, KNN_evaluate, Best_KNN_evaluate]

for model, evaluator in zip(models, evaluators):
    if (model == KNN_model or model == Vectorized_KNN_model):
        start_time = time.time()
        obj = model(3, "manhattan", features_to_normalize)
        obj.train(normalized_df)
        obj.split_data(validation_split=0.1, test_split=0.1)
        eval = evaluator(obj)
        validation_metrics = eval.evaluate(obj.valid_set)
        inference_times.append(time.time() - start_time)
    else:
        start_time = time.time()
        obj = Best_KNN_model(3, 'manhattan', features_to_normalize)
        obj.train(normalized_df)
        obj.split_data(validation_split=0.1, test_split=0.1)
        eval = Best_KNN_evaluate(obj)
        validation_metrics = eval.evaluate(obj.X_valid, obj.y_valid)
        end_time = time.time()
        inference_times.append(end_time - start_time)

start_time = time.time()
sklearn_knn.fit(obj.X_train, obj.y_train)
sklearn_knn.predict(obj.X_valid)
end_time = time.time()
inference_times.append(end_time - start_time)

model_names = ['Initial_KNN', 'Vectorized KNN','Best KNN','sklearn KNN']
plt.bar(model_names, inference_times)
plt.ylabel('Inference Time (seconds) 10000 data points')
plt.title('Inference Time Comparison for Different KNN Models')
# plt.show()
plt.savefig("figures/inference_time.png")


# Inference time vs Train Dataset size

# As the number of data points (size of the dataset), the time complexity is expected and tends to increase as well.
# Reason for this is both the distance calculation and the sorting steps have dependencies on the number of data points.
# The growth initially seems to be linear for small dataset sizes. But as the dataset size increases, it becomes quadratic.

from sklearn.neighbors import KNeighborsClassifier

features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                            'liveness', 'valence', 'tempo']

sklearn_knn = KNeighborsClassifier(n_neighbors = 3, metric = 'manhattan')
models = [KNN_model, Vectorized_KNN_model, Best_KNN_model, sklearn_knn]
evaluators = [KNN_evaluate, KNN_evaluate, Best_KNN_evaluate]
inference_times = {model: [] for model in models}
sklearn_inferece_times = []

n_rows = [100, 500, 1000, 2000, 4000, 6000, 10000]
for n_row in n_rows:
    df = pd.read_csv("../../data/external/spotify.csv", nrows=n_row)
    df.dropna()
    normalized_df = (df[features_to_normalize] - df[features_to_normalize].min()) / (df[features_to_normalize].max() - df[features_to_normalize].min())
    normalized_df['track_genre'] = df['track_genre']

    for model, evaluator in zip(models, evaluators):
        if (model == KNN_model or model == Vectorized_KNN_model):
            start_time = time.time()
            obj = model(3, "manhattan", features_to_normalize)
            obj.train(normalized_df)
            obj.split_data(validation_split=0.1, test_split=0.1)
            eval = evaluator(obj)
            validation_metrics = eval.evaluate(obj.valid_set)
            inference_times[model].append(time.time() - start_time)
        else:
            start_time = time.time()
            obj = Best_KNN_model(3, 'manhattan', features_to_normalize)
            obj.train(normalized_df)
            obj.split_data(validation_split=0.1, test_split=0.1)
            eval = Best_KNN_evaluate(obj)
            validation_metrics = eval.evaluate(obj.X_valid, obj.y_valid)
            end_time = time.time()
            inference_times[model].append(end_time - start_time)

    start_time = time.time()
    sklearn_knn.fit(obj.X_train, obj.y_train)
    sklearn_knn.predict(obj.X_valid)
    end_time = time.time()
    sklearn_inferece_times.append(end_time - start_time)

inference_times[sklearn_knn] = sklearn_inferece_times
for model, times in inference_times.items():
    plt.plot(n_rows, times, label=model_names[models.index(model)])
plt.xlabel('Training Dataset Size')
plt.ylabel('Inference Time (seconds)')
plt.title('Inference Time vs Training Dataset Size')
plt.legend()
# plt.show()
plt.savefig("figures/inference_time_vs_data_size.png")


#########################################################SECOND DATASET##############################################################

import json

train_df = pd.read_csv("../../data/external/spotify-2/train.csv")
test_df = pd.read_csv("../../data/external/spotify-2/test.csv")
valid_df = pd.read_csv("../../data/external/spotify-2/validate.csv")

train_df.dropna()   
test_df.dropna()
valid_df.dropna()

features_to_normalize = ['popularity', 'danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 
                         'liveness', 'valence', 'tempo']

normalized_train_df = (train_df[features_to_normalize] - train_df[features_to_normalize].min()) / (train_df[features_to_normalize].max() - train_df[features_to_normalize].min())
normalized_train_df['track_genre'] = train_df['track_genre']
normalized_test_df = (test_df[features_to_normalize] - test_df[features_to_normalize].min()) / (test_df[features_to_normalize].max() - test_df[features_to_normalize].min())
normalized_test_df['track_genre'] = test_df['track_genre']
normalized_valid_df = (valid_df[features_to_normalize] - valid_df[features_to_normalize].min()) / (valid_df[features_to_normalize].max() - valid_df[features_to_normalize].min())
normalized_valid_df['track_genre'] = valid_df['track_genre']
X_valid = np.array(normalized_valid_df[features_to_normalize].values)
y_valid = np.array(normalized_valid_df['track_genre'].values)


model = Best_KNN_model(k=15, distance_metrics="cosine", features=features_to_normalize)
model.train(normalized_train_df)
model.split_data(validation_split=0.0, test_split=0.0)
model.validation_size = normalized_valid_df.shape[0]
evaluator = Best_KNN_evaluate(model)
validation_metrics = evaluator.evaluate(X_valid, y_valid)
evaluator.print_metrics(validation_metrics, "Validation")

"""
Validation Set Results:
accuracy: 0.2048
macro_p: 0.1988
macro_r: 0.2055
macro_f1: 0.1952
micro_p: 0.2048
micro_r: 0.2048
micro_f1: 0.2048
avg_time: 0.0190
"""

"""
Observations:

The accuracy of KNN model trained on second dataset is slightly lower than the first dataset.
Although, it is not significantly lower, but this shows that the first dataset is slightly better for predicting genre than second dataset.
"""



#-------------------------------------------------LINEAR-REGRESSION--------------------------------------------------------------------------
# Import Necessary libraries
import numpy as np # Matrix calculations
import pandas as pd # reading data
import matplotlib.pyplot as plt # plotting graphs

class LinearRegression:
    def __init__(self, learning_rate = 0.01, num_steps = 1000, lambda_regularization = 0):
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.lambda_regularization = lambda_regularization # lambda
        self.weights = None # beta1
        self.bias = None # beta0

    def split_data(self, data, validation_split=0.1, test_split=0.1):
        shuffled_df = data.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_size = int(validation_split * len(data))
        test_size = int(test_split * len(data))
        train_size = len(data) - (valid_size + test_size)

        X = data['x'].values.reshape(-1, 1)
        y = data['y'].values.reshape(-1, 1)

        self.X_train = shuffled_df.iloc[:train_size, :-1].values
        self.y_train = shuffled_df.iloc[:train_size, -1].values
        self.X_val = shuffled_df.iloc[train_size:train_size + valid_size, :-1].values
        self.y_val = shuffled_df.iloc[train_size:train_size + valid_size, -1].values
        self.X_test = shuffled_df.iloc[train_size + valid_size:, :-1].values
        self.y_test = shuffled_df.iloc[train_size + valid_size:, -1].values

    def train(self, X, y):
        self.fit(X, y)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.num_steps):
            y_predicted = np.dot(X, self.weights) + self.bias

            # Gradient descent
            dw = (1 / n_samples) * (np.dot(X.T, (y_predicted - y)) + self.lambda_regularization * self.weights)
            db = (1 / n_samples) * (np.sum(y_predicted - y))

            # backpropagate
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
    
class LR_EvaluationMetrics:
    def __init__(self, model):
        self.model = model

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def variance(self, y):
        return np.var(y)
    
    def std_dev(self, y):
        return np.std(y)
    
    def calculate_metrics(self):
        y_train_pred = self.model.predict(self.model.X_train)
        y_val_pred = self.model.predict(self.model.X_val)
        y_test_pred = self.model.predict(self.model.X_test)

        train_mse = self.mse(self.model.y_train, y_train_pred)
        valid_mse = self.mse(self.model.y_val, y_val_pred)
        test_mse = self.mse(self.model.y_test, y_test_pred)
        train_var = self.variance(y_train_pred)
        valid_var = self.variance(y_val_pred)
        test_var = self.variance(y_test_pred)
        train_sd = self.std_dev(y_train_pred)
        valid_sd = self.std_dev(y_val_pred)
        test_sd = self.std_dev(y_test_pred)

        return (train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd)
    
    def plot_without_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        plt.title('Training data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        y_pred = self.model.predict(self.model.X_train)

        plt.plot(self.model.X_train, y_pred, color='r', label='Fitted line')

        plt.title('Training data with Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_combined_graph(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, color='b', label='Train split', alpha=0.3)
        plt.scatter(self.model.X_val, self.model.y_val, color='r', label='Validation split', alpha=0.7)
        plt.scatter(self.model.X_test, self.model.y_test, color='y', label='Test split', alpha=0.7)

        y_pred = self.model.predict(self.model.X_train)

        plt.plot(self.model.X_train, y_pred, color='black', label='Fitted line')

        plt.title('Training data with Fitted Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

df = pd.read_csv("../../data/external/linreg.csv")

# Train the model
model = LinearRegression(learning_rate=0.01, num_steps=1000, lambda_regularization=0)
model.split_data(data = df)
model.train(model.X_train, model.y_train)

metrics = LR_EvaluationMetrics(model)
train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()

# print metrics
print("Train MSE:", train_mse)
print("Validation MSE:", valid_mse)
print("Test MSE:", test_mse)
print("Train Variance:", train_var)
print("Validation Variance:", valid_var)
print("Test Variance:", test_var)
print("Train SD:", train_sd)
print("Validation SD:", valid_sd)
print("Test SD:", test_sd)

"""
Train MSE: 0.37490540425081165
Validation MSE: 0.26539284847383726
Test MSE: 0.24162966564242144
Train Variance: 0.8109231739222448
Validation Variance: 0.6659623498811269
Test Variance: 0.911540114865996
Train SD: 0.900512728351046
Validation SD: 0.8160651627665079
Test SD: 0.954746099686192
"""

metrics.plot_without_regression_line()

metrics.plot_without_regression_line()

# Testing which learning rate works best

lr_list = [0.001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3]

valid_mse_list = []
valid_var_list = []
valid_sd_list = []

min_mse = float('inf')
best_lr = None

for i, lr in enumerate(lr_list):
    model = LinearRegression(learning_rate=lr, num_steps=1000, lambda_regularization=0)
    model.split_data(data = df)
    model.train(model.X_train, model.y_train)
    metrics = LR_EvaluationMetrics(model)
    train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()

    valid_mse_list.append(valid_mse)
    valid_var_list.append(valid_var)
    valid_sd_list.append(valid_sd)

    if valid_mse < min_mse:
        min_mse = valid_mse
        best_lr = lr

print(f"Best lr: {best_lr}")
print(f"Minimum MSE: {min_mse}")

plt.figure(figsize=(10, 6))
plt.plot(lr_list, valid_mse_list, marker='o', label='Validation MSE', color='r')
plt.plot(lr_list, valid_var_list, marker='s', label='Validation Variance', color='g')
plt.plot(lr_list, valid_sd_list, marker='^', label='Validation Standard Deviation', color='b')

plt.title('Learning Rate vs Validation Metrics')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Metric Value')
plt.xscale('log')  # Log scale for better visualization
plt.legend()
plt.grid(True)
plt.show()


metrics.plot_combined_graph()


#-------------------------------------------------POLYNOMIAL-REGRESSION--------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class PolynomialRegression:
    def __init__(self, k=2, learning_rate=0.01, num_steps=1000, regularization = 'none', lambda_regularization=0):
        self.k = k # degree
        self.learning_rate = learning_rate
        self.num_steps = num_steps
        self.regularization = regularization.lower()
        self.lambda_regularization = lambda_regularization
        self.weights = None
        self.bias = None

    def set_k(self, k):
        self.k = k

    def polynomial_features(self, X):
        poly_features = X
        for i in range(2, self.k + 1):
            poly_features = np.concatenate((poly_features, X ** i), axis=1)
        return poly_features

    def split_data(self, data, validation_split=0.1, test_split=0.1):
        shuffled_df = data.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_size = int(validation_split * len(data))
        test_size = int(test_split * len(data))
        train_size = len(data) - (valid_size + test_size)

        self.X_train = shuffled_df.iloc[:train_size, :-1].values
        self.y_train = shuffled_df.iloc[:train_size, -1].values
        self.X_val = shuffled_df.iloc[train_size:train_size + valid_size, :-1].values
        self.y_val = shuffled_df.iloc[train_size:train_size + valid_size, -1].values
        self.X_test = shuffled_df.iloc[train_size + valid_size:, :-1].values
        self.y_test = shuffled_df.iloc[train_size + valid_size:, -1].values

    def train(self, X, y, save_frames = False, path_to_image_folder = ".", save_steps=10, seed = None):
        self.fit(X, y, save_frames, path_to_image_folder, save_steps, seed)

    def fit(self, X, y, save_frames = False, path_to_image_folder = ".", save_steps = 10, seed = None):
        X_poly = self.polynomial_features(X)
        n_samples, n_features = X_poly.shape
        if seed == None:
            self.weights = np.zeros(n_features)
        else:
            np.random.seed(seed)
            self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0

        for i in range(self.num_steps):
            y_predicted = np.dot(X_poly, self.weights) + self.bias

            if self.regularization == 'none':
                dw = (1 / n_samples) * (np.dot(X_poly.T, (y_predicted - y)))
            elif self.regularization == 'l1':
                dw = (1 / n_samples) * (np.dot(X_poly.T, (y_predicted - y)) + self.lambda_regularization * np.sign(self.weights))
            elif self.regularization == 'l2':
                dw = (1 / n_samples) * (np.dot(X_poly.T, (y_predicted - y)) + self.lambda_regularization * self.weights)
            else:
                raise ValueError("Invalid regularization type. Should be none or l1 or l2 according to doc.")
            
            db = (1 / n_samples) * (np.sum(y_predicted - y))

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if save_frames == True:
                if i % save_steps == 0 or i == self.num_steps - 1:  # Save image every 10th step
                    self.plot_convergence(i, y, y_predicted, path_to_image_folder)

    def predict(self, X):
        X_poly = self.polynomial_features(X)
        return np.dot(X_poly, self.weights) + self.bias
    
    def plot_convergence(self, step, y_true, y_pred, path_to_image_folder = "."):
        fig = plt.figure(figsize=(12, 8))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])

        ax0 = plt.subplot(gs[0])
        ax0.scatter(self.X_train, self.y_train, color='b', alpha=0.3, label='Training data')
        sorted_indices = np.argsort(self.X_train[:, 0])
        X_sorted = self.X_train[sorted_indices]
        y_sorted_pred = y_pred[sorted_indices]
        ax0.plot(X_sorted, y_sorted_pred, color='black', label='Fitted curve')
        ax0.set_title(f'Fitted Curve at Step {step}')
        ax0.set_xlabel('x')
        ax0.set_ylabel('y')
        ax0.legend()
        ax0.grid(True)

        ax1 = plt.subplot(gs[1])
        mse = np.mean((y_true - y_pred) ** 2)
        ax1.bar(['MSE'], [mse], color='orange')
        ax1.set_title('MSE')
        ax1.set_ylim([0, max(1, mse)])

        ax2 = plt.subplot(gs[2])
        variance = np.var(y_pred)
        ax2.bar(['Variance'], [variance], color='green')
        ax2.set_title('Variance')
        ax2.set_ylim([0, max(1, variance)])

        ax3 = plt.subplot(gs[3])
        std_dev = np.std(y_pred)
        ax3.bar(['Standard Deviation'], [std_dev], color='red')
        ax3.set_title('Standard Deviation')
        ax3.set_ylim([0, max(1, std_dev)])

        plt.tight_layout()

        # Save the frame
        filename = f"{path_to_image_folder}frame_{step}.png"
        plt.savefig(filename)
        plt.close()
    
    def save_model(self, filename):
        np.savez(filename, weights=self.weights, bias=self.bias)

    def load_model(self, filename):
        model_data = np.load(filename)
        self.weights = model_data['weights']
        self.bias = model_data['bias']
    

class PR_EvaluationMetrics:
    def __init__(self, model):
        self.model = model 

    def mse(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def variance(self, y):
        return np.var(y)

    def std_dev(self, y):
        return np.std(y)

    def calculate_metrics(self):
        y_train_pred = self.model.predict(self.model.X_train)
        y_val_pred = self.model.predict(self.model.X_val)
        y_test_pred = self.model.predict(self.model.X_test)

        train_mse = self.mse(self.model.y_train, y_train_pred)
        valid_mse = self.mse(self.model.y_val, y_val_pred)
        test_mse = self.mse(self.model.y_test, y_test_pred)
        train_var = self.variance(y_train_pred)
        valid_var = self.variance(y_val_pred)
        test_var = self.variance(y_test_pred)
        train_sd = self.std_dev(y_train_pred)
        valid_sd = self.std_dev(y_val_pred)
        test_sd = self.std_dev(y_test_pred)

        return (train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd)
    
    def plot_without_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        plt.title('Training data')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_regression_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, marker='o', color='b', label='Training data', alpha=0.3)

        # Sorting the training data to get a smooth curve
        sorted_indices = np.argsort(self.X_train[:, 0])
        X_sorted = self.model.X_train[sorted_indices]
        
        # Generating predictions for the sorted X values
        y_pred = self.model.predict(X_sorted)
        
        # Plotting the fitted polynomial curve
        plt.plot(X_sorted, y_pred, color='black', label='Fitted curve')

        plt.title('Training, Validation, and Test Data with Fitted Polynomial Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_combined_graph_wo_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, color='b', label='Train split', alpha=0.3)
        plt.scatter(self.model.X_val, self.model.y_val, color='r', label='Validation split', alpha=0.7)
        plt.scatter(self.model.X_test, self.model.y_test, color='y', label='Test split', alpha=0.7)

        plt.title('Training, Validation, and Test Data with Fitted Polynomial Curve')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_combined_graph_w_line(self):
        plt.figure(figsize=(8, 8))
        plt.scatter(self.model.X_train, self.model.y_train, color='b', label='Train split', alpha=0.3)
        plt.scatter(self.model.X_val, self.model.y_val, color='r', label='Validation split', alpha=0.7)
        plt.scatter(self.model.X_test, self.model.y_test, color='y', label='Test split', alpha=0.7)

        # Sorting the training data to get a smooth curve
        sorted_indices = np.argsort(self.model.X_train[:, 0])
        X_sorted = self.model.X_train[sorted_indices]
        
        # Generating predictions for the sorted X values
        y_pred = self.model.predict(X_sorted)
        
        # Plotting the fitted polynomial curve
        plt.plot(X_sorted, y_pred, color='black', label='Fitted curve')

        plt.title(f'Training, Validation, and Test Data with Fitted Polynomial Curve (k = {self.model.k})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"../../assignments/1/figures/{self.model.regularization}_{self.model.k}.png")
        # plt.show()


df = pd.read_csv('../../data/external/linreg.csv')
poly_reg = PolynomialRegression(k=19, learning_rate=0.01, num_steps=10000)
poly_reg.split_data(df)
poly_reg.train(poly_reg.X_train, poly_reg.y_train)

metrics = PR_EvaluationMetrics(poly_reg)
train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()

# print metrics
print("Train MSE:", train_mse)
print("Validation MSE:", valid_mse)
print("Test MSE:", test_mse)
print("Train Variance:", train_var)
print("Validation Variance:", valid_var)
print("Test Variance:", test_var)
print("Train SD:", train_sd)
print("Validation SD:", valid_sd)
print("Test SD:", test_sd)

"""
Train MSE: 0.012361189488729434
Validation MSE: 0.010233938814714277
Test MSE: 0.012796955386455682
Train Variance: 1.2349697893293432
Validation Variance: 0.8068292012443099
Test Variance: 0.919424974096772
Train SD: 1.111291946038188
Validation SD: 0.8982367178223734
Test SD: 0.9588665048361904
"""

metrics.plot_combined_graph_w_line()

### Checking for best learning rate for Polynomial Regression

# Best lr: 0.1
# Minimum MSE: 0.04868740596031792

# Best learning rate in case of Polynomial Regression is higher than simple linear regression

lr_list = [0.001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.3]

valid_mse_list = []
valid_var_list = []
valid_sd_list = []

min_mse = float('inf')
best_lr = None

for i, lr in enumerate(lr_list):
    model = PolynomialRegression(k = 3, learning_rate=lr, num_steps=1000, lambda_regularization=0)
    model.split_data(data = df)
    model.train(model.X_train, model.y_train)
    metrics = PR_EvaluationMetrics(model)
    train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()

    valid_mse_list.append(valid_mse)
    valid_var_list.append(valid_var)
    valid_sd_list.append(valid_sd)

    if valid_mse < min_mse:
        min_mse = valid_mse
        best_lr = lr

print(f"Best lr: {best_lr}")
print(f"Minimum MSE: {min_mse}")

plt.figure(figsize=(10, 6))
plt.plot(lr_list, valid_mse_list, marker='o', label='Validation MSE', color='r')
plt.plot(lr_list, valid_var_list, marker='s', label='Validation Variance', color='g')
plt.plot(lr_list, valid_sd_list, marker='^', label='Validation Standard Deviation', color='b')

plt.title('Learning Rate vs Validation Metrics')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Metric Value')
plt.xscale('log')  # Log scale for better visualization
plt.legend()
plt.grid(True)
plt.show()

### Finding best k (degree) value in range 1 to 20 which minimizes MSE on test set

# Best k: 15

# Minimum MSE: 0.008329993315670487

k_values = list(i for i in range(1, 20, 1))
df = pd.read_csv("../../data/external/linreg.csv")

valid_mse_list = []
valid_var_list = []
valid_sd_list = []
train_mse_list = []
train_var_list = []
train_sd_list = []

min_mse = float('inf')
best_k = None

for k in k_values:
    model = PolynomialRegression(k = k, learning_rate=0.1, num_steps=10000, lambda_regularization=0)
    model.split_data(data = df)
    model.train(model.X_train, model.y_train)
    metrics = PR_EvaluationMetrics(model)
    train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()

    valid_mse_list.append(valid_mse)
    valid_var_list.append(valid_var)
    valid_sd_list.append(valid_sd)
    train_mse_list.append(train_mse)
    train_var_list.append(train_var)
    train_sd_list.append(train_sd)

    if valid_mse < min_mse:
        min_mse = valid_mse
        best_k = k
        model.save_model('best_model_k15_lr0.1.npz') # save the best model

print(f"Best k: {best_k}")
print(f"Minimum MSE: {min_mse}")

plt.figure(figsize=(10, 6))
plt.plot(k_values, valid_mse_list, marker='o', label='Validation MSE', color='r')
plt.plot(k_values, valid_var_list, marker='s', label='Validation Variance', color='g')
plt.plot(k_values, valid_sd_list, marker='^', label='Validation Standard Deviation', color='b')

plt.plot(k_values, train_mse_list, marker='s', label='Training MSE', color='r')
plt.plot(k_values, train_var_list, marker='o', label='Training Variance', color='g')
plt.plot(k_values, train_sd_list, marker='s', label='Training Standard Deviation', color='b')


plt.title('Learning Rate vs Validation Metrics')
plt.xlabel('Learning Rate (log scale)')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(True)
plt.show()

### Making GIF

# Reference: https://www.kaggle.com/discussions/general/501746
# without seed (i.e., 0 initialization)

df = pd.read_csv('../../data/external/linreg.csv')
poly_reg = PolynomialRegression(k=5, learning_rate=0.1, num_steps=500)
poly_reg.split_data(df)
poly_reg.fit(poly_reg.X_train, poly_reg.y_train, save_frames=True, path_to_image_folder="../../assignments/1/figures/frames_k3_dummy/", save_steps=100)

# metrics without any seed (0 initialization)
metrics = PR_EvaluationMetrics(poly_reg)
train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()

# print metrics
print("Train MSE:", train_mse)
print("Validation MSE:", valid_mse)
print("Test MSE:", test_mse)
print("Train Variance:", train_var)
print("Validation Variance:", valid_var)
print("Test Variance:", test_var)
print("Train SD:", train_sd)
print("Validation SD:", valid_sd)
print("Test SD:", test_sd)

"""
Train MSE: 0.05789603488757748
Validation MSE: 0.03200320207501341
Test MSE: 0.05389501789216936
Train Variance: 1.1511659980000502
Validation Variance: 0.7928366057503093
Test Variance: 1.0702873916707198
Train SD: 1.0729240411138388
Validation SD: 0.8904137272921556
Test SD: 1.0345469499596043
"""

# with seed = 42

df = pd.read_csv('../../data/external/linreg.csv')
poly_reg = PolynomialRegression(k=5, learning_rate=0.1, num_steps=500)
poly_reg.split_data(df)
poly_reg.fit(poly_reg.X_train, poly_reg.y_train, save_frames=True, path_to_image_folder="../../assignments/1/figures/frames_k5/", save_steps=5, seed=42)

metrics = PR_EvaluationMetrics(poly_reg)
train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()

# print metrics
print("Train MSE:", train_mse)
print("Validation MSE:", valid_mse)
print("Test MSE:", test_mse)
print("Train Variance:", train_var)
print("Validation Variance:", valid_var)
print("Test Variance:", test_var)
print("Train SD:", train_sd)
print("Validation SD:", valid_sd)
print("Test SD:", test_sd)

"""
Train MSE: 0.05790725301502266
Validation MSE: 0.031964620164098095
Test MSE: 0.05391394463509984
Train Variance: 1.1511521971701026
Validation Variance: 0.7925797503188459
Test Variance: 1.0702212802404791
Train SD: 1.0729176096840347
Validation SD: 0.8902694818530207
Test SD: 1.03451499759089
"""

# Few point(s) about random seed vs no seed:
# - Potential of overfitting: slightly lower validation MSE with seed initialization might suggest a marginally better fit (although almost similar)

# Image conversions
import imageio
from PIL import Image, ImageDraw, ImageFont

import glob
import cv2
import numpy as np
import re

def natural_sort_key(file_name):
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', file_name)]

# Initialize some settings
image_folder = "../../assignments/1/figures/frames_k3_dummy/"
output_gif_path = "../../assignments/1/figures/k3_dummy_seed42.gif"
duration_per_frame = 100  # milliseconds

# Collect all image paths
image_paths = glob.glob(f"{image_folder}*.png")
image_paths.sort(key=natural_sort_key)  # Sort the images to maintain sequence; adjust as needed

# Initialize an empty list to store the images
frames = []

# Loop through each image file to add text and append to frames
for image_path in image_paths:
    img = Image.open(image_path)

    # Reduce the frame size by 50%
    img = img.resize((int(img.width * 0.5), int(img.height * 0.5)))

    frames.append(img)

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
out = cv2.VideoWriter('animated_presentation.mp4', fourcc, 20.0, (int(img.width), int(img.height)))

# Loop through each image frame (assuming you have the frames in 'frames' list)
for img_pil in frames:
    # Convert PIL image to numpy array (OpenCV format)
    img_np = np.array(img_pil)

    # Convert RGB to BGR (OpenCV uses BGR instead of RGB)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Write frame to video
    out.write(img_bgr)

# Release the VideoWriter
out.release()

# Save frames as an animated GIF
frames[0].save(output_gif_path,
               save_all=True,
               append_images=frames[1:],
               duration=duration_per_frame,
               loop=0,
               optimize=True)


# Regularisation

"""
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
"""


df = pd.read_csv("../../data/external/regularisation.csv")
df.dropna()

# Without Regularisation:
"""
Observations:

1. As k increases, `train_mse` decreases almost monotonically.
2. However, the validation loss increases.
3. This shows Overfitting since model is learning more and more noises in training set. Hence, it's performance on validation set is degrading.
"""
k_values = []
for i in range(1, 30, 1):
    k_values.append(i)

results = [] # to store the evaluation results

poly_reg = PolynomialRegression(k = 1, regularization='none', num_steps=5000)
poly_reg.split_data(df)

for k in k_values:
    poly_reg.set_k(k)
    poly_reg.train(poly_reg.X_train, poly_reg.y_train)
    metrics = PR_EvaluationMetrics(poly_reg)
    metrics.plot_combined_graph_w_line()
    train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()
    results.append((k, train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd))

results_df = pd.DataFrame(results, columns=['k', 'train_mse', 'valid_mse', 'test_mse', 'train_var', 'valid_var', 'test_var', 'train_sd', 'valid_sd', 'test_sd'])
print(results_df.to_string(index = False))

"""
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
"""

# L1 Regularisation
"""
Observations:

1. As k increases, `train_mse` doesn't decrease that drastically as earlier with no regularization.
2. Also, `valid_loss` doesn't increase that drastically here in L1 regularization.
3. This shows that penalizing Loss function with L1 norm, helps model to overcome Overfitting issue.
"""

k_values = []
for i in range(1, 30, 1):
    k_values.append(i)

results = [] # to store the evaluation results

poly_reg = PolynomialRegression(k = 1, regularization='l1', lambda_regularization=0.5, num_steps=5000)
poly_reg.split_data(df)

for k in k_values:
    poly_reg.set_k(k)
    poly_reg.fit(poly_reg.X_train, poly_reg.y_train)
    metrics = PR_EvaluationMetrics(poly_reg)
    metrics.plot_combined_graph_w_line()
    train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()
    results.append((k, train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd))

results_df = pd.DataFrame(results, columns=['k', 'train_mse', 'valid_mse', 'test_mse', 'train_var', 'valid_var', 'test_var', 'train_sd', 'valid_sd', 'test_sd'])
print(results_df.to_string(index = False))

"""
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
"""

# L2 Regularization

"""
Observations:

1. As k increases, `train_mse` doesn't decrease that drastically as earlier with no regularization.
2. Also, `valid_loss` doesn't increase that drastically here in L1 regularization.
3. This shows that penalizing Loss function with L2 norm , also, helps model to overcome Overfitting issue. But this doesn't work as well as L1 regularization.
4. This could be becoz we only have one feature, and L1 penalty is proportional to absolute value of the magnitude of coefficients.
"""

k_values = []
for i in range(1, 21, 1):
    k_values.append(i)

results = [] # to store the evaluation results

poly_reg = PolynomialRegression(k = 1, regularization='l2', lambda_regularization=0.5, num_steps=5000)
poly_reg.split_data(df)

for k in k_values:
    poly_reg.set_k(k)
    poly_reg.fit(poly_reg.X_train, poly_reg.y_train)
    metrics = PR_EvaluationMetrics(poly_reg)
    metrics.plot_combined_graph_w_line()
    train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd = metrics.calculate_metrics()
    results.append((k, train_mse, valid_mse, test_mse, train_var, valid_var, test_var, train_sd, valid_sd, test_sd))

results_df = pd.DataFrame(results, columns=['k', 'train_mse', 'valid_mse', 'test_mse', 'train_var', 'valid_var', 'test_var', 'train_sd', 'valid_sd', 'test_sd'])
print(results_df.to_string(index = False))

"""
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
"""


#----------------------------------------------------------------------------------------------------------------------------------------------------