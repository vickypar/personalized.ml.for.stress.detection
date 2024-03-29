# Personalized Machine Learning Benchmarking for Stress Detection

## Table of Contents

[0. Abstract](https://github.com/vickypar/personalized_ml_for_stress_prediction#0-abstract)

[1. Installation](https://github.com/vickypar/personalized_ml_for_stress_prediction#1-installation)

[2. Data](https://github.com/vickypar/personalized_ml_for_stress_prediction#2-data)

[3. Methodology and Models](https://github.com/vickypar/personalized_ml_for_stress_prediction#3-methodology-and-models)

[4. Evaluation](https://github.com/vickypar/personalized_ml_for_stress_prediction#4-evaluation)

## 0. Abstract
This work investigates the feasibility of **stress detection** using **physiological measurements** captured by smartwatches and personalized machine learning and deep learning techniques. Smartwatches offer unobtrusive stress tracking in real-time by measuring stress indicators such as heart rate, heart rate variability, skin conductance, physical activity, and sleep quality. The early detection of stress can prevent long-term negative consequences for physical and mental health. We group users based on various stress-indicative attributes such as exercising and sleep staging and train personalized machine learning models for each group or use **multitask learning (MTL)**. MTL deep neural networks with shared layers and task-specific layers are used to exploit similarities between users. We also evaluate the performance of "fuzzy" clustering, where users belong to all clusters with a membership degree, and compare our approach to generic and user-based models. We evaluated our methods on four datasets, including two lab and two in-the-wild datasets, to understand how our approach would work in real-world settings. User-based models perform better than generic ones, emphasizing the importance of personalization. The best results are obtained using a model that groups users based on multiple attributes, with up to a **0.9960** f1-score in lab setting datasets and up to a **0.8131** f1-score in the datasets collected in-the-wild. Our results demonstrate high performance in both lab and in-the-wild settings, but also highlight the challenges of working with everyday data. Overall, this work sheds light on the potential of personalized machine learning in healthcare research beyond stress detection, towards analytics for user health and well-being in general.

## 1. Installation
The code requires **Python** versions of 3.* and general libraries available through the Anaconda package.

## 2. Data
In our study, we evaluated stress detection using **four datasets**, two of which were collected in controlled **laboratory** environments, while the other two were collected **in-the-wild**, where participants carried out their daily activities as usual. In particular:

### 2.1 WESAD
The [WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) dataset is a public dataset collected in a lab setting for the purpose of detecting stress and affect using wearable devices. 15 participants, with 12 of them being male, participated in the study and were monitored using a wrist-worn Empatica E4 device that recorded **heart rate**, **skin conductance**, **body temperature**, and **three-axis acceleration**. The participants also provided self-reported information such as age, gender, dominant hand, coffee intake, exercise, smoking, and illness status. The experiment was divided into four conditions: baseline, amusement, stress, and meditation. The baseline aimed to create a neutral affective state, while the amusement condition involved watching funny videos. The stress condition included a public speaking and mental arithmetic task as part of the Trier Social Stress Test. Finally, the meditation condition was a guided meditation after the stress and amusement conditions. In the analysis, the "non-stress" class was created by combining "meditation" and "baseline" and "amusement" was not considered.

### 2.2 ADARP
The ``[Alcohol and Drug Abuse Research Program](https://zenodo.org/record/6640290#.Y-zvm61Bw7d)'' dataset was collected from 11 individuals (10 of whom were female) suffering from alcohol use disorder, in an in-the-wild setting. The study aimed to examine the connection between daily experiences of patients diagnosed with alcohol disorder and physiological markers of stress. The data was collected continuously for 14 days using an Empatica E4 wearable device, and it included **heart rate**, **skin conductance**, **body temperature**, and **three-axis acceleration**. Additionally, the participants completed daily self-reported surveys on emotions, alcohol cravings, and stress and participated in structured interviews to validate their self-reported stress and alcohol use. 
 
### 2.3 SWELL-KW
The [SWELL-KW](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624/tab/2) dataset was collected from 25 participants (8 females, 17 males) performing typical knowledge work tasks, such as writing reports, making presentations, reading emails, and searching for information. The experiment consisted of three one-hour blocks (one neutral and two stressful) with relaxation breaks in between. Recorded data includes physiological data, such as **heart rate**, **heart rate variability**, and **skin conductance** measured using a Mobi device (TMSI). Participants also answered some questionnaires which provide information about their age, gender, dominant hand, occupation, and whether they wear glasses, have heart disease, or take medicine. They were also asked about their recent smoking habits, coffee and alcohol consumption, exercise patterns, and stress levels. In addition, they filled in the "Internal Control Index" questionnaire to measure their Locus of Control, which refers to their belief in their ability to control events that impact them. This may impact how they perceive and respond to stress.

### 2.4 LifeSnaps
The [LifeSnaps](https://zenodo.org/record/6832242?token=eyJhbGciOiJIUzUxMiIsImV4cCI6MTY4OTI4NTU5OSwiaWF0IjoxNjU3NzkwNDAxfQ.eyJkYXRhIjp7InJlY2lkIjo2ODMyMjQyfSwiaWQiOjI0NjU2LCJybmQiOiIwMDI3MjcwMiJ9.0dOhspFs0wGL-UWIKIBxuhN41y7jGx5xoNj-KqtbMRIZ6IZtFmVPdx5nU1SDGu94Dyt2LTPeqCfrU-A2XLVgIw#.Y-z1W61Bw7e) dataset is a publicly available, multi-modal dataset collected in-the-wild using Fitbit Sense smartwatches. It encompasses a vast amount of physiological data, including **physical activity**, **sleep**, **heart rate**, **temperature**, and more, gathered continuously for 4 months from 71 participants (42 males and 29 females) residing in four different countries. Participants were encouraged to wear their Fitbit as frequently as possible and to continue their regular activities. In addition to physiological data, participants reported their mood and location through a mobile application. They also completed a survey that evaluated them based on the Big Five Personality Traits model, which consists of Conscientiousness, Extraversion, Agreeableness, Emotional Stability, and Intellect. Personality has been demonstrated to influence exposure to stressors and the stressor experience, with higher levels of Conscientiousness, Agreeableness, and Extraversion being linked to lower stress levels. 

## 3. Methodology and Models
We propose five techniques to accurately detect stress. Each approach includes a **machine-learning** and a **deep learning** equivalent which are visualized below.

![model_definitions](https://user-images.githubusercontent.com/95586847/219072159-7f8a22ee-6317-4ba5-ad44-f2aed9bdc95b.png)

### 3.1 Generic Models
The first and simplest approach is a person-independent model where a single function is learned using all available data. The machine learning approach uses various algorithms, such as linear algorithms (**Linear Discriminant Analysis, SVM with Linear Kernel, Logistic Regression, Ridge Classifier**), non-linear algorithms (**Decision Tree, Naive Bayes, K Nearest Neighbors, Quadratic Discriminant Analysis**), and boosting algorithms (**Random Forest, Extra Trees, AdaBoost, Gradient Boosting, Light Gradient Boosting Machine**), from the Python libraries [**scikit-learn**](https://scikit-learn.org/stable/) and [**Pycaret**](https://pycaret.gitbook.io/docs/). Subsequently, the best algorithm is selected based on F1-Score. To consider the effect of participant ID, two models are built for each dataset, one with participant ID as a feature and one without. In the deep learning approach, a generic binary classification neural network is trained for each dataset using all available data with the aid of [Tensorflow](https://www.tensorflow.org/overview) library. 

### 3.2 User-based Models 
However, a one-size-fits-all model is not able to capture the differences between individuals. To this end, we employ personalization by building a separate model for each individual. The machine-learning approach involves building **personalized models** for each participant using the same machine learning algorithms as in the previous approach, and train them on each participant's data. The best algorithm is then selected for each individual based on its performance. The evaluation of this approach is done by calculating the mean F1-Score across all participants.

Correspondingly, the deep learning approach adopts the use of **multitask learning (MTL)** to train a binary classification neural network for each dataset. MTL is a type of transfer learning where different models are trained simultaneously on related tasks by sharing information through similarity constraints. In our work, we apply MTL with neural networks and in this approach we treat the **detection of stress for a single participant as a task**. The architecture includes shared hidden layers trained on data from all users, connected to smaller task-specific layers unique to each individual. In this way, this approach not only accounts for differences between users, but also takes into account their similarities. 

### 3.3 Single-Attribute-based Models
The "User-based" approach necessitates a considerable amount of data per individual since lack of data increases the risk of overfitting and poses difficulties in incorporating new users, resulting in the "cold-start" problem. Consequently, this approach groups participants based on their **personality** since it influences stress. In particular, the available features relevant to the personality of each participant are preprocessed using Label Encoding and Standard Scaler. The number of groups (k) is determined using the **Elbow Method** and **Silhouette Score** and the users are grouped using the **K-means algorithm**. After grouping, a machine learning and a deep learning approach are used to make predictions. In this way, when new users appear, their personality is used to assign them to the appropriate group and predictions are made according to the pre-trained model for that group.

The first approach involves training a variety of machine learning algorithms on the data of each group and select the algorithm that performs best for each group. We use the same algorithms as in the generic method, and we evaluate the approach by computing the mean F1-Score across all groups. In the deep learning approach, we utilize MTL in a similar manner as before, but with a distinct approach. The difference is that we treat the **detection of stress for a single personality group as a task**. 

### 3.3 Multi-Attribute-based Models
Obtaining personality information from participants requires extra effort (such as completing questionnaires), which can be tedious and time-consuming. However, personality is not the only factor impacting stress. Studies show that **caffeine** intake can lead to higher stress levels through the release of stress hormones and symptoms. On the contrary, **physical activity** acts as a stress reliever while stress and **sleep** have a reciprocal relationship; stress can cause insomnia, and lack of sleep leads to higher cortisol levels and fatigue, which contribute to greater stress levels.

Thus, in this approach, we aim to categorize individuals using these features when they are available as well as information about their age, gender, personality, etc. Apart from this, we group all physiological features by user and calculate for each one the **mean** value, the **minimum** value, as well as the **standard deviation**. Both additional features and these statistics are then used for user-grouping. As a preprocessing step, we apply **Label Encoder** to categorical features and scale numerical features using **"MinMaxScaler"**. If there are many features, we perform **Principal Component Analysis (PCA)** to avoid the curse of dimensionality. The number of groups (k) is selected using the **Elbow Method** and **Silhouette Score**, and users are grouped using the **K-means algorithm**. After grouping, two approaches are followed.

In the first approach, we train a diverse range of machine learning algorithms on each group's data and choose the one that performs best for each group. The algorithms tested are consistent with those used in the generic approach. We evaluate this approach by calculating the average F1-Score across all groups. Similar to the previous method, the deep learning approach uses MTL. The difference in this approach lies in treating the **detection of stress for a multi-attribute-group as a task**. 

### 3.3 Fuzzy-based Models
Finally, in order to address the possibility of users belonging to multiple clusters with varying membership degrees, we employ **fuzzy** clustering using the **Fuzzy C-means (FCM)** algorithm. In this approach, the participants are assigned **membership degrees** for each cluster, instead of being assigned to discrete clusters as in K-means. This means that each user has a degree of belonging to each cluster, ranging from 0 to 1, and the sum of the degrees of each user must be 1. The features used for clustering and their preprocessing are the same as in the "Multi-Attribute-based Models", with FCM replacing K-means as the user grouping algorithm.

The application of fuzzy clustering is illustrated in the Figures below which display the membership degrees of users from one of the datasets, both numerically and visually through color intensity. For instance, the participant with ID "0" has a 1.9% membership to cluster 0, 7.9% to cluster 1, 56% to cluster 2, etc. To implement the machine learning and deep learning approaches, the instances of each user are split based on their membership degrees to each cluster. This means that 1.9% of participant's with ID "0" instances will be assigned to cluster's 0 data, 56% to cluster's 2 data, and so on.
Next, two methods are applied to the resulting groups.

<div align="center">
<table>
  <tr>
    <td align="center"><img src="https://user-images.githubusercontent.com/95586847/219320720-746cf624-eda8-43e1-9689-ef9ace706be9.png" width="400"></td>
    <td align="center"><img src="https://user-images.githubusercontent.com/95586847/219320851-717ad1ba-2b55-4024-ae58-2377d394ddc6.png" width="400"></td>
  </tr>
</table>
</div>

The first approach involves training multiple machine learning algorithms on the data of each group and select the one that performs best. The algorithms tested are the same as those used in the generic approach. We evaluate the approach by computing the average F1-Score across all groups. In the deep learning approach, we implement MTL and treat the **detection of stress in a fuzzy group as a task**. 

## 4. Evaluation
The **"F1-Score"** was selected as the most appropriate evaluation metric for the stress detection models, as two out of the four datasets are imbalanced (ADARP and LifeSnaps). Thus, the following barplots present the F1-Score of the machine learning and the deep learning approaches.

<p align="center">
 <img src="https://user-images.githubusercontent.com/95586847/219374497-2fc131c1-ce57-4b3b-9adf-5466b2755778.png" width="800" align="center">
</p>

<p align="center">
 <img src="https://user-images.githubusercontent.com/95586847/219374943-ba67bf30-cf1b-44a4-be8d-612d4d8fc262.png" width="800" align="center">
</p>

The following points summarize the key findings of this study.

- **The Importance of Personalization.** The results showed that the user-based models outperformed the generic ones, confirming the crucial role of personalization in stress detection.
- **Superior Performance of Multi-Attribute-Based Models.** Our results indicated that the multi-attribute-based models yielded the best performance, both in the case of machine learning and in the case of deep learning approaches. The only exception was the WESAD dataset, where the generic model with participant ID achieved the highest F1-Score (0.9984), with a negligible 0.24% difference from the multi-attribute splitting performance.
- **The Benefits of Multitask Learning.** In the deep learning approach, the "MTL Multi-Attribute-Group-As-Task" improved the F1-Score by **0.88-158.7%** compared to the "Single Task Learning" models.
- **The Impact of the Number of Participants.** Our results showed that the improvement of personalized models’ performance over the generic ones was dramatically higher in the LifeSnaps dataset compared to the other three datasets. This was probably due to the fact that this dataset had a larger number of participants (71), which facilitated the grouping of similar individuals.
- **The Impact of Dataset Size.** The results indicated that the machine learning approaches performed better than the deep learning ones, which was attributed to the limited size of the datasets. 
- **The Influence of Data Collection Environment.** The results showed that the performance of the stress detection system was higher in datasets that were collected in a lab setting than in the ones that have been created in a real-world environment (in-the-wild). This was expected as stress detection is more challenging in uncontrolled environments where individuals' movements are unrestricted and their context is unknown, and unsupervised participants may wear smartwatches improperly, leading to tampered measurements.
