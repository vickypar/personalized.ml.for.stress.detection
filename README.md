# Personalized Machine Learning Benchmarking for Stress Detection

This project is my Thesis for the fulfilment of MSc Data and Web Science, School of Informatics, Aristotle University of Thessaloniki.

## Table of Contents

[0. Abstract](https://github.com/vickypar/personalized_ml_for_stress_prediction#0-abstract)

[1. Installation](https://github.com/vickypar/personalized_ml_for_stress_prediction#1-installation)

[2. Data](https://github.com/vickypar/personalized_ml_for_stress_prediction#2-data)

[3. Methodology and Models](https://github.com/vickypar/personalized_ml_for_stress_prediction#3-methodology-and-models)

[4. Evaluation](https://github.com/vickypar/personalized_ml_for_stress_prediction#4-evaluation)

## 0. Abstract
This thesis investigates the feasibility of stress detection using physiological measurements captured by smartwatches and personalized machine learning and deep learning techniques. Smartwatches offer unobtrusive stress tracking in real-time by measuring stress indicators such as heart rate, heart rate variability, skin conductance, physical activity, and sleep quality. The early detection of stress can prevent long-term negative consequences for physical and mental health. We group users based on various stress-indicative attributes such as exercising and sleep staging and train personalized machine learning models for each group or use multitask learning (MTL). MTL deep neural networks with shared layers and task-specific layers are used to exploit similarities between users. We also evaluate the performance of "fuzzy" clustering, where users belong to all clusters with a membership degree, and compare our approach to generic and user-based models. We evaluated our methods on four datasets, including two lab and two in-the-wild datasets, to understand how our approach would work in real-world settings. User-based models perform better than generic ones, emphasizing the importance of personalization. The best results are obtained using a model that groups users based on multiple attributes, with up to a 0.9960 f1-score in lab setting datasets and up to a 0.8131 f1-score in the datasets collected in-the-wild. Our results demonstrate high performance in both lab and in-the-wild settings, but also highlight the challenges of working with everyday data. Overall, this work sheds light on the potential of personalized machine learning in healthcare research beyond stress detection, towards analytics for user health and well-being in general.

## 1. Installation
The code requires Python versions of 3.* and general libraries available through the Anaconda package.

## 2. Data
In our study, we evaluated stress detection using four datasets, two of which were collected in controlled laboratory environments, while the other two were collected in-the-wild, where participants carried out their daily activities as usual. In particular:

### 2.1 WESAD
The [WESAD](https://archive.ics.uci.edu/ml/datasets/WESAD+%28Wearable+Stress+and+Affect+Detection%29) dataset is a public dataset collected in a lab setting for the purpose of detecting stress and affect using wearable devices. 15 participants, with 12 of them being male, participated in the study and were monitored using a wrist-worn Empatica E4 device that recorded heart rate, skin conductance, body temperature, and three-axis acceleration. The participants also provided self-reported information such as age, gender, dominant hand, coffee intake, exercise, smoking, and illness status. The experiment was divided into four conditions: baseline, amusement, stress, and meditation. The baseline aimed to create a neutral affective state, while the amusement condition involved watching funny videos. The stress condition included a public speaking and mental arithmetic task as part of the Trier Social Stress Test. Finally, the meditation condition was a guided meditation after the stress and amusement conditions. In the analysis, the "non-stress" class was created by combining "meditation" and "baseline" and "amusement" was not considered.

### 2.2 ADARP
The ``[Alcohol and Drug Abuse Research Program](https://zenodo.org/record/6640290#.Y-zvm61Bw7d)'' dataset was collected from 11 individuals (10 of whom were female) suffering from alcohol use disorder, in an in-the-wild setting. The study aimed to examine the connection between daily experiences of patients diagnosed with alcohol disorder and physiological markers of stress. The data was collected continuously for 14 days using an Empatica E4 wearable device, and it included heart rate, skin conductance, body temperature, and three-axis acceleration. Additionally, the participants completed daily self-reported surveys on emotions, alcohol cravings, and stress and participated in structured interviews to validate their self-reported stress and alcohol use. 
 
### 2.3 SWELL-KW
The [SWELL-KW](https://easy.dans.knaw.nl/ui/datasets/id/easy-dataset:58624/tab/2) dataset was collected from 25 participants (8 females, 17 males) performing typical knowledge work tasks, such as writing reports, making presentations, reading emails, and searching for information. The experiment consisted of three one-hour blocks (one neutral and two stressful) with relaxation breaks in between. More specifically, in the neutral (no-stress) condition the participants completed a task without time constraint. In the stressful conditions, they had to complete the same task in 2/3 of the time taken in the neutral condition (time pressure) and answer 8 emails while completing the task (interruption). Recorded data includes physiological data, such as heart rate, heart rate variability, and skin conductance measured using a Mobi device (TMSI). Participants also answered some questionnaires which provide information about their age, gender, dominant hand, occupation, and whether they wear glasses, have heart disease, or take medicine. They were also asked about their recent smoking habits, coffee and alcohol consumption, exercise patterns, and stress levels. In addition, they filled in the ``Internal Control Index'' questionnaire to measure their Locus of Control, which refers to their belief in their ability to control events that impact them. This may impact how they perceive and respond to stress.

### 2.4 LifeSnaps
The [LifeSnaps](https://zenodo.org/record/6832242?token=eyJhbGciOiJIUzUxMiIsImV4cCI6MTY4OTI4NTU5OSwiaWF0IjoxNjU3NzkwNDAxfQ.eyJkYXRhIjp7InJlY2lkIjo2ODMyMjQyfSwiaWQiOjI0NjU2LCJybmQiOiIwMDI3MjcwMiJ9.0dOhspFs0wGL-UWIKIBxuhN41y7jGx5xoNj-KqtbMRIZ6IZtFmVPdx5nU1SDGu94Dyt2LTPeqCfrU-A2XLVgIw#.Y-z1W61Bw7e) dataset is a publicly available, multi-modal dataset collected in-the-wild using Fitbit Sense smartwatches. It encompasses a vast amount of physiological data, including physical activity, sleep, heart rate, temperature, and more, gathered continuously for 4 months from 71 participants (42 males and 29 females) residing in four different countries. Participants were encouraged to wear their Fitbit as frequently as possible and to continue their regular activities. In addition to physiological data, participants reported their mood and location through a mobile application. They also completed a survey that evaluated them based on the Big Five Personality Traits model, which consists of Conscientiousness, Extraversion, Agreeableness, Emotional Stability, and Intellect. Personality has been demonstrated to influence exposure to stressors and the stressor experience, with higher levels of Conscientiousness, Agreeableness, and Extraversion being linked to lower stress levels. 

## 3. Methodology and Models

### 3.1 Generic Models
### 3.2 User-based Models 
### 3.3 Single-Attribute-based Modes
### 3.3 Multi-Attribute-based Modes
### 3.3 Fuzzy-based Modes

## 4. Evaluation
