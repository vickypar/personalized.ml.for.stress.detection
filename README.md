# Personalized Machine Learning Benchmarking for Stress Detection

This project is my Thesis for the fulfilment of MSc Data and Web Science, School of Informatics, Aristotle University of Thessaloniki.

## Table of Contents

[0. Abstract](https://github.com/vickypar/personalized_ml_for_stress_prediction#0-abstract)

[1. Installation](https://github.com/vickypar/personalized_ml_for_stress_prediction#1-installation)

[2. Data](https://github.com/vickypar/personalized_ml_for_stress_prediction#2-data)

[3. Methodology & Models](https://github.com/vickypar/personalized_ml_for_stress_prediction#3-methodology-&-models)

[4. Evaluation](https://github.com/vickypar/personalized_ml_for_stress_prediction#4-evaluation)

## 0. Abstract
This thesis investigates the feasibility of stress detection using physiological measurements captured by smartwatches and personalized machine learning and deep learning techniques. Smartwatches offer unobtrusive stress tracking in real-time by measuring stress indicators such as heart rate, heart rate variability, skin conductance, physical activity, and sleep quality. The early detection of stress can prevent long-term negative consequences for physical and mental health. We group users based on various stress-indicative attributes such as exercising and sleep staging and train personalized machine learning models for each group or use multitask learning (MTL). MTL deep neural networks with shared layers and task-specific layers are used to exploit similarities between users. We also evaluate the performance of "fuzzy" clustering, where users belong to all clusters with a membership degree, and compare our approach to generic and user-based models. We evaluated our methods on four datasets, including two lab and two in-the-wild datasets, to understand how our approach would work in real-world settings. User-based models perform better than generic ones, emphasizing the importance of personalization. The best results are obtained using a model that groups users based on multiple attributes, with up to a 0.9960 f1-score in lab setting datasets and up to a 0.8131 f1-score in the datasets collected in-the-wild. Our results demonstrate high performance in both lab and in-the-wild settings, but also highlight the challenges of working with everyday data. Overall, this work sheds light on the potential of personalized machine learning in healthcare research beyond stress detection, towards analytics for user health and well-being in general.

## 1. Data
## 2. Installation
## 3. Methodology & Models

### 3.1 Generic Models
### 3.2 User-based Models 
### 3.3 Single-Attribute-based Modes
### 3.3 Multi-Attribute-based Modes
### 3.3 Fuzzy-based Modes

## 4. Evaluation
