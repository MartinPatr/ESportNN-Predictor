# ESport Match Predictor

## Overview

This project aims to predict the outcomes of professional esports matches, specifically focusing on the game League of Legends (LoL). The prediction is based on various factors such as team compositions, champion picks, and historical performance. The project utilizes machine learning techniques and TensorFlow to build a predictive model.

## Files

### 1. `filter_df.py`
- Reads and processes raw esports data from '2023_LoL_esports.csv'.
- Extracts relevant information, including team and champion picks.
- Creates rolling averages for team and opponent statistics.
- Maps champions and teams to numerical values for model compatibility.
- Splits the dataset into training and evaluation sets.
- Outputs processed data to 'LCK_training_data.csv' and 'LCK_evaluation_data.csv'. 


### 2. `ESportModel.py`
- Builds, trains, and evaluates a machine learning model using TensorFlow.
- Utilizes the processed data from 'LCK_training_data.csv' and 'LCK_evaluation_data.csv'.
- Implements a linear classifier to predict match outcomes.

## Dependencies
- Python
- pandas
- Tensorflow
- scikit-learn
- jupyter (optional)