import numpy as np
import pandas as pd
from IPython.display import clear_output
import json

import tensorflow as tf

# Load the JSON file
with open('constants.json', 'r') as json_file:
    constants_data = json.load(json_file)
NUMERIC_COLUMNS = constants_data['NUMERIC_COLUMNS']
CHAMPION_AMOUNT = constants_data['CHAMPION_AMOUNT']
TEAM_AMOUNT = constants_data['TEAM_AMOUNT']

# Load dataset.
dftrain = pd.read_csv('LCK_training_data.csv') # training data
dfeval = pd.read_csv('LCK_evaluation_data.csv') # testing data
y_train = dftrain.pop('result')
y_eval = dfeval.pop('result')
# Pop result from NUMERIC_COLUMNS since it is not a feature
NUMERIC_COLUMNS.pop(-1)

# Define the categorical columns for champions and teams
categorical_columns = ['Champion_1_Number', 'Champion_2_Number', 'Champion_3_Number', 'Champion_4_Number', 'Champion_5_Number',
                        'Champion_Banned_Number1', 'Champion_Banned_Number2', 'Champion_Banned_Number3', 'Champion_Banned_Number4', 'Champion_Banned_Number5',
                        'Team_Number', 'Opponent_Team_Number',
                        'opponent_Champion_1_Number', 'opponent_Champion_2_Number', 'opponent_Champion_3_Number', 'opponent_Champion_4_Number', 'opponent_Champion_5_Number',
                        'opponent_Champion_Banned_Number1', 'opponent_Champion_Banned_Number2', 'opponent_Champion_Banned_Number3', 'opponent_Champion_Banned_Number4', 'opponent_Champion_Banned_Number5']

# Create feature columns
feature_columns = []

for feature_name in NUMERIC_COLUMNS:
    if feature_name in categorical_columns:
        # For categorical columns, use embedding_column
        if feature_name != 'Team_Number' and feature_name != 'Opponent_Team_Number':
            num_buckets = CHAMPION_AMOUNT
        else:
            num_buckets = TEAM_AMOUNT
        embedding_column = tf.feature_column.embedding_column(
            tf.feature_column.categorical_column_with_identity(feature_name, num_buckets),
            dimension=20  
        )
        feature_columns.append(embedding_column)
    else:
        # For numerical columns, use numeric_column
        feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

# Input function
def make_input_fn(data_df, label_df, num_epochs=15, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for the number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# Create the model
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# Train the model
linear_est.train(train_input_fn)  # train
result = linear_est.evaluate(eval_input_fn)  # get model metrics/stats by testing on testing data

clear_output()  # clears console output
print(result['accuracy'])

# Predictions
result = list(linear_est.predict(eval_input_fn))
