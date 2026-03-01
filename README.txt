Label value 1 -> forward
Label value 2 -> backward
Label value 3 -> left
Label value 4 -> right

sensor_data_merge.py: merge all post-processed sensor data from a given folder into one dataframe
forward_processing.py: preprocess forward fall sensor data
backward_processing.py: preprocess backward fall sensor data
left_processing.py: preprocess left fall sensor data
right_processing.py: preprocess right fall sensor data
sensor_model_train.py: create TCN model and use training data to train model and evaluate model accuracy