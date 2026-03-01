import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os


print("TensorFlow version:", tf.__version__)

# Shape example: [1 sample, 6 time steps, 2 channels]
X_np = np.array([
    [
        [0.1, 1.0],   # t1
        [0.2, 0.9],   # t2
        [0.3, 0.8],   # t3
        [0.4, 0.7],   # t4
        [0.5, 0.6],   # t5
        [0.6, 0.5],   # t6
    ],
     [
        [0.1, 1.0],   # t1
        [0.2, 0.9],   # t2
        [0.3, 0.8],   # t3
        [0.4, 0.7],   # t4
        [0.5, 0.6],   # t5
        [0.6, 0.5],   # t6
    ],
])

print("Shape:", X_np.shape)

NUM_CLASSES = 5
TIME_STEPS = 10
NUM_CHANNELS = 6
STRIDE_STEPS = 2

#N_SAMPLES = 524  # total samples

# Example synthetic IMU-like data
#rng = np.random.default_rng(42)
#X = rng.normal(size=(N_SAMPLES, TIME_STEPS, NUM_CHANNELS)).astype("float32")
#y = rng.integers(low=0, high=NUM_CLASSES, size=(N_SAMPLES,), dtype=np.int32)
#print("X shape:", X.shape)  # (N, 20, 6)
#print("y shape:", y.shape)  # (N,)

def generate_sample_from_df(df, X_in, y_in):
# In general,
# Output label use forward if label 1 shows in 18+ out of 20 steps
# Output label use backward if label 2 shows in 18+ out of 20 steps
# Output label use left if label 3 shows in 18+ out of 20 steps
# Output label use right if label 4 shows in 18+ out of 20 steps
    window_size = TIME_STEPS
    stride = STRIDE_STEPS
    t = 0
    T_total = df.shape[0]  # returns N of 3D array with shape (N, 20, 6)
    print(T_total)

    while t + window_size <= T_total:
        window = df[t : t + window_size]  # shape (20, 7 (including label))
        x_window = window.iloc[:, :6] 
        y_array = window.iloc[:, 6].to_numpy()
        low_threshold = window_size*0.7
        low = int(low_threshold)
    
        # Accumulate value of "window_size" labels
        count_1 = np.sum(y_array == 1)
        count_2 = np.sum(y_array == 2)
        count_3 = np.sum(y_array == 3)
        count_4 = np.sum(y_array == 4)
    
        label = 0
      
        if count_1 >= low and count_1 <= window_size:
            label = 1
        if count_2 >= low and count_2 <= window_size:
            label = 2
        if count_3 >= low and count_3 <= window_size:
            label = 3
        if count_4 >= low and count_4 <= window_size:
            label = 4

        window_3d = np.expand_dims(x_window, axis=0)
        if window_3d.shape == (1, TIME_STEPS, 6):
            X_in = np.concatenate([X_in, window_3d], axis=0)
            # Both inputs to concatenate() have to be array type.    
            y_1d = np.array([label])
            #print(y_1d, t)
            y_in = np.concatenate([y_in, y_1d], axis=0)

        t += stride

    # Don't know why X first entry is empty, remove it.
    X_in = X_in[1:, :, :]
    return X_in, y_in
 

# create training data
######### Generate training data #################

# Define the root directory where your subfolders are located
# Replace 'path/to/your/root_folder' with your actual path
root_directory = 'data_training'

# Use glob to find all files matching a pattern in the root directory and all subdirectories
# For CSV files:
file_paths = glob.glob(os.path.join(root_directory, '**/*.csv'), recursive=True)

# Create a list of DataFrames using a list comprehension
# Use the appropriate read function (pd.read_csv, pd.read_excel, etc.)
list_of_dfs = [pd.read_csv(file) for file in file_paths]

# Combine all DataFrames in the list into a single DataFrame
# ignore_index=True resets the row indices for the combined DataFrame
combined_df = pd.concat(list_of_dfs, ignore_index=True)

# Optional: Add a column to identify the source file
# combined_df['source_file'] = [os.path.basename(f) for f in file_paths for _ in range(pd.read_csv(f).shape[0])] # This requires careful handling of row counts

# Display the resulting DataFrame
print(combined_df.head()) 

shape = (1, TIME_STEPS, 6)
X_train = np.empty(shape)
y_train = np.empty(0)

X_train, y_train = generate_sample_from_df(combined_df, X_train, y_train)

print(X_train.shape)
print(y_train.shape)


# Load Evalation data
root_directory = 'data_eval'

# Use glob to find all files matching a pattern in the root directory and all subdirectories
# For CSV files:
file_paths = glob.glob(os.path.join(root_directory, '**/*.csv'), recursive=True)

# Create a list of DataFrames using a list comprehension
# Use the appropriate read function (pd.read_csv, pd.read_excel, etc.)
list_of_dfs = [pd.read_csv(file) for file in file_paths]

# Combine all DataFrames in the list into a single DataFrame
# ignore_index=True resets the row indices for the combined DataFrame
combined_df = pd.concat(list_of_dfs, ignore_index=True)
# Display the resulting DataFrame
print(combined_df.head()) 
print(combined_df.shape)

shape = (1, TIME_STEPS, 6)
X_val = np.empty(shape)
y_val = np.empty(0)

X_val, y_val = generate_sample_from_df(combined_df, X_val, y_val)

print(X_val.shape)
print(y_val.shape)
print(y_val)
#quit()


def tcn_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.2):
    """
    One residual TCN block:
    Conv1D(causal, dilation) → BN → ReLU → Dropout → Conv1D → BN → ReLU → Dropout + Residual
    """
    x_in = x  # for residual

    # First conv
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",         # important for temporal causality
        dilation_rate=dilation_rate,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Second conv
    x = layers.Conv1D(
        filters=filters,
        kernel_size=kernel_size,
        padding="causal",
        dilation_rate=dilation_rate,
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(dropout_rate)(x)

    # Residual connection: match channels if needed
    if x_in.shape[-1] != filters:
        x_in = layers.Conv1D(filters=filters, kernel_size=1, padding="same")(x_in)

    x = layers.Add()([x, x_in])
    x = layers.Activation("relu")(x)
    return x
    
def build_imu_tcn_model(
    time_steps=TIME_STEPS,
    num_channels=6,
    num_classes=5,
    num_filters_list=(32, 32, 64),
    kernel_size=3,
    dropout_rate=0.2,
):
    inputs = keras.Input(shape=(time_steps, num_channels))

    x = inputs
    for i, num_filters in enumerate(num_filters_list):
        dilation = 2 ** i
        x = tcn_block(
            x,
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            dropout_rate=dropout_rate,
        )

    # x shape: (batch, time_steps, last_num_filters)
    x = layers.GlobalAveragePooling1D()(x)  # pool over time

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="imu_tcn_classifier")
    return model


NUM_CLASSES = int(len(np.unique(y_train)))  # infer from labels

model = build_imu_tcn_model(
    time_steps=TIME_STEPS,
    num_channels=NUM_CHANNELS,
    num_classes=NUM_CLASSES,
    num_filters_list=(32, 32, 64),
    kernel_size=5,
    dropout_rate=0.3,
)

#model.summary()

# Train Model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,           # or 8–10 if val loss is noisy
    restore_best_weights=True
)

class_weights = {0:1.0, 1:5.0, 2:2.0, 3:1.0, 4:2.0}

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=50,       # you said batch size = 50
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=1,
)

# Prediction
print("===============================")

probs = model.predict(X_val)
y_val_pred = np.argmax(probs, axis=1)

print("Pred shape:", y_val_pred.shape)
print("First 10 predictions:", y_val_pred[:100])
print("First 10 true labels:", y_val[:100])

# Confusion matrix
cm = confusion_matrix(y_val, y_val_pred)
cm_percent = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100

print("Confusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_val, y_val_pred))

#class_names = [str(c) for c in np.unique(y_val)]  
# If you have real names:
class_names = ["No fall", "Forward", "Backward", "Left", "Right"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    #cm,
    cm_percent,
    annot=True,
    #fmt="d",
    fmt=".1f",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

