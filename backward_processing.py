import pandas as pd
import matplotlib.pyplot as plt

def create_and_update_column(df, column_name, default_value, start_n, length_m, target_value):
    """
    Creates a new column in a pandas DataFrame with a default value, 
    and sets a specific range of rows (from index start_n to start_n + length_m - 1) 
    to a given target value.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the new column to create.
        default_value: The default value for all rows.
        start_n (int): The starting index of the range to update (inclusive).
        length_m (int): The number of rows to update.
        target_value: The value to set for the specified range.
    """
    
    # 1. Create the new column with the default value for all rows
    df[column_name] = default_value
    
    # Calculate the end index (exclusive for iloc, which is standard Python slicing)
    end_index = start_n + length_m
    
    # 2. Update the specific range of rows using .iloc for positional indexing
    # Slicing is up to (but not including) the end_index
    df.iloc[start_n:end_index, df.columns.get_loc(column_name)] = target_value
    
    return df
    
gyro = pd.read_csv('gyro_back_3.csv')
acc = pd.read_csv('acc_back_3.csv')

gyro_notime = gyro.drop('Time', axis=1)
acc_notime = acc.drop('Time', axis=1)

acc_gyro = pd.concat([acc_notime, gyro_notime], axis=1)
clean_acc_gyro = acc_gyro.dropna()
rename_acc_gyro = clean_acc_gyro.rename(columns={'Acceleration x':'AccX', 'Acceleration y':'AccY', 'Acceleration z':'AccZ', \
                                                'Gyroscope x':'GyroX', 'Gyroscope y':'GyroY','Gyroscope z':'GyroZ'})

rename_acc_gyro.plot(kind='line', title='Back fall set 3')
plt.ylabel('Y-Values')
plt.xlabel('Index')
plt.show()


final_acc_gyro = create_and_update_column(rename_acc_gyro, "direction", 0, 39, 20, 2)
print(final_acc_gyro)
final_acc_gyro.to_csv('data\\backward_set_3.csv', index=False)


