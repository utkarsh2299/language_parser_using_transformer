# import pandas as pd
import pandas as pd

from sklearn.model_selection import train_test_split
# Load your data
df = pd.read_csv('hindi.csv')
df.dropna(inplace=True)

# Rename columns for clarity
df.columns = ['input_text', 'target_text']

# Add the "task prefix" which is a common practice for T5
df['input_text'] = "translate Hindi to Parsed: " + df['input_text']

# Split the data into training and validation sets
train_df, val_df = train_test_split(df, test_size=0.1)

# Save the prepared data to disk
train_df.to_csv('train_data.csv', index=False)
val_df.to_csv('val_data.csv', index=False)