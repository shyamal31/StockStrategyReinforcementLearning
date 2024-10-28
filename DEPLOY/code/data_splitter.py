import pandas as pd
from sklearn.model_selection import train_test_split

class DataSplitter:
    def __init__(self, data, test_size=0.2, random_state=42):
        self.data = data
        self.test_size = test_size
        self.random_state = random_state

    def split_data(self):
        unique_dates = self.data['timestamp'].dt.date.unique()
        train_dates, test_dates = train_test_split(
            unique_dates, test_size=self.test_size, random_state=self.random_state
        )
        train_data = self.data[self.data['timestamp'].dt.date.isin(train_dates)]
        test_data = self.data[self.data['timestamp'].dt.date.isin(test_dates)]
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)
