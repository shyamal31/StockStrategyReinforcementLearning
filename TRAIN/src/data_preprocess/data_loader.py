import pandas as pd

class DataLoader:

    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def load_data(self):
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.dropna().sort_values(by='timestamp').reset_index(drop=True)
        return self.data

    def get_live_data(self, ticker, time_horizon):
        self.data = self.load_data()
        return self.data[:time_horizon]