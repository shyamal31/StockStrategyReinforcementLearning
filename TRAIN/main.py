from src.data_preprocess.data_loader import DataLoader
from src.utils.config import config
from src.data_preprocess.data_splitter import DataSplitter
from src.pipelines.training_pipeline.trainer import TradingExecutionTrainer
from src.pipelines.testing_pipeline.evaluator import Evaluator
import pandas as pd
import json
from argparse import ArgumentParser

def load_train_test_split():
    dl = DataLoader("data/merged_bid_ask_ohlcv_data.csv")
    data = dl.load_data()
    
    # Split data
    splitter = DataSplitter(data, test_size=config["test_size"], random_state=config["random_state"])
    train_data, test_data = splitter.split_data()
    return data, train_data, test_data


def train():
    _,train_data , _ = load_train_test_split()

    # Train the model
    trainer = TradingExecutionTrainer(train_data, config)
    trainer.train_model()

def evaluate():
    data, _, test_data = load_train_test_split()
    evaluator = Evaluator(test_data, config)
    evaluator.evaluate_model()

    one_day_data = data[data['timestamp'].dt.date == pd.to_datetime("2023-09-12").date()] #currently only taking first data from the data given 

    trade_schedule = evaluator.inference(one_day_data)
    trade_schedule_json = json.dumps(trade_schedule, default=str)  # default=str to handle datetime serialization
    print("The desired output in json format: ")
    print()
    print(trade_schedule_json)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('-tr','--train', help= "Train/Evaluate model", action = 'store_true')

    args = parser.parse_args()
    if args.train:
        train()
    else:
        evaluate()
    # if args.pretrained is not None:
    #     usePreTrainedModel(args.pretrained)
    # else:
    #     if args.save:
    #         fullTraining(True)
    #     else:
    #         fullTraining()

    # Load data
    # dl = DataLoader("data/merged_bid_ask_ohlcv_data.csv")
    # data = dl.load_data()
    
    # # Split data
    # splitter = DataSplitter(data, test_size=config["test_size"], random_state=config["random_state"])
    # train_data, test_data = splitter.split_data()

    # # Train the model
    # # trainer = TradingExecutionTrainer(train_data, config)
    # # trainer.train_model()

    # # Evaluate the saved model
    # evaluator = Evaluator(test_data, config)
    # evaluator.evaluate_model()

    # one_day_data = data[data['timestamp'].dt.date == pd.to_datetime("2023-09-12").date()]

    # trade_schedule = evaluator.inference(one_day_data)
    # trade_schedule_json = json.dumps(trade_schedule, default=str)  # default=str to handle datetime serialization
    # print("The desired output in json format: ")
    # print()
    # print(trade_schedule_json)



