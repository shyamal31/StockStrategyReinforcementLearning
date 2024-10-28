import os
import json
import torch
from model import DQNModel
from environment import MultiDayTradingExecutionEnv
from utils import get_initial_mid_spread
from benchmark import Benchmark
from data_loader import DataLoader


def model_fn(model_dir):
    """Load the model from the SageMaker model directory."""
    model = DQNModel(input_size=29, output_size=101)  # Assuming 29 inputs and 101 actions
    model.load_state_dict(torch.load(f"{model_dir}/model/dqn_model.pth"))#change directory path if running locally
    model.eval()
    return model

def input_fn(request_body, request_content_type='application/json'):
    """Deserialize the JSON input data."""
    if request_content_type == 'application/json':
        return json.loads(request_body)
    raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    """Run inference on the input data using the model."""
    ticker = input_data["ticker"]
    target_shares = input_data["shares"]
    time_horizon = input_data.get("time_horizon", 390)
    curr_dir = os.getcwd()
    dl = DataLoader(f'{curr_dir}/code/merged_bid_ask_ohlcv_data.csv')#change directory if running locally. 
    one_day_data = dl.get_live_trade_data(ticker, time_horizon)


    env = MultiDayTradingExecutionEnv(
        one_day_data,
        initial_mid_spread=get_initial_mid_spread(one_day_data),
        target_shares=target_shares
    )

    trade_schedule = []
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0)

    with torch.no_grad():
        done = False
        while not done:
            action = torch.argmax(model(state)).item()
            next_state, _, done, _ = env.step(action)
            trade_info = {
                "timestamp": env.current_day_data.iloc[env.current_step - 1]['timestamp'].isoformat(),
                "shares_to_sell": action
            }
            trade_schedule.append(trade_info)
            state = torch.FloatTensor(next_state).unsqueeze(0)

    return trade_schedule

def output_fn(prediction, content_type='application/json'):
    """Serialize the prediction result to JSON."""
    if content_type == 'application/json':
        return json.dumps(prediction)
    raise ValueError("Unsupported content type: {}".format(content_type))

#to run locally
# if __name__ == "__main__":
    # mock_request_body = json.dumps({
    #     "ticker": "AAPL",
    #     "shares": 1000,
    #     "time_horizon": 390
    # })
    # mock_content_type = 'application/json'


    # # Step 1: Load the Model (model_fn)
    # print("Loading model...")
    # model = model_fn('/Users/shyamalgandhi/Desktop/Shyamal/sagemaker_blockhouse/model')
    # print("Model loaded successfully.")

    # # Step 2: Process the Input (input_fn)
    # print("Processing input data...")
    # input_data = input_fn(mock_request_body, mock_content_type)
    # print("Input data processed:", input_data)


    # # Perform Inference
    # print("Running inference...")
    # trade_schedule = predict_fn(input_data, model)
    # print("Inference completed.")

    # # Step 4: Format the Output (output_fn)
    # print("Formatting output...")
    # output = output_fn(trade_schedule, mock_content_type)
    # print("Final Inference Output:", output)

