## File Structure
- **Data** : This folder contains data stock trading data for 259 days.
- **src** : This folder contains entire logic of this project.
    - **.py** : Accepts input data.
    - **data_preprocess** : data_loader.py file loads the data and data_splitter.py splits the data into train and test set. 
    - **model_creation** : Create environment and model files for our RL model. 
    - **pipelines** : Trainer and Evaluator logic. 
    - **utils** : Benchmark, Config and Utils files. 
- **main.py** : Contains driver code.
- **artifacts** : This folder contains saved RL model while training.
- **output** : This folder contains generated strategy and benchmark comparison files.
- **Experiment.ipynb** : This notebook contains logic for Q-learning logic from the paper and PPO methods. 
- **README.md** : This markdown file you are reading.
- **requirements.txt** : Required imports to run the program successfully.


## Installation
Clone this github repo in your local machine
```commandline
git clone https://github.com/shyamal31/StockStrategyReinforcementLearning.git
```
(Optional) setup a virtual environment to install necessary packages using the following command:
``` commandline
python3 -m venv venv
source .venv/bin/activate
```
Install the packages listed in Requirements.txt file
```shell
pip install -r requirements.txt
```
Run evaluation on pretrained model. 
```shell
python3 main.py 
```
Train RL model
```shell
python3 main.py -tr
```
OR
```shell
python3 main.py --train
```
