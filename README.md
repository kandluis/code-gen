### Steps for reproducing milestone 2 results: 
### Fine-tuning GPT-2 on CodeSearchNet for Code Generation

## Setup

- From /cs230/gpt-2-csn:

$ pip install -r path/to/requirements.txt

$ python download_model.py 117M

## Data

- Note that CodeSearchNet python training data is stored in ./src/pythonTrain/python_train_all.txt and git lfs tracked for space efficiency.

- This dataset has been encoded and stored in ./src/pythonTrainPreprocessed/python_train_all.npz for space efficiency.

- To encode your own data, the following script is compatible with a minimum python 3.7.x: $ python encode.py trainingData.txt trainingData.npz

## Training

$ python /src/train.py --dataset /src/pythonTrainPreprocessed/python_train_all.npz --model_name 117M --batch_size 2

- Samples will automatically be generated every 100 steps. Additional parameters such as learning rate can be specified on the commandline.
