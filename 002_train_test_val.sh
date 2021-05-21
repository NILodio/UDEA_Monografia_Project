#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python 002_train_test_val.py
# python 001_down_data.py --name_data=Sheet_detection --name_path=data
