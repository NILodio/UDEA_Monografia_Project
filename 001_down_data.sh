#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python 001_down_data.py --name_data=Word_Detection --name_path=data
# python 001_down_data.py --name_data=Sheet_detection --name_path=data
