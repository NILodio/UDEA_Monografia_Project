#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python 004_generate_tfrecord.py -x data/train.csv -l data/annotations/Word_Detection/label_map.pbtxt -o data/annotations/Word_Detection/train.record -i data/train
python 004_generate_tfrecord.py -x data/test.csv -l data/annotations/Word_Detection/label_map.pbtxt -o data/annotations/Word_Detection/test.record -i data/test
