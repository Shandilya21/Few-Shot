#!/bin/sh
read -p "Is train: (1 - Yes, 0 - no): " IS_TRAIN
echo "Setting Training network ....,"
python -m proto_nets --dataset fashionNet --k-test 5 --n-test 1 --k-train 4 --n-train 1 --q-train 3
python -m proto_nets --dataset fashionNet --k-test 5 --n-test 1 --k-train 5 --n-train 1 --q-train 2
python -m proto_nets --dataset fashionNet --k-test 1 --n-test 5 --k-train 6 --n-train 1 --q-train 2

echo "Training Completed !!!"