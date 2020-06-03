#!/bin/sh
read -p "Is train: (1 - Yes, 0 - no): " IS_TRAIN
read -p "Please enter distance metric: (1 - cosine, 0 - l2): " distance
echo "Setting Training network ....,"

echo "Start with 1 shot variant"
python -m proto_nets --dataset fashionNet --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 5
python -m proto_nets --dataset fashionNet --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 10
python -m proto_nets --dataset fashionNet --k-test 5 --n-test 1 --k-train 20 --n-train 1 --q-train 15

# echo "Start with 5 shot variant"
python -m experiments.proto_nets --dataset miniImageNet --k-test 5 --n-test 5 --k-train 20 --n-train 5 --q-train 5
python -m experiments.proto_nets --dataset miniImageNet --k-test 5 --n-test 5 --k-train 20 --n-train 5 --q-train 10
python -m experiments.proto_nets --dataset miniImageNet --k-test 5 --n-test 5 --k-train 20 --n-train 5 --q-train 15

echo "Training Completed !!!"