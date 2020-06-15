#!/bin/sh
read -p "Is train: (1 - Yes, 0 - no): " IS_TRAIN
echo "Setting Training network ......"
echo "ProtoTypical Network"
python -m proto_nets --dataset fashionNet --k-test 5 --n-test 1 --k-train 4 --n-train 1 --q-train 3
python -m proto_nets --dataset fashionNet --k-test 5 --n-test 1 --k-train 5 --n-train 1 --q-train 2
python -m proto_nets --dataset fashionNet --k-test 1 --n-test 5 --k-train 6 --n-train 1 --q-train 2
python -m proto_nets --dataset fashionNet --k-test 1 --n-test 10 --k-train 4 --n-train 1 --q-train 1

echo "Setting MAML Training ...."
python -m experiments.maml --dataset miniImageNet --order 1 --n 1 --k 5 --q 3 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400
python -m experiments.maml --dataset miniImageNet --order 1 --n 5 --k 5 --q 3 --meta-batch-size 4 \
    --inner-train-steps 5 --inner-val-steps 10 --inner-lr 0.01 --eval-batches 40 --epoch-len 400

echo "Training Completed !!!"

