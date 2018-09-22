#!/usr/bin/env sh
set -e

DATE=$(date +%Y%m%d)

LOG_DIR=./examples/resnet/log/

if [ ! -d "$LOG_DIR" ]; then
    mkdir "$LOG_DIR"
fi

LOG_FILE=$LOG_DIR/$DATE.log
echo $LOG_FILE

./build/tools/caffe train \
    --solver=examples/resnet/solver.prototxt \
    --weights=examples/resnet/model/resnet_v4_iter_60000.caffemodel \
    --gpu 1 2>&1 | tee $LOG_FILE


