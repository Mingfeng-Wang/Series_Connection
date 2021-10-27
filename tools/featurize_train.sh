#!/usr/bin/env bash
  
set -x

CONFIG=$1
WORK_DIR=$2
python -u tools/train.py ${CONFIG} --work-dir=${WORK_DIR} --seed 1334 --deterministic
