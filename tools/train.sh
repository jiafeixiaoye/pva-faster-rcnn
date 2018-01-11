#!/bin/bash
tools/train_net.py --gpu 0 --solver models/pvanet/example_train/solver.prototxt --weights models/pvanet/pretrained/pvanet_frcnn_iter_1350000.caffemodel --iters 500000 --cfg models/pvanet/cfgs/train.yml --imdb voc_2007_trainval
