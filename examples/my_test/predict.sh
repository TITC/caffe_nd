#!/bin/bash
GLOG_logtostderr=1 /tempspace/tzeng/caffe_nd_sense_segmetation/.build_release/tools/predict_seg.bin deploy_test.prototxt \
patch_test_snapshot_iter_1000.caffemodel \
mean_file.binary \
/tempspace/tzeng/bigneuron/big_neuron_hackthon_tesla/data/hd5_valid.h5 \
/tempspace/tzeng/bigneuron/big_neuron_hackthon_tesla/data/predict_valid.h5 \
GPU \
5