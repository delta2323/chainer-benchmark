#!/usr/bin/env bash

logdir=log.`date +%y%m%d-%H%M%S`
mkdir ${logdir}

caffe time --model=./imagenet_winners/alexnet_128.prototxt  --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/alexnet.log 2>&1
caffe time --model=./imagenet_winners/overfeat_128.prototxt --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/overfeat.log 2>&1
caffe time --model=./imagenet_winners/vgg_a_16.prototxt     --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/vgg_a.log 2>&1

caffe time --model=proto_forceGradInput/conv1_64.prototxt  --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/conv1.log 2>&1
caffe time --model=proto_forceGradInput/conv2_32.prototxt  --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/conv2.log 2>&1
caffe time --model=proto_forceGradInput/conv3_64.prototxt  --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/conv3.log 2>&1
caffe time --model=proto_forceGradInput/conv4_128.prototxt --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/conv4.log 2>&1
caffe time --model=proto_forceGradInput/conv5_128.prototxt --iterations=10 --gpu 0 --logtostderr=1 > ${logdir}/conv5.log 2>&1
