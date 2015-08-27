# Chainer Benchmark Test 

This benchmark measures elapsed time of forward and backward
operations of several networks(Alexnet, Overfeat, VGG)
and several single convolutional functions.

The architecture of each network is referenced from
`convnet-benchmarks <https://github.com/soumith/convnet-benchmarks>`_ by Soumith Chintala


## Run All Benchmarks


```
bash profile.sh
```


This command executes benchmarks for all models, parse log files, and aggregate benchmark results.
The benchmark result file for each model is generated as `<model>.aggregate`


## Usage of Profile Script

```
python profile.py [-g <gpu>] [-m <model>] [-i <iteration>]
```

## Options

* `-g` (`int`): Device ID of GPU to use. If negative value is specified, it runs with CPU-mode.
* `-m` (`str`): Architecture to use. It must be one of `alex`, `overfeat`, `vgg`, or `conv[1-5]`.
* `-i` (`int`): The number of iterations of forward and backward computations.

## Sample

```
python profile.py -g 0 -m conv3 -i 10
```
