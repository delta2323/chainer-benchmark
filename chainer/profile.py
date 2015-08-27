import argparse

from chainer import cuda
import numpy
import six

import timer

parser = argparse.ArgumentParser(description='Profiler')
parser.add_argument('--model', '-m', type=str, default='alex',
                    help='network architecture (alex|overfeat|vgg|conv[1-5])')
parser.add_argument('--iteration', '-i', type=int, default=10,
                    help='iteration')
parser.add_argument('--gpu', '-g', type=int, default=0,
                    help='gpu to use')
args = parser.parse_args()


if args.model == 'alex':
    import alex
    model = alex.Alex()
elif args.model == 'overfeat':
    import overfeat
    model = overfeat.Overfeat()
elif args.model == 'vgg':
    import vgg
    model = vgg.VGG()
elif args.model.startswith('conv'):
    import conv
    number = args.model[4:]
    model = getattr(conv, 'Conv{}'.format(number))()
else:
    raise ValueError('Invalid model name')

print('Architecture\t{}'.format(args.model))
print('Iteration\t{}'.format(args.iteration))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
else:
    raise ValueError('Currently only GPU mode is supported')


total_timer = timer.Timer()
total_timer.start()

forward_preprocess_timer = timer.Timer()
forward_timer = timer.Timer()
backward_preprocess_timer = timer.Timer()
backward_timer = timer.Timer()
iter_timer = timer.Timer()

forward_preprocess_times = numpy.zeros((args.iteration,), numpy.float32)
forward_times = numpy.zeros((args.iteration,), numpy.float32)
backward_preprocess_times = numpy.zeros((args.iteration,), numpy.float32)
backward_times = numpy.zeros((args.iteration,), numpy.float32)
iter_times = numpy.zeros((args.iteration,), numpy.float32)

for iteration in six.moves.range(args.iteration):
    iter_timer.start()

    forward_preprocess_timer.start()
    x_batch = numpy.random.uniform(-1, 1,
                                   (model.batchsize,
                                    model.in_channels,
                                    model.insize,
                                    model.insize)).astype(numpy.float32)
    if args.gpu >= 0:
        x_batch = cuda.to_gpu(x_batch)
    forward_preprocess_times[iteration] = forward_preprocess_timer.milliseconds()


    forward_timer.start()
    y = model.forward(x_batch)
    forward_times[iteration] = forward_timer.milliseconds()

    backward_preprocess_timer.start()
    if args.gpu >= 0:
        y.grad = cuda.ones_like(y.data)
    else:
        y.grad = numpy.ones_like(y.data)
    backward_preprocess_times[iteration] = backward_preprocess_timer.milliseconds()

    backward_timer.start()
    y.backward()
    backward_times[iteration] = backward_timer.milliseconds()

    iter_times[iteration] = iter_timer.milliseconds()

total_timer.stop()

print('Forward Preprocess:')
print('average-forward-preprocess-pass\t{}\tms'.format(forward_preprocess_times.mean()))
print('Forward:')
print('each-forward-pass\t{}\t'.format('\t'.join(map(str, forward_times))))
print('first-forward-pass\t{}\tms'.format(forward_times[0]))
print('average-forward-pass(iter2-)\t{}\tms'.format(forward_times[1:].mean()))
print('Backward Preprocess:')
print('average-backward-preprocess-pass\t{}\tms'.format(backward_preprocess_times.mean()))
print('Backward:')
print('each-backward-pass\t{}\t'.format('\t'.join(map(str, backward_times))))
print('first-backward-pass\t{}\tms'.format(backward_times[0]))
print('average-backward-pass(iter2-)\t{}\tms'.format(backward_times[1:].mean()))
print('Iteration:')
print('each-iteration-pass\t{}'.format('\t'.join(map(str, iter_times))))
print('first-iteration-pass\t{}\tms'.format(iter_times[0]))
print('average-iteration-pass(iter2-)\t{}\tms'.format(iter_times[1:].mean()))
print('total-pass\t{}\tms'.format(total_timer.milliseconds()))
