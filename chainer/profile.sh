#!/usr/bin/env bash

logdir=log.`date +%y%m%d-%H%M%S`
mkdir ${logdir}

trial=${1:-10}
iteration=${2:-11}

for i in `seq 1 ${trial}`
do
    python profile.py -g 0 -i ${iteration} -m alex     -b 128 > ${logdir}/alex.log.${i}
    python profile.py -g 0 -i ${iteration} -m overfeat -b 128 > ${logdir}/overfeat.log.${i}
    python profile.py -g 0 -i ${iteration} -m vgg      -b 16  > ${logdir}/vgg.log.${i}

    python profile.py -g 0 -i ${iteration} -m conv1    -b 64  > ${logdir}/conv1.log.${i}
    python profile.py -g 0 -i ${iteration} -m conv2    -b 32  > ${logdir}/conv2.log.${i}
    python profile.py -g 0 -i ${iteration} -m conv3    -b 64  > ${logdir}/conv3.log.${i}
    python profile.py -g 0 -i ${iteration} -m conv4    -b 128 > ${logdir}/conv4.log.${i}
    python profile.py -g 0 -i ${iteration} -m conv5    -b 128 > ${logdir}/conv5.log.${i}
done

for model in alex overfeat vgg conv1 conv2 conv3 conv4 conv5
do
    for kw in 'first-forward-pass' 'first-backward-pass' 'average-forward-pass(iter2-)' 'average-backward-pass(iter2-)' 'first-iteration-pass' 'average-iteration-pass(iter2-)'
    do
	sum=`grep ${kw} ${logdir}/${model}.log.* | cut -f 2 | paste -d+ -s - | bc`

	if [[ kw == *average* ]]
	then
	    denom=`expr ${trial} \* ( ${iteration} - 1 )`
	else
	    denom=${trial}
	fi

	echo ${kw} >> ${logdir}/${model}.aggregate
	python -c "print(${sum} / ${denom})" >> ${logdir}/${model}.aggregate
    done
done
