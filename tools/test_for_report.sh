#!/bin/bash
for file in /home/work-station/pva-faster-rcnn/output/model_for_testing/model/*
do
	echo ${file}
	if test -f $file
	then
		filename=`basename $file .caffemodel`
		num=${filename#*_}
		echo ${num}
		def_name_prefix='model_'
		def_name_suffix='.prototxt'
		def_name=${def_name_prefix}${num}${def_name_suffix}
		file_suffix='.caffemodel'
		export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64
		/usr/bin/python2.7 test_net.py --net ../output/model_for_testing/model/${filename}${file_suffix} --def ../output/model_for_testing/definition/${def_name} --cfg ../models/pvanet/cfgs/submit_1019.yml --gpu 0
		res_name_path=../data/VOCdevkit2007/VOC2007/folder_results.txt
		if [ -f $res_name_path ]
		then
			des_path_prefix='../data/VOCdevkit2007/VOC2007/folder_results/folder_results_'
			des_path_suffix='.txt'
			des_path=${des_path_prefix}${num}${des_path_suffix}
			mv $res_name_path ${des_path}
		fi
	fi
done
