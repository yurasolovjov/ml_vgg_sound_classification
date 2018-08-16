python main.py --layer flatten \
	       --lr_lim 0.00005 \
	       --batch_size 32 \
	       --folds 3 \
	       --factor 6 \
	       --lr 0.000035 \
	       --input ../../track2/data/_raw_data_16000/_aug_data \
	       -l ../serilization \
	       --merge 3 \
	       --etalon_class_id ../cut_class_map.csv \
	       --load_test_data ../test_set \
	       --scheduler_mode 'triangular2' \
	       --optimizer adam	
		#2>&1 | tee -a ../dnn_1024_1024_41.log
