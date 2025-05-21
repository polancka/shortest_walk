# What file does what
- sp_zip.py -- ``python sp_zip.py <filename.zip>``
	Reads zip file and execute **sp.py** on 
	each txt file
- sum_log_time.py -- ``python sum_log_time.py <file2read.txt>``
	reads log.txt and outputs sum of every value in it, meaning it will tell u how long overall on all txt files inside zip did algorithm took
	example if log.txt contains
	
	```
	├───22
	│       dense-n100_k2.txt -> Execution time: 0.013001 seconds
	│       dense-n1k_k1.txt -> Execution time: 1.343077 seconds
	│       dense-n500_k-1.txt -> Execution time: 0.306018 seconds
	│       dense-n5k_k0.txt -> ZeroDivisionError
	├───23
	│       instance_100.txt -> Execution time: 0.000000 seconds
	│       instance_1000.txt -> Execution time: 0.007000 seconds
	│       instance_10000.txt -> Execution time: 0.063004 seconds
	│       instance_500.txt -> ZeroDivisionError
	│       instance_5000.txt -> Execution time: 0.064004 seconds
	└───24
	        100.txt -> ZeroDivisionError
	        1000.txt -> Execution time: 0.008001 seconds
	        10000.txt -> Execution time: 0.085005 seconds
	        500.txt -> ZeroDivisionError
	        5000.txt -> Execution time: 0.042002 seconds
	```
	**sum_log_time.py** will output
	```
	Total execution time: 1.931112 seconds
	```
- sp_numba_fail.py -- ``python sp_numba_fail.py 5000.txt``
	First attempt at speeding up dijkstra by attempting to JIT compile it with numba
	PS: if you want to run it with **sp_zip.py** and execute it on zip file, rename the file to **sp_zip.py**
- sp_numba.py -- ``python sp_faster_fail.py 5000.txt``
	Second attempt at speeding up dijkstra by not using numba but optimizing the code 	
		PS: if you want to run it with **sp_zip.py** and execute it on zip file, rename the file to **sp_zip.py**

