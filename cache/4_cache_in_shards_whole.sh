# note: if you find a good way to parallelize this, please let me know
# this is just a quick hack using taskset to limit CPU usage per process
taskset -c 0 python cache/4_cache_whole_f0.py LJSpeech val 0 2000 &
taskset -c 1 python cache/4_cache_whole_f0.py LJSpeech test 0 2000 &
taskset -c 2 python cache/4_cache_whole_f0.py LJSpeech train 0 2000 &
taskset -c 3 python cache/4_cache_whole_f0.py LJSpeech train 2000 4000 &
taskset -c 4 python cache/4_cache_whole_f0.py LJSpeech train 4000 6000 &
taskset -c 5 python cache/4_cache_whole_f0.py LJSpeech train 6000 8000 &
taskset -c 6 python cache/4_cache_whole_f0.py LJSpeech train 8000 10000 &
taskset -c 7 python cache/4_cache_whole_f0.py LJSpeech train 10000 12000 &


wait