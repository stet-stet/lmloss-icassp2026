mkdir inferred/metrics_test
taskset -c 0 python -m src.inference.run_wer_norm_concat transcribed > inferred/metrics_test/metric_wer_norm_concat.txt &
taskset -c 1 python -m src.inference.run_intrusives pesq inferred > inferred/metrics_test/metric_pesq.txt &
taskset -c 3 python -m src.inference.run_intrusives warpq_norm inferred > inferred/metrics_test/metric_warpq_norm.txt & 
wait
