import subprocess
import re
import time


def get_gpu_utilization():
    command = "nvidia-smi --query-gpu=utilization.gpu --format=csv,nounits,noheader"
    result = subprocess.run(command, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
    utilization_str = result.stdout.strip()
    gpu_utilization = int(utilization_str)
    return gpu_utilization


interval = 100  # ?~O?~M~A?~R?~S?~G??~@次
total_utilization = 0
count = 0

try:
    while True:
        gpu_utilization = get_gpu_utilization()
        total_utilization += gpu_utilization
        count += 1

        if count % (interval // 1) == 0:  # ?~O?~M~A?~R?~S?~G??~@次
            average_utilization = total_utilization / count
            print(f"Average GPU Utilization in the last {interval} seconds: {average_utilization}%")
            total_utilization = 0
            count = 0

        time.sleep(0.1)
except KeyboardInterrupt:
    print("\nExiting the loop.")
