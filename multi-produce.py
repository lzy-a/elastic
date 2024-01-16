import multiprocessing
import subprocess
import signal
import sys


def run_produce_script():
    subprocess.run(['python', 'producer-criteo.py'])


def signal_handler(signum, frame):
    # 在收到 Ctrl+C 信号时关闭所有子进程
    print("Ctrl+C received. Terminating all processes.")
    for process in processes:
        process.terminate()
    sys.exit(0)


if __name__ == "__main__":
    # 启动五个进程
    num_processes = 5
    processes = []
    # 注册 Ctrl+C 信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    for _ in range(num_processes):
        process = multiprocessing.Process(target=run_produce_script)
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()
