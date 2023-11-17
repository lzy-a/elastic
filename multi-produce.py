import multiprocessing
import subprocess


def run_produce_script():
    subprocess.run(['python', 'produce-cripto.py'])


if __name__ == "__main__":
    # 启动五个进程
    num_processes = 5
    processes = []

    for _ in range(num_processes):
        process = multiprocessing.Process(target=run_produce_script)
        processes.append(process)
        process.start()

    # 等待所有进程完成
    for process in processes:
        process.join()
