import time

import requests
import pandas as pd
import csv
import aiflow

if __name__ == '__main__':
    kml_controller = aiflow.KMLAIFlowController(28236)
    kml_controller.start()
    kml_controller.set_cold_start()
    for i in range(1, 16):
        kml_controller.change_replicas('worker', int(i))
        kml_controller.change_batch_size(int(16384 / i))
        kml_controller.submit_sparse_config()
        kml_controller.submit_record()
        time.sleep(60 * 30)
        kml_controller.stop_record()
    i = 32
    kml_controller.change_replicas('worker', int(i))
    kml_controller.change_batch_size(int(16384 / i))
    kml_controller.submit_sparse_config()
    kml_controller.submit_record()
    time.sleep(60 * 30)
    kml_controller.stop_record()
