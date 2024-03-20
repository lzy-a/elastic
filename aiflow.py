import requests
import os
import json
import sys
import yaml
import time
import datetime

# set header,use wangqiang07 self token
headers = {'Content-Type': 'application/json', 'KML-Project-ID': "529",
           'KML-Auth-Token': 'tSNuV2N3jGnFuBsyYZokbD04WsqD9avLlu5UDxMEht45uQHekzTjT4bCIzjKVZ4M'}


#                     'KML-Auth-Token': 'D0ndLr4EqjIeZEW3gL2HdX12U6ALQOzDNQ4KvqoGG0ag345FOScH9NnH3fwzfkuL'}


class KMLHttpException(Exception):
    "this is user's Exception for check the length of name "

    def __init__(self, http_code, msg):
        self.http_code = http_code
        self.msg = msg

    def __str__(self):
        formstr = 'request:{}, failed, http status code:{}'.format(self.msg, self.http_code)
        return formstr


class KMLAIFlowController(object):
    def __init__(self, flowid):
        self.flowid = flowid  # workspace id
        self.basic_info_comid = None  # 迭代详情id
        self.config_comid = None  # 训练配置id
        self.task_comid = None  # 模型训练id
        self.sparse_basicinfo = None # 迭代详情
        self.sparse_config_yamldict = None # 训练配置
        self.submit_config = {"config": {"globalConfig": {"enableDebug": False}, "runtimeModeConfigs": [
            {"runtimeMode": "train", "coldStart": True, "willSaveModel": True, "useBtq": False,
             "willSaveRollbackModel": False}], "defaultRuntimeMode": "train"}}
        self.log_url = None

    ''' 
    start: get comid from flowid
    iteration main page:  https://kml.corp.kuaishou.com/v2/#/project/529/sparse/workspace/817
    '''

    def start(self):
        workspace_url = 'https://kml.corp.kuaishou.com/v2/ai-flow/api/v1/workspace/{}'.format(flowid)
        http_ret = requests.get(workspace_url, headers=headers)
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "start stage, workspace_url:{}".format(workspace_url))

        self.workspace_detail = json.loads(http_ret.text)
        data = self.workspace_detail
        print('workspace_detail:', json.dumps(data, indent=4))
        for i in range(len(self.workspace_detail['components'])):
            comp = self.workspace_detail['components'][i]
            if comp['identity'] == 'sparse-basic':
                self.basic_info_comid = comp['id']
            elif comp['identity'] == 'sparse-training-config':
                self.config_comid = comp['id']
            elif comp['identity'] == 'sparse-training-task':
                self.task_comid = comp['id']

        print(self.basic_info_comid, self.config_comid, self.task_comid)
        basicinfo_url = 'https://kml.corp.kuaishou.com/v2/ai-flow/api/v1/com/{}/basic-info'.format(
            self.basic_info_comid)
        # print('basicinfo_url:', basicinfo_url)
        http_ret = requests.get(basicinfo_url, headers=headers)
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "get_basicinfo stage, url:{}".format(basicinfo_url))
        self.sparse_basicinfo = json.loads(http_ret.text)
        data = self.sparse_basicinfo
        # print('sparse_basicinfo:', json.dumps(data, indent=4))

        sparse_config_url = 'http://kml.corp.kuaishou.com/v2/ai-flow/api/v1/com/{}/sparse-training-config'.format(
            self.config_comid)
        http_ret = requests.get(sparse_config_url, headers=headers)
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "get_sparse_config stage, url:{}".format(sparse_config_url))
        sparse_config = json.loads(http_ret.text)
        self.sparse_config_yamldict = yaml.load(sparse_config['config'], Loader=yaml.Loader)
        data = self.sparse_config_yamldict
        # print('sparse_config_yamldict:', json.dumps(data, indent=4))

    def change_image(self, image_name):
        print('new image name', image_name)
        self.sparse_basicinfo['imageName'] = image_name
        basicinfo_url = 'https://kml.corp.kuaishou.com/v2/ai-flow/api/v1/com/{}/basic-info'.format(
            self.basic_info_comid)
        http_ret = requests.post(basicinfo_url, headers=headers, data=json.dumps(self.sparse_basicinfo))
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "change_image stage, url:{}".format(basicinfo_url))
        print('change image success', http_ret.text)

    def change_train_begin_time_ms(self, begin_time_ms):
        self.sparse_config_yamldict['io_config']['train']['begin_time_ms'] = begin_time_ms

    def change_train_end_time_ms(self, end_time_ms):
        self.sparse_config_yamldict['io_config']['train']['end_time_ms'] = end_time_ms

    def submit_sparse_config(self):
        #提交新的训练配置
        sparse_config_url = 'http://kml.corp.kuaishou.com/v2/ai-flow/api/v1/com/{}/sparse-training-config'.format(
            self.config_comid)
        new_config_yamlstr = yaml.dump(self.sparse_config_yamldict, default_flow_style=False, allow_unicode=True)
        print('new yaml config', new_config_yamlstr)
        pconf = dict()
        pconf['config'] = new_config_yamlstr
        http_ret = requests.post(sparse_config_url, headers=headers, data=json.dumps(pconf))
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "get_basicinfo stage, url:{}".format(sparse_config_url))

    def submit_record(self):

        task_url = 'http://kml.corp.kuaishou.com/v2/ai-flow/api/v1/com/{}/task-record'.format(self.task_comid)
        http_ret = requests.post(task_url, headers=headers, data=json.dumps(self.submit_config))
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "submit_record stage, url:{}".format(task_url))

        record_detail = json.loads(http_ret.text)
        self.panda_id = record_detail['pandaTaskRecordId']

        panda_url = 'https://kml.corp.kuaishou.com/v2/panda/api/v1/task-records/{}'.format(self.panda_id)
        http_ret = requests.get(panda_url, headers=headers)
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "get_record_status stage, url:{}".format(panda_url))

        record_status = json.loads(http_ret.text)
        task_id = record_status['taskMetaId']
        cluster_name = record_status['clusterName']
        self.log_url = 'http://kml.corp.kuaishou.com/v2/log-manager/api/v1/logs?namespace=gray-dedicated&taskType=task&taskId={}&instance=kml-task-{}-record-{}-prod-launcher-0&clusterName={}&limit=100&trimTimestamp=true'.format(
            self.panda_id, task_id, self.panda_id, cluster_name)

    def compose_log_url(self):
        self.get_record_status()
        task_id = self.latest_record_status['taskMetaId']
        cluster_name = self.latest_record_status['clusterName']
        self.log_url = 'http://kml.corp.kuaishou.com/v2/log-manager/api/v1/logs?namespace=gray-dedicated&taskType=task&taskId={}&instance=kml-task-{}-record-{}-prod-launcher-0&clusterName={}&limit=100&trimTimestamp=true'.format(
            self.panda_id, task_id, self.panda_id, cluster_name)

    def get_launcher_log(self):
        http_ret = requests.get(self.log_url, headers=headers)
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "get_launcher_log stage, log_url:{}".format(self.log_url))
        return http_ret.text

    def get_record_status(self):
        url = 'https://kml.corp.kuaishou.com/v2/panda/api/v1/task-records/{}'.format(self.panda_id)
        http_ret = requests.get(url, headers=headers)
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "get_record_status stage, url:{}".format(url))
        self.latest_record_status = json.loads(http_ret.text)

    def stop_record(self):
        url = 'https://kml.corp.kuaishou.com/v2/panda/api/v1/task-records/{}/stop'.format(self.panda_id)
        http_ret = requests.post(url, headers=headers)
        if http_ret.status_code != 200:
            raise KMLHttpException(http_ret.status_code, "stop_record stage, url:{}".format(url))


if __name__ == '__main__':
    try:
        flowid = 28236
        kml_controller = KMLAIFlowController(flowid)
        kml_controller.start()
        # image_name = 'registry.corp.kuaishou.com/kml-platform/kai_v2:cid-b408474_c408474_master_548955c_all_gpu-python-stream-cuda-11.4-nvtf-1.15-20221207_153429'
        # kml_controller.change_image(image_name)
        #
        # # get time for yesterday 01:00:00 to 02:00:00
        # yesterday = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')
        # bts = int(time.mktime(time.strptime('{} 01:00:00'.format(yesterday), '%Y-%m-%d %H:%M:%S'))) * 1000
        # ends = int(time.mktime(time.strptime('{} 03:00:00'.format(yesterday), '%Y-%m-%d %H:%M:%S'))) * 1000
        #
        # kml_controller.change_train_begin_time_ms(bts)
        # kml_controller.change_train_end_time_ms(ends)
        # kml_controller.submit_sparse_config()
        kml_controller.submit_record()
        # kml_controller.panda_id = 4293359
        time.sleep(5*60)
        kml_controller.stop_record()
        kml_controller.submit_record()
        kml_controller.compose_log_url()
        #
        while True:
            rawlog = kml_controller.get_launcher_log()
            logtxt = json.loads(rawlog)
            print(logtxt['logs'])
            kml_controller.get_record_status()
            print('status', kml_controller.latest_record_status['status'])
            time.sleep(10)
    except KMLHttpException as e:
        print(e)
