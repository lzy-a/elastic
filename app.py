# 文件名: app.py
from flask import Flask, request, jsonify
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

app = Flask(__name__)


class Args:
    def __init__(self, is_training=1, itr=3, task_id='kai', model='informer', mode_select='random', modes=64,
                 version='Fourier', L=3, base='legendre', cross_activation='tanh', data='custom',
                 root_path='./dataset/',
                 data_path='predict_data.csv', features='S', target='target', freq='h', detail_freq='h',
                 checkpoints='./checkpoints/',
                 seq_len=96, label_len=48, pred_len=4, enc_in=1, dec_in=1, c_out=1, e_layers=2,
                 d_layers=1, moving_avg=[24], factor=3, distil=True, dropout=0.05, embed='timeF',
                 activation='gelu',d_model=512, n_heads=8, d_ff=2048,
                 output_attention=False, do_predict=True, num_workers=10, train_epochs=10, batch_size=32, patience=3,
                 learning_rate=0.0001, des='test', loss='mse', lradj='type1', use_amp=False, use_gpu=True, gpu=0,
                 use_multi_gpu=False, devices='0,1'):
        self.is_training = is_training
        self.itr = itr
        self.task_id = task_id
        self.model = model
        self.mode_select = mode_select
        self.modes = modes
        self.version = version
        self.L = L
        self.base = base
        self.cross_activation = cross_activation
        self.data = data
        self.root_path = root_path
        self.data_path = data_path
        self.features = features
        self.target = target
        self.freq = freq
        self.detail_freq = detail_freq
        self.checkpoints = checkpoints
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.dec_in = dec_in
        self.c_out = c_out
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.moving_avg = moving_avg
        self.factor = factor
        self.distil = distil
        self.dropout = dropout
        self.embed = embed
        self.activation = activation
        self.output_attention = output_attention
        self.do_predict = do_predict
        self.num_workers = num_workers
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.learning_rate = learning_rate
        self.des = des
        self.loss = loss
        self.lradj = lradj
        self.use_amp = use_amp
        self.use_gpu = use_gpu
        self.gpu = gpu
        self.use_multi_gpu = use_multi_gpu
        self.devices = devices


# 模拟的推理函数
def perform_inference(input_data):
    # 在这里调用你的模型推理逻辑，这里简单地返回输入数据
    return f"Model Inference Result: {input_data}"


@app.route('/predict', methods=['GET'])
def predict():
    try:
        # 调用模型推理
        result = modelPridict()

        # 返回推理结果
        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def modelTrain():
    Exp = Exp_Main
    args = Args(
        itr=1,
        model='FEDformer',
        seq_len=96,
        pred_len=96,
    )
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)

            # if args.do_predict:
            #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            #     exp.predict(setting, True)

            torch.cuda.empty_cache()


def modelPridict():
    Exp = Exp_Main
    args = Args(
        itr=1,
        model='transformer',
        seq_len=96,
        label_len=48,
        pred_len=4,
        data='custom',
        features='S',
    )
    result = []
    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_modes{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.task_id,
                args.model,
                args.mode_select,
                args.modes,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            exp = Exp(args)  # set experiments

            if args.do_predict:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                result = exp.predict(setting, True)

            torch.cuda.empty_cache()

    return result


if __name__ == '__main__':
    # modelTrain()
    app.run(host='0.0.0.0', port=5000)
