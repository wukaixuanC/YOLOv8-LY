import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["WANDB_API_KEY"] = '5f97e2c25db042b527e78bf9424bdc1f8ac74016'
os.environ["WANDB_MODE"] = "offline"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
CUDA_LAUNCH_BLOCKING = 1
from ultralytics import YOLO



if __name__ == '__main__':
    # # 直接使用预训练模型创建模型.
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg': 'ultralytics/yolo/cfg/exp1.yaml', 'data': 'dataset/data.yaml'})

    # 使用yaml配置文件来创建模型,并导入预训练权重.
    model = YOLO('ultralytics/models/v8/yolov8.yaml')
    # model.load('weights/yolov8n.pt')
    model.train(**{'cfg': 'ultralytics/yolo/cfg/default.yaml', 'data': r'C:dataset\data.yaml'})

    # # 模型验证
    # model = YOLO('runs/detect/train30/weights/best.pt')
    # model.val(**{'data': 'dataset/data.yaml'})

    # # 模型推理
    # model = YOLO('runs/detect/yolov8n_exp/weights/best.pt')
    # model.predict(source='dataset/images/test', **{'save': True})
