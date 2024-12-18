import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
from scipy.signal import savgol_filter


pwd = os.getcwd()

names = [
          'YOLOv8s-LY', 'best_ckpt'
        ]

# names = [
#           'YOLOv8s-LY(inner-siou)-84.8',
#           'YOLOv8n-79.5', 'YOLOv8s-81.1',
#           'YOLOv6n-78.2', 'YOLOv6s-79.6',
#           'YOLOv5n-79.4', 'YOLOv5s-80.6']

# names = [ 'YOLOv5s-80.6',
#           'YOLOv6s-79.6',
#           'YOLOv8s-81.1',
#           'YOLOv8s-LY(inner-siou)-84.8']

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    # 对数据进行滑动窗口平滑处理
    smoothed_data = data.rolling(window=4).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(smoothed_data['   metrics/precision(B)'], label=i)
plt.xlabel('epoch')
plt.title('precision')
plt.legend()

plt.subplot(2, 2, 2)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    smoothed_data = data.rolling(window=4).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(smoothed_data['      metrics/recall(B)'], label=i)
plt.xlabel('epoch')
plt.title('recall')
plt.legend()

plt.subplot(2, 2, 3)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    smoothed_data = data.rolling(window=4).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(smoothed_data['       metrics/mAP50(B)'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5')
plt.legend()

plt.subplot(2, 2, 4)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    smoothed_data = data.rolling(window=4).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(smoothed_data['    metrics/mAP50-95(B)'], label=i)
plt.xlabel('epoch')
plt.title('mAP_0.5:0.95')
plt.legend()

plt.tight_layout()
plt.savefig('metrice_curve.png')
print(f'metrice_curve.png save in {pwd}/metrice_curve.png')

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    # smoothed_data = data.rolling(window=10).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(data['         train/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/box_loss')
plt.legend()

plt.subplot(2, 3, 2)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    # smoothed_data = data.rolling(window=10).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(data['         train/dfl_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/dfl_loss')
plt.legend()

plt.subplot(2, 3, 3)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    # smoothed_data = data.rolling(window=10).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(data['         train/cls_loss'], label=i)
plt.xlabel('epoch')
plt.title('train/cls_loss')
plt.legend()

plt.subplot(2, 3, 4)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    # smoothed_data = data.rolling(window=10).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(data['           val/box_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/box_loss')
plt.legend()

plt.subplot(2, 3, 5)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    # smoothed_data = data.rolling(window=10).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(data['           val/dfl_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/dfl_loss')
plt.legend()

plt.subplot(2, 3, 6)
for i in names:
    data = pd.read_csv(f'runs/train/{i}/results.csv')
    # smoothed_data = data.rolling(window=10).mean()  # 使用rolling方法进行简单移动平均
    plt.plot(data['           val/cls_loss'], label=i)
plt.xlabel('epoch')
plt.title('val/cls_loss')
plt.legend()

plt.tight_layout()
plt.savefig('loss_curve.png')
print(f'loss_curve.png save in {pwd}/loss_curve.png')