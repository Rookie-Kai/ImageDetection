import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector

import numpy as np
import cv2
import matplotlib.pyplot as plt


def init_model(config, checkpoint, device):
    # Load the config
    config = mmcv.Config.fromfile(config)
    # Set pretrained to be None since we do not need pretrained model here
    config.model.pretrained = None

    # Initialize the detector
    model = build_detector(config.model)

    # Load checkpoint
    checkpoint = load_checkpoint(model, checkpoint, map_location=device)

    # Set the classes of models for inference
    model.CLASSES = checkpoint['meta']['CLASSES']

    # We need to set the model's cfg for inference
    model.cfg = config

    # Convert the model to GPU
    model.to(device)
    # Convert the model into evaluation mode
    model.eval()
    return model


config='/data/home/scv9243/run/mmdetection/configs/mask_rcnn_ballon/mask_rcnn_ballon_101.py'
checkpoint='/data/home/scv9243/run/mmdetection/work/mask_rcnn_ballon_101/best_segm_mAP_epoch_90.pth'

model = init_model(config=config, checkpoint=checkpoint, device='cuda:0')

cap = cv2.VideoCapture('/data/home/scv9243/run/mmdetection/work/test_video.mp4')

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print((height, width))
size = (width, height)
video = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, size)
print('start processing...')
while(cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        print("Can't receive valid frame/ arrive to end of video")
        video.release()
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = inference_detector(model, img)

    mask = np.array(result[1][0]).sum(axis=0).clip(0,1).astype(np.uint8)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # gray = gray[..., None].repeat(3,2)
    # 使用掩模图像mask创建一个与原图像大小相同的全0图像
    result = np.zeros_like(img)

    # 使用cv2.bitwise_and函数计算按位与运算
    cv2.bitwise_and(img, img, dst=result, mask=mask)

    # 将原图像中的mask外的地方设为灰色
    gray = gray[..., None].repeat(3, axis=2)
    mask = mask[..., None].repeat(3, axis=2)
    result[np.where(mask==0)] = gray[np.where(mask==0)]
    # plt.imshow(result)
    # plt.show()
    # break
    video.write(result)
    # break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()

cv2.destroyAllWindows()
print('done!!!')
