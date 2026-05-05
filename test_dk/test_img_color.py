import cv2
import numpy as np

img_array = np.fromfile("/home/k202/0425_replayoutput/000000/gopro/1777213760.861609.jpg", dtype=np.uint8)
img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
cv2.imwrite("/home/k202/lerobot/test_dk/weihao_color.jpg", img_bgr)


