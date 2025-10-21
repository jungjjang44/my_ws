import getCameraCylindricalLut as utils
import cv2
import json
import numpy as np
from matplotlib import pyplt as plt

image = cv2.cvtColor(cv2.imread("ELP-USB16MP01-BL180-2048x1536_EXTRINSIC.png", -1), cv2.COLOR_BGR2RGB)
calib = json.load(open("ELP-USB16MP01-BL180-2048x1536_calibration.json", "r"))

origin_height, origin_width, _ = image.shape
target_height, target_width  = origin_height, origin_width

intrinsic = np.array(calib['ELP-USB16MP01-BL180-2048x1536']['Intrinsic']['K']).reshape(3, 3)
intrinsic[0, :] *= (target_width/origin_width)
intrinsic[1, :] *= (target_height/origin_height)
distortion = np.array(calib['ELP-USB16MP01-BL180-2048x1536']['Intrinsic']['D'])

map_x, map_y = utils.get_camera_cylindrical_spherical_lut(intrinsic, distortion, "cylindrical", target_width, target_height, hfov_deg=180, vfov_deg=180, roll_degree=0, pitch_degree=0, yaw_degree=0)
new_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
plt.imshow(new_image)