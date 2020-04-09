import SCNN_Lane_Detection as scnn_lane_detection
import cv2
scnn_lane_detection.init('/home/joes/Development/ml-model/vgg_SCNN_DULR_w9.pth')

url = 'https://i.ytimg.com/vi/szhG6iPJmE4/maxresdefault.jpg'

scnn_lane_detection.predictThreshold(0.05)
img, lane_img = scnn_lane_detection.demo(url)
res = scnn_lane_detection.getAddWeight(img, lane_img)

cv2.imshow("test", res)
cv2.waitKey(0)

