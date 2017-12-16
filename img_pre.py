import cv2
import os

img_path = "/Users/JohnsmithMBP/PycharmProjects/ffb/renamed_images/F204"
num = 231
gt_path = ""

width, height = 1210, 1210
out_path = os.path.join(os.getcwd(), "img_" + str(width))
os.mkdir(out_path)

for i in range(1, num + 1):
    fname = "image{}.jpg".format(i)
    fpath = os.path.join(img_path, fname)
    img = cv2.imread(fpath)
    img_resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_path, fname), img_resized)


