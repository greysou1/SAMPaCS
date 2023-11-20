import cv2

# "person_bb": [[11, -5], [53, 297]]
x_min, y_min, x_max, y_max = 11, -5, 53, 297
x_min, y_min, x_max, y_max = [7, 15, 77, 275]
frame = cv2.imread("/home/c3-0/datasets/LTCC/LTCC_ReID/train/094_1_c9_015923.png")

cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
cv2.imwrite('debug-bb.jpg', frame)