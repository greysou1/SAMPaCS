import os
import cv2
import numpy as np

# def find_white_pixel_coordinates(image_path):
#     # Read the binary image
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#     # Find the coordinates of white pixels
#     white_pixel_coordinates = np.column_stack(np.where(image == 255))
#     return white_pixel_coordinates

# image_path = 'path/to/your/binary/image.png'
# white_coordinates = find_white_pixel_coordinates(image_path)
# print("White pixel coordinates:", white_coordinates)


# items in root 
img_root = "/home/c3-0/datasets/LTCC/LTCC_ReID/query"
sil_root = "outputs/LTCC/silhouettes/query/silhouettes-person"

imgs = os.listdir(img_root)
sils = os.listdir(sil_root)
for sil in sils:
    if sil not in imgs:
        print(sil)


# delete_files = ["001_1_c11_015843.png",
#                 "001_1_c11_015844.png",
#                 "001_1_c11_015838.png",
#                 "001_1_c11_015836.png",
#                 "001_1_c11_015852.png",
#                 "001_1_c11_015845.png",
#                 "001_1_c11_015839.png",
#                 "001_1_c11_015842.png",
#                 "001_1_c4_015855.png",
#                 "001_1_c11_015853.png",
#                 "001_1_c11_015854.png",
#                 "001_1_c4_015860.png",
#                 "001_1_c4_015856.png",
#                 "001_1_c4_015858.png",
#                 "001_1_c5_015867.png",
#                 "001_1_c11_015850.png",
#                 "001_1_c4_015863.png",
#                 "001_1_c5_015869.png",
#                 "001_1_c4_015864.png",
#                 "001_1_c11_015848.png",
#                 "001_1_c11_015834.png",
#                 "001_1_c11_015841.png",
#                 "001_1_c11_015846.png",
#                 "001_1_c11_015900.png",
#                 "001_1_c4_015865.png",
#                 "001_1_c4_015862.png",
#                 "001_1_c5_015868.png",
#                 "001_1_c5_015866.png",
#                 "001_1_c4_015859.png",
#                 "001_1_c4_015857.png",
#                 "001_1_c11_015847.png",
#                 "001_1_c5_015870.png",
#                 "001_1_c11_015840.png",
#                 "001_1_c11_015849.png",
#                 "001_1_c11_015835.png"]

# for d_file in delete_files:
#     os.remove(os.path.join(sil_root, d_file))