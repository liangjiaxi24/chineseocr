import glob
import os


jpg_list = glob.glob("./ocr/*/*.jpg")
txt_list = glob.glob("./ocr/*/*.txt")
print(len(jpg_list), len(txt_list))


# for index, img_name in enumerate(jpg_list):
#     txt_name = img_name[:-4] + ".txt"
#     if txt_name not in txt_list:
#         os.remove(img_name)
#         print(img_name)
#     if index % 1000 == 0:
#         print("current step {}".format(index))
#
#
# for index, txt_name in enumerate(txt_list):
#     img_name = txt_name[:-4] + ".jpg"
#     if img_name not in jpg_list:
#         os.remove(txt_name)
#         print(txt_name)
#     if index % 1000 == 0:
#         print("current step {}".format(index))
