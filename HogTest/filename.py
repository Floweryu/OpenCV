import os

root_path = 'D:\\Learn_Files\\OpenCV\\HogTest\\neg\\'
image_name_list = os.listdir(root_path)
with open('neg.txt', 'w') as fw:
    for name in image_name_list:
        path = os.path.join(root_path, name)
        # print(path)
        fw.write('{}\n'.format(path))