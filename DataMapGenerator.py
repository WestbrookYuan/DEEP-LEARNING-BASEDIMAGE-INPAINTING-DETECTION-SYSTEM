import os


def generate_map(root_dir):
    # 得到当前绝对路径
    current_path = os.path.abspath(root_dir)
    print(current_path)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    print(father_path)
    with open(root_dir + 'map.txt', 'w') as wfp:
        for file_name in os.listdir(current_path):
            abs_name = os.path.join(current_path, file_name)
            wfp.write('{file_dir}\n'.format(file_dir=abs_name))
    wfp.close()