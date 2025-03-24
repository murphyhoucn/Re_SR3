import os
# only works for linux kernels
base_path = "./datasets/Nature/x128"
common_path = "./datasets/Nature/x128_all"

classes = os.listdir(base_path)
class_paths = [os.path.join(base_path, _class) for _class in classes]

for _class_path in class_paths:
    images = os.listdir(_class_path)
    _class = _class_path.split("/")[-1]
    print(_class)
    for image in images:
        preimg = os.path.join(_class_path, image)
        postimg = os.path.join(common_path, f"{_class}_{image}")
        os.system(f"mv {preimg} {postimg}")