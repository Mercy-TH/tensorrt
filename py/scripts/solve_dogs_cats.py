import os, shutil

original_dataset_dir = "../data/dogs-vs-cats/train/train"  # 原始数据集解压目录

# 创建一个一级文件夹
base_dir = "../data/dogs-vs-cats_small"  # 保存小数据集的目录
if os.path.exists(base_dir):  # 如果不存在small文件夹则执行
    shutil.rmtree(base_dir)
os.mkdir(base_dir)  # 创建文件夹

# 创建三个二级文件夹：train、validation、test
train_dir = os.path.join(base_dir, 'train')
os.mkdir(train_dir)  # 定义训练数据文件夹并创建
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)  # 定义验证数据文件夹并创建
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)  # 定义测试数据文件夹并创建

# 创建train下cats文件夹
train_cats_dir = os.path.join(train_dir, 'cats')
os.mkdir(train_cats_dir)
# 创建train下dogs文件夹
train_dogs_dir = os.path.join(train_dir, 'dogs')
os.mkdir(train_dogs_dir)

# 创建validation下cats文件夹
validation_cats_dir = os.path.join(validation_dir, 'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir, 'dogs')
os.mkdir(validation_dogs_dir)  # 在训练集文件夹下定义dogs文件夹

test_cats_dir = os.path.join(test_dir, 'cats')
os.mkdir(test_cats_dir)  # 在训练集文件夹下定义cats文件夹
test_dogs_dir = os.path.join(test_dir, 'dogs')
os.mkdir(test_dogs_dir)  # 在训练集文件夹下定义dogs文件夹

# 将原始数据集的train中cat和dog各前1000张复制到train/cats和train/dogs
for className in ["cat", "dog"]:
    imageNames = ["{}.{}.jpg".format(className, i) for i in range(1000)]
    for name in imageNames:
        src = os.path.join(original_dataset_dir, name)
        if className == "cat":
            dst = os.path.join(train_cats_dir, name)
        else:
            dst = os.path.join(train_dogs_dir, name)
        shutil.copyfile(src, dst)

# 将原始数据集的train中cat和dog各1000~1500张复制到validation/cats和validation/dogs
for className in ["cat", "dog"]:
    imageNames = ["{}.{}.jpg".format(className, i) for i in range(1000, 1500)]
    for name in imageNames:
        src = os.path.join(original_dataset_dir, name)
        if className == "cat":
            dst = os.path.join(validation_cats_dir, name)
        else:
            dst = os.path.join(validation_dogs_dir, name)
        shutil.copyfile(src, dst)

# 将原始数据集的train中cat和dog各1500~2000张复制到test/cats和test/dogs
for className in ["cat", "dog"]:
    imageNames = ["{}.{}.jpg".format(className, i) for i in range(1500, 2000)]
    for name in imageNames:
        src = os.path.join(original_dataset_dir, name)
        if className == "cat":
            dst = os.path.join(test_cats_dir, name)
        else:
            dst = os.path.join(test_dogs_dir, name)
        shutil.copyfile(src, dst)
