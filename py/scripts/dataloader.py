import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch import Tensor

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
])


class Datasets(Dataset):
    def __init__(self, path):
        super(Datasets, self).__init__()
        self.path = path
        # self.x_y = self.load_x_y()
        self.x_y_paths = self.load_x_y_path()

    def __getitem__(self, index):
        x = preprocess(Image.open(self.x_y_paths[index]['x']))
        y = self.x_y_paths[index]['y']
        return Tensor(x), Tensor(y)

    def __len__(self):
        return len(self.x_y_paths)

    def load_x_y(self):
        cat_dir = os.path.join(self.path, 'cats')
        dog_dir = os.path.join(self.path, 'dogs')
        cat_label = [0]
        dog_label = [1]
        x_y = []
        for file in os.listdir(cat_dir):
            cat_file_path = os.path.join(cat_dir, file)
            cat = preprocess(Image.open(cat_file_path))
            x_y.append({'x': cat, 'y': cat_label})
        for file in os.listdir(dog_dir):
            dog_file_path = os.path.join(dog_dir, file)
            dog = preprocess(Image.open(dog_file_path))
            x_y.append({'x': dog, 'y': dog_label})
        return x_y

    def load_x_y_path(self):
        cat_dir = os.path.join(self.path, 'cats')
        dog_dir = os.path.join(self.path, 'dogs')
        x_y_paths = []
        cat_label = [0]
        dog_label = [1]
        for file in os.listdir(cat_dir):
            cat_file_path = os.path.join(cat_dir, file)
            x_y_paths.append({'x': cat_file_path, 'y': cat_label})
        for file in os.listdir(dog_dir):
            dog_file_path = os.path.join(dog_dir, file)
            x_y_paths.append({'x': dog_file_path, 'y': dog_label})
        return x_y_paths