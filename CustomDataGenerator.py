from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms as T
from PIL import Image

# class DatasetLoader(Dataset):
#     def __init__(self, base_dir, d_type):
#         self.X = []
#         self.D = []
#         self.Labels = []
#         self.Y = []
#         classes = os.listdir(os.path.join(base_dir, d_type))
#         for index, class_name in enumerate(classes):
#             temp_label = index
#             single_class = os.path.join(base_dir,d_type, class_name)
#             single_class_images = os.listdir(os.path.join(d_type,single_class, 'Images'))
#             # single_class_images = os.listdir(os.path.join(d_type, single_class))
#
#
#             for image_name in single_class_images:
#                 # depth_full_path = os.path.join(single_class, 'Depth', image_name)
#                 image_full_path = os.path.join(d_type, single_class,  'Images', image_name)
#                 label_full_path = os.path.join(d_type, single_class,  'Labels', image_name)
#
#                 # image_full_path = os.path.join(d_type, single_class, image_name)
#                 # label_full_path = os.path.join(d_type, single_class, image_name)
#
#                 # self.D.append(depth_full_path)
#                 self.X.append(image_full_path)
#                 self.Y.append(temp_label)
#                 self.Labels.append(label_full_path)
#         self.length = len(self.X)
#     def __len__(self):
#         return self.length
#     def __getitem__(self, index):
#         x_full_path = self.X[index]
#         y = self.Y[index]
#         label_full_path = self.Labels[index]
#         x = Image.open(x_full_path).convert('RGB')
#         width, height = x.size
#         label = Image.open(label_full_path).convert('L')
#         transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
#         x = transform(x)
#         label = transform(label)
#
#         return x , width, height,  y , label, x_full_path

class ClassifierDatasetLoader(Dataset):
    def __init__(self, base_dir, d_type):
        self.X = []
        self.D = []
        self.Labels = []
        self.Y = []
        classes = os.listdir(os.path.join(base_dir, d_type))
        for index, class_name in enumerate(classes):
            temp_label = index
            single_class = os.path.join(base_dir,d_type, class_name)
            single_class_images = os.listdir(os.path.join(d_type,single_class))
            # single_class_images = os.listdir(os.path.join(d_type, single_class))


            for image_name in single_class_images:
                # depth_full_path = os.path.join(single_class, 'Depth', image_name)
                image_full_path = os.path.join(d_type, single_class, image_name)

                # self.D.append(depth_full_path)
                self.X.append(image_full_path)
                self.Y.append(temp_label)


        self.length = len(self.X)
    def __len__(self):
        return self.length
    def __getitem__(self, index):
        x_full_path = self.X[index]
        y = self.Y[index]
        x = Image.open(x_full_path).convert('RGB')
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        x = transform(x)


        return x , y