import os
import struct
from array import array
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader


class myMnistReader(Dataset):
    
    train_img_file = "train-images-idx3-ubyte"
    train_label_file = "train-labels-idx1-ubyte"
    test_img_file = "t10k-images-idx3-ubyte"
    test_label_file = "t10k-labels-idx1-ubyte"
    
    def __init__(self, mode="Train", transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        if mode == "Train":
            self.images = self.load_images(train_img_file)
            self.labels = self.load_labels(train_label_file)
        else:    
            self.images = self.load_images(test_img_file)
            self.labels = self.load_labels(test_label_file)        
        
    def __len__(self):
        
        return len(self.labels)
    
    def __getitem__(self, idx):
        

        img = self.images[idx]
        img = np.array(img, dtype=np.uint8)
        img = img.reshape(28,28)
        img = Image.fromarray(img, mode='L')        
                
        
        target = self.labels[idx]
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target

    data_dir = "data"




    def load_labels(self,fileName):
        file_path = os.path.join(data_dir, fileName)
        file = open(file_path, "rb")
        magic, size = struct.unpack(">II", file.read(8))
    
        if magic != 2049:
            raise ValueError("Wrong magic")
    
        labels = array("B", file.read())
        return labels
        
    def load_images(self, fileName):
        file_path = os.path.join(data_dir, fileName)
        file = open(file_path, "rb")
        magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
    
        if magic != 2051:
            raise ValueError("Wrong magic")
    
        image_data = array("B", file.read())
    
    
        images = []
    
        for i in range(size):
            images.append([0] * rows * cols)

            for i in range(size):
                images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]


        return images

    def show_images(self, images, labels, idx, num):
    
        for i in range(num):
            img = np.reshape(images[idx+i],(28,28))
            ax = plt.subplot(num/8, 8, i + 1)
            ax.set_title(labels[idx+i])
            plt.imshow(img)
            plt.show()
        
