import torch
import pandas as pd
import PIL
import os
import glob

class CSVdataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        
        self.df = pd.read_csv(csv_path)
        
        self.images_folder = images_folder
        self.images= glob.glob(f"{images_folder}/*.png")
        #self.images = [im.split("/")[-1].split(".")[0] for im in images]
        self.transform = transform
        self.class2index = {"airplane":0, "automobile":1, "bird": 2, "cat": 3, "deer": 4, "dog": 5, "frog": 6, "horse": 7, "ship": 8, "truck": 9}

    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        filename = self.images[index]
       
        id = int(filename.split("/")[-1].split(".")[0])
        label = self.df.loc[self.df.id == id]["label"].tolist()[0]
        print(label)
        label = self.class2index[label]
        image = PIL.Image.open(os.path.join(self.images_folder, f"{filename}"))
        if self.transform is not None:
            image = self.transform(image)
        return image, label
        