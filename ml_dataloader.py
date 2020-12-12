from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import torch

from PIL import Image
import pandas as pd
import numpy as np 

class Dataset(torch.utils.data.Dataset):
  def __init__(self, data_npy, label, transform = None):
    data = np.load(data_npy)
    self.X = np.array(data).reshape(-1, 28, 28).astype("float32")
    self.label = np.asarray(label)
    self.transform = transform

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    item = self.X[idx]
    label = self.label[idx]

    if self.transform is not None:
      pil_image = Image.fromarray(np.uint8(item))
      item = self.transform(pil_image)
    return (item, label)

class Dataset_Transform(torch.utils.data.Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.base_dataset[index]
        return self.transform(img), target

    def __len__(self):
        return len(self.base_dataset)


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return (x-self.mean)/self.std

# transform
augmentation_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1))
    #, transforms.ToTensor()
])


default_transforms = transforms.Compose([
    transforms.ToTensor()
])


def compute_mean_std(loader):
    # mean over minibatches
    mean_img = None
    for imgs, _ in loader:
        if mean_img is None:
            mean_img = torch.zeros_like(imgs[0])
        mean_img += imgs.sum(dim=0)
    mean_img /= len(loader.dataset)

    # std over minibatches
    std_img = torch.zeros_like(mean_img)
    for imgs, _ in loader:
        std_img += ((imgs - mean_img)**2).sum(dim=0)
    std_img /= len(loader.dataset)
    std_img = torch.sqrt(std_img)

    std_img[std_img == 0] = 1

    return mean_img, std_img

    


def fashion_mnist_data_loaders(batch_size, valid_ratio, normalized=False, augment_transform=False):
    data_transforms = {'train' : transforms.ToTensor(),
                       'valid' : transforms.ToTensor(),
                       'test'  : transforms.ToTensor()}

    train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                train=True,
                                download = True
                               )

    test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                     train=False,
                                                     download = True
                                                    )

    nb_train, nb_valid = int((1.0 - valid_ratio) * len(train_dataset)), int(valid_ratio * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.dataset.random_split(train_dataset, [nb_train, nb_valid])                            

    if augment_transform:
        data_transforms['train'] = transforms.Compose([augmentation_transforms, transforms.ToTensor()])

    if normalized:
        normalizing_train = Dataset_Transform(train_dataset, transforms.ToTensor())
        normalizing_train_loader = torch.utils.data.DataLoader(dataset=normalizing_train,
                                                                batch_size=batch_size)

        mean_train, std_train = compute_mean_std(normalizing_train_loader)
        normalizing_function = Normalize(mean_train, std_train)

        for k, old_transforms in data_transforms.items():
            data_transforms[k] = transforms.Compose([old_transforms,
                                                    transforms.Lambda(lambda x: normalizing_function(x))])
    else:
        normalizing_function = None

    train_dataset = Dataset_Transform(train_dataset, data_transforms['train'])
    val_dataset = Dataset_Transform(val_dataset, data_transforms['valid'])
    test_dataset  = Dataset_Transform(test_dataset , data_transforms['test'])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                              batch_size=batch_size, 
                                              shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    '''
    # validation dataloader
    val_label_read = pd.read_csv('../input/ml-mnist-fashion/validation.csv')
    val_label = val_label_read['class']
    val_dir = '../input/ml-mnist-fashion/validation.npy'
    val_dataset = dataset(data_npy=val_dir, 
                            label=val_label, 
                            transform=default_transforms)
    val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=batch_size,
                            shuffle=False)
    '''


    
    return train_loader, val_loader, test_loader

    
    
    
'''
    
def fashion_mnist_data_loaders_normalized(batch_size, augment_data=False, normalized=False):
    train_loader, test_loader, val_loader, train_normalized, _, _ = fashion_mnist_data_loaders(batch_size, augment_data)
    
    
    if normalized:
        # dataset
        train_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                          train=True,
                                                          download = True,
                                                          transform=train_normalized,
                                                           )
        
        test_dataset = torchvision.datasets.FashionMNIST(root='./data',
                                                         train=False,
                                                         download = True,
                                                         transform=train_normalized
                                                        )
        
        
        val_label_read = pd.read_csv('../input/ml-mnist-fashion/validation.csv')
        val_label = val_label_read['class']
        val_dir = '../input/ml-mnist-fashion/validation.npy'
        val_dataset = dataset(data_npy=val_dir, 
                            label=val_label, 
                            transform=train_normalized)
        

        # loader
        train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True)
        test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False)
    
        val_loader = DataLoader(dataset=val_dataset, 
                            batch_size=batch_size,
                            shuffle=False)
        
        return train_loader, test_loader, val_loader
        
    else:
        return train_loader, test_loader, val_loader
    
 '''   
    
    
    