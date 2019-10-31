import h5py
from torch.utils import data

class H5Loader(data.Dataset):
    def __init__(self, filename):
        self.h5_file = h5py.File(filename, 'r')
        self.data = self.h5_file.get('data')
            
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def __del__(self):
        self.h5_file.close()


def get_h5_dataset(path='shoes_images/shoes.hdf5', batch_size=128, shuffle=True):
    return data.DataLoader(dataset=H5Loader(path), batch_size=batch_size, shuffle=shuffle)