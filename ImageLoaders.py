import os
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from utils import compute_loader_otsu_mask

class PairedImageSet(data.Dataset):
    def __init__(self, set_path, set_type, use_mask, size, aug):
        self.augment = aug
        self.size = size
        self.use_mask = use_mask
        self.to_tensor = transforms.ToTensor()
        clean_path_dir = '{}/{}/{}_C'.format(set_path, set_type, set_type)
        self.gt_images_path = []  # 初始化4个实例变量
        self.masks_path = []
        self.inp_images_path = []
        self.num_samples = 0

        if size is not None:
            self.resize = transforms.Resize(self.size, interpolation=InterpolationMode.BICUBIC)
        else:
            self.resize = None

        for dirpath, dnames, fnames \
                in os.walk("{}/{}/{}_A/".format(set_path, set_type, set_type)):
            for f in fnames:
                orig_path = os.path.join(dirpath, f)
                clean_path = os.path.join(clean_path_dir, os.path.splitext(f)[0] + '.jpg')
                # clean_path = os.path.join(clean_path_dir, f[:3] + "_BaseColor.jpg")
                # _A应该是input图像文件夹，但有一个input就有一个gt，所以也在clean路径中添加一个，同时注意格式为jpg
                self.gt_images_path.append(clean_path)
                self.inp_images_path.append(orig_path)
                self.num_samples += 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        inp_data = Image.open(self.inp_images_path[index])
        gt_data = Image.open(self.gt_images_path[index])
        smat_data = compute_loader_otsu_mask(inp_data, gt_data)

        if self.resize is not None:
            gt_data = self.resize(gt_data)
            smat_data = self.resize(smat_data)
            inp_data = self.resize(inp_data)

        tensor_gt = self.to_tensor(gt_data)
        tensor_msk = self.to_tensor(smat_data)
        tensor_inp = self.to_tensor(inp_data)
        return tensor_gt, tensor_msk, tensor_inp
