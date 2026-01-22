from os import path as osp
from torch.utils import data as data
from torchvision.transforms.functional import normalize

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, imfrombytes, img2tensor


class FFHQDataset(data.Dataset):
    def __init__(self, opt):
        super(FFHQDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.gt_folder = opt['dataroot_gt']
        self.mean = opt['mean']
        self.std = opt['std']

        if self.io_backend_opt['type'] == 'lmdb':
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "
                                 f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.paths = [
                osp.join(self.gt_folder, f'{v:08d}.png') for v in range(70000)
            ]

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        gt_path = self.paths[index]
        img_bytes = self.file_client.get(gt_path)
        img_gt = imfrombytes(img_bytes, float32=True)

        img_gt = augment(img_gt, hflip=self.opt['use_hflip'], rotation=False)
        img_gt = img2tensor(img_gt, bgr2rgb=True, float32=True)
        normalize(img_gt, self.mean, self.std, inplace=True)
        return {'gt': img_gt, 'gt_path': gt_path}

    def __len__(self):
        return len(self.paths)
