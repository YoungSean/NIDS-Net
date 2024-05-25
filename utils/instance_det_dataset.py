# import PIL
# PIL.Image.ANTIALIAS = PIL.Image.LANCZOS
import glob
import json
import math
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional
from PIL import Image, ImageFile

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def find_mask_bbox(mask_array):
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return (cmin, rmin, cmax+1, rmax+1)  # PIL uses (left, upper, right, lower)


class BaseDataset(Dataset):
    def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0, freq=1, obj_folder_pattern='*', file_pattern='*g',
                 mask_pattern='*g'):
        self.samples = []
        self.sample_masks = []
        self.transform = transform
        self.imsize = imsize

        source_list = sorted(glob.glob(os.path.join(data_dir, obj_folder_pattern)))
        self.num_template = -1
        self.objID_offset = objID_offset
        for i, source_dir in enumerate(source_list):
            image_paths = sorted(glob.glob(os.path.join(source_dir, file_pattern)))[::freq]
            if self.num_template == -1:
                self.num_template = len(image_paths)
            if len(image_paths) != self.num_template:
                print(f"Object {source_dir} has different number of images (!= {self.num_template})")
            mask_paths = sorted(glob.glob(os.path.join(source_dir, mask_pattern)))[::freq]

            self.samples.extend(image_paths)
            self.sample_masks.extend(mask_paths)
            assert len(image_paths) == self.num_template, "The number of images should be equal to num_template"

    def __len__(self):
        return len(self.samples)

    def get_resized_mask(self, mask):
        mask_size = self.imsize // 14
        resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
        resized_mask_numpy = np.array(resized_mask) / 255.0
        tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
        tensor_mask[tensor_mask > 0.5] = 1.0
        tensor_mask = tensor_mask.unsqueeze(0).long()
        if tensor_mask.sum() == 0:
            tensor_mask = torch.ones_like(tensor_mask)
        return tensor_mask

    def __getitem__(self, index):
        path = self.samples[index]
        label = index // self.num_template + self.objID_offset
        mask_path = self.sample_masks[index]

        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        w, h = img.size
        if self.imsize and min(w, h) > self.imsize:
            img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
            mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)
        else:
            new_w, new_h = math.ceil(w / 14) * 14, math.ceil(h / 14) * 14
            img, mask = img.resize((new_w, new_h), Image.LANCZOS), mask.resize((new_w, new_h), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)
            mask = self.get_resized_mask(mask)

        return img, label, mask

class MaskedImageDataset(BaseDataset):
    def __init__(self, data_dir, json_path, transform=None, imsize=448, objID_offset=0, freq=1, obj_folder_pattern='*', file_pattern='*g', mask_pattern='*g'):
        super().__init__(data_dir, None, imsize, objID_offset, freq, obj_folder_pattern, file_pattern, mask_pattern)
        self.object_features = self.load_json_tensors(json_path)
        self.masked_transform = transform

    def load_json_tensors(self, json_path):
        # Load JSON file and convert to a dictionary of tensors
        with open(json_path, 'r') as file:
            feat_dict = json.load(file)

        object_features = torch.Tensor(feat_dict['features']).cuda()
        object_features = nn.functional.normalize(object_features, dim=1, p=2)
        return object_features

    def __getitem__(self, index):
        img, label, mask = super().__getitem__(index)
        if mask is not None:
            # Convert the PIL mask to a binary mask where 1s are where the mask is not zero
            mask_array = np.array(mask)
            x0, y0, x1, y1 = find_mask_bbox(mask_array)
            binary_mask = (mask_array > 0).astype(np.uint8)

            # Apply the mask to each channel of the image
            img_array = np.array(img)
            # get masked part, non-mask part is white
            # masked_img_array = np.where(np.stack([binary_mask]*3, axis=-1), img_array, 255 * np.ones_like(img_array))
            masked_img_array = img_array * np.stack([binary_mask]*3, axis=-1)  # Stack the mask across the channel dimension

            # Convert the masked image back to a PIL Image
            # masked_img = Image.fromarray(masked_img_array)
            # masked_img.show()

            # show mask
            cropped_img = masked_img_array[y0:y1, x0:x1]
            cropped_img = Image.fromarray(cropped_img)
            # cropped_img.show()

        # Apply transformations
        if self.masked_transform:
            img = self.masked_transform(cropped_img)

        # Retrieve the additional tensor for this index, if available
        dinov2_object_feature = self.object_features[index]

        return img, label, dinov2_object_feature

class FewSOLRealObjects(BaseDataset):
    def __init__(self, data_dir, transform=None, imsize=448, objID_offset=100):
        super().__init__(data_dir, transform, imsize, objID_offset, freq=1, file_pattern='*color.jpg', mask_pattern='*label*')

class SimpleInstanceDataset(BaseDataset):
    def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0):
        super().__init__(data_dir, transform, imsize, objID_offset, freq=1, file_pattern='images/*g', mask_pattern='masks/*g')


class BOPDataset(BaseDataset):
    def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0, freq=10):
        super().__init__(data_dir, transform, imsize, objID_offset, freq, 'obj_*','rgb/*g', 'mask/*g')

class OWIDDataset(BaseDataset):
    def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0, freq=4):
        super().__init__(data_dir, transform, imsize, objID_offset, freq, 'rgb/*g', 'mask/*g')


class RealWorldDataset(Dataset):
    def __init__(self, data_dir, dataset, data=None, transform=None, imsize=None):

        if dataset == 'Object':

            num_obj = []
            image_dir = []
            mask_dir = []
            count = []

            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            for _, source_dir in enumerate(source_list):
                num_obj.append(source_dir.split('/')[-1].split('.')[0])
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'images', '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'masks', '*'))
                                     if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                mask_dir.extend(mask_paths)
                count.append(len(image_paths))

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['mask_dir'] = mask_dir
            cfg['obj_name'] = num_obj  # object lists for Object
            cfg['length'] = count

            self.samples = cfg['image_dir']

        elif dataset == 'Scene':

            num_scene = []
            image_dir = []
            proposals = []
            count = []

            with open(os.path.join(os.path.dirname(data_dir),
                                   'proposals_on_' + data_dir.split('/')[-1] + '.json')) as f:
                proposal_json = json.load(f)

            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            for idx, source_dir in enumerate(source_list):
                scene_name = source_dir.split('/')[-1]
                num_scene.append(scene_name)

                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                count.append(len(image_paths))
                proposals.extend(proposal_json[scene_name])

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['proposals'] = proposals
            cfg['scene_name'] = num_scene  # scene list for Scene
            cfg['length'] = count

            self.samples = cfg['image_dir']

        else: # for demo scene image
            with open(os.path.join(data_dir, 'proposals_on_' + dataset + '.json')) as f:
                proposal_json = json.load(f)

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = None
            cfg['proposals'] = proposal_json
            cfg['scene_name'] = [dataset]  # scene list for Scene
            cfg['length'] = [len(data)]

            self.samples = data


        self.cfg = cfg
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):


        if "test" in self.cfg['dataset']: #  or 'rgb' in self.cfg['dataset']
            img = self.samples[index]
        else:
            path = self.samples[index]
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')

        w, h = img.size

        if (self.imsize is not None) and (min(w, h) > self.imsize):
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        else:
            new_w = math.ceil(w / 14) * 14
            new_h = math.ceil(h / 14) * 14
            img = img.resize((new_w, new_h), Image.ANTIALIAS)



        if self.transform is not None:
            img = self.transform(img)
        return img, index

class RealWorldDatasetWithMask(Dataset):
    def __init__(self, data_dir, dataset, data=None, transform=None, imsize=None):
        self.sample_masks = None
        if dataset == 'Object':

            num_obj = []
            image_dir = []
            mask_dir = []
            count = []

            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            for _, source_dir in enumerate(source_list):
                num_obj.append(source_dir.split('/')[-1].split('.')[0])
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'images', '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'masks', '*'))
                                     if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                mask_dir.extend(mask_paths)
                count.append(len(image_paths))

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['mask_dir'] = mask_dir
            cfg['obj_name'] = num_obj  # object lists for Object
            cfg['length'] = count

            self.samples = cfg['image_dir']
            self.sample_masks = cfg['mask_dir']

        elif dataset == 'Scene':

            num_scene = []
            image_dir = []
            proposals = []
            count = []

            with open(os.path.join(os.path.dirname(data_dir),
                                   'proposals_on_' + data_dir.split('/')[-1] + '.json')) as f:
                proposal_json = json.load(f)

            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            for idx, source_dir in enumerate(source_list):
                scene_name = source_dir.split('/')[-1]
                num_scene.append(scene_name)

                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                count.append(len(image_paths))
                proposals.extend(proposal_json[scene_name])

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['proposals'] = proposals
            cfg['scene_name'] = num_scene  # scene list for Scene
            cfg['length'] = count

            self.samples = cfg['image_dir']

        else: # for demo scene image
            with open(os.path.join(data_dir, 'proposals_on_' + dataset + '.json')) as f:
                proposal_json = json.load(f)

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = None
            cfg['proposals'] = proposal_json
            cfg['scene_name'] = [dataset]  # scene list for Scene
            cfg['length'] = [len(data)]

            self.samples = data


        self.cfg = cfg
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        mask = None

        if "test" in self.cfg['dataset']: #  or 'rgb' in self.cfg['dataset']
            img = self.samples[index]
        else:
            path = self.samples[index]
            if self.sample_masks:
                mask_path = self.sample_masks[index]
                mask = Image.open(mask_path)
            ImageFile.LOAD_TRUNCATED_IMAGES = True

            with open(path, 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')

        w, h = img.size

        if (self.imsize is not None) and (min(w, h) > self.imsize):
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        else:
            new_w = math.ceil(w / 14) * 14
            new_h = math.ceil(h / 14) * 14
            img = img.resize((new_w, new_h), Image.ANTIALIAS)
            if mask:
                mask = mask.resize((new_w, new_h), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)
        return img, index, mask

class InstanceDataset(Dataset):
    def __init__(self, data_dir, dataset, data=None, transform=None, imsize=None):
        self.sample_masks = None
        if dataset == 'Object':

            num_obj = []
            image_dir = []
            mask_dir = []
            count = []

            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            for _, source_dir in enumerate(source_list):
                num_obj.append(source_dir.split('/')[-1].split('.')[0])
                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'images', '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                mask_paths = sorted([p for p in glob.glob(os.path.join(source_dir, 'masks', '*'))
                                     if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                mask_dir.extend(mask_paths)
                count.append(len(image_paths))

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['mask_dir'] = mask_dir
            cfg['obj_name'] = num_obj  # object lists for Object
            cfg['length'] = count

            self.samples = cfg['image_dir']
            self.sample_masks = cfg['mask_dir']

        elif dataset == 'Scene':

            num_scene = []
            image_dir = []
            proposals = []
            count = []

            with open(os.path.join(os.path.dirname(data_dir),
                                   'proposals_on_' + data_dir.split('/')[-1] + '.json')) as f:
                proposal_json = json.load(f)

            source_list = sorted(glob.glob(os.path.join(data_dir, '*')))

            for idx, source_dir in enumerate(source_list):
                scene_name = source_dir.split('/')[-1]
                num_scene.append(scene_name)

                image_paths = sorted([p for p in glob.glob(os.path.join(source_dir, '*'))
                                      if re.search('/*\.(jpg|jpeg|png|gif|bmp|pbm)', str(p))])
                image_dir.extend(image_paths)
                count.append(len(image_paths))
                proposals.extend(proposal_json[scene_name])

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = image_dir
            cfg['proposals'] = proposals
            cfg['scene_name'] = num_scene  # scene list for Scene
            cfg['length'] = count

            self.samples = cfg['image_dir']

        else: # for demo scene image
            with open(os.path.join(data_dir, 'proposals_on_' + dataset + '.json')) as f:
                proposal_json = json.load(f)

            cfg = dict()
            cfg['dataset'] = dataset
            cfg['data_dir'] = data_dir
            cfg['image_dir'] = None
            cfg['proposals'] = proposal_json
            cfg['scene_name'] = [dataset]  # scene list for Scene
            cfg['length'] = [len(data)]

            self.samples = data


        self.cfg = cfg
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def get_resized_mask(self, mask):
        mask_size = self.imsize // 14
        resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
        resized_mask_numpy = np.array(resized_mask)
        # show resized mask
        # plt.imshow(resized_mask_numpy)
        # plt.show()
        resized_mask_numpy = resized_mask_numpy / 255.0
        tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
        tensor_mask[tensor_mask > 0.5] = 1.0
        tensor_mask = tensor_mask.unsqueeze(0).long()  # .to(self.device)
        if tensor_mask.sum() == 0:
            tensor_mask = torch.ones_like(tensor_mask)
        return tensor_mask

    def __getitem__(self, index):
        # mask = None
        #
        # if "test" in self.cfg['dataset']: #  or 'rgb' in self.cfg['dataset']
        #     img = self.samples[index]
        # else:
        path = self.samples[index]
        label = int(path.split('/')[-3][:3])
        # if self.sample_masks:
        mask_path = self.sample_masks[index]
        mask = Image.open(mask_path)
        mask = mask.convert('L')
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        w, h = img.size

        if (self.imsize is not None) and (min(w, h) > self.imsize):
            img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
            mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)
        else:
            new_w = math.ceil(w / 14) * 14
            new_h = math.ceil(h / 14) * 14
            img = img.resize((new_w, new_h), Image.LANCZOS)
            mask = mask.resize((new_w, new_h), Image.BILINEAR)

        if self.transform is not None:
            img = self.transform(img)
            mask = self.get_resized_mask(mask)
        return img, label, mask


class MVImgDataset(BaseDataset):
    def __init__(self, data_dir, transform=None, imsize=448, freq=3, num_template=24, objID_offset=0):
        super().__init__(data_dir, transform, imsize)
        self.freq = freq
        self.num_template = num_template
        self.samples = []
        self.sample_masks = []
        self.objID_offset = objID_offset

        source_list = sorted(glob.glob(os.path.join(data_dir, '*', '*')))
        print("object number: ", len(source_list))
        for _, source_dir in enumerate(source_list):
            mask_source_dir = source_dir.replace('/data/', '/mask/')
            if not os.path.exists(mask_source_dir) or not os.path.isdir(mask_source_dir):
                continue
            obj_image_paths = sorted(glob.glob(os.path.join(source_dir, 'images', '[0-9][0-9][0-9].jpg')))
            if len(obj_image_paths) < num_template:  # the object has a small number of multiview images
                continue
            elif len(obj_image_paths) >= freq * num_template:
                image_paths = obj_image_paths[::freq]
            elif len(obj_image_paths) >= 2 * num_template:
                image_paths = obj_image_paths[::2]  # change the frequency
            else:
                image_paths = obj_image_paths

            image_paths = image_paths[:num_template]
            assert len(image_paths) == num_template, "The number of images should be equal to num_template"

            mask_paths = []
            mask_exist = True
            for img_path in image_paths:
                words = img_path.replace('/data/', '/mask/') + '.png'
                words = words.split("/")
                mask_path = '/'.join(words[:-2] + [words[-1]])
                if not os.path.exists(mask_path) or not os.path.isfile(mask_path):
                    mask_exist = False
                mask_paths.append(mask_path)
            if not mask_exist:
                continue

            self.samples.extend(image_paths)
            self.sample_masks.extend(mask_paths)

        print(f'Valid object number: {len(self.samples) // num_template}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path = self.samples[index]
        mask_path = self.sample_masks[index]

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        if self.imsize:
            img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
            mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)

        if self.transform:
            img = self.transform(img)
            mask = self.get_resized_mask(mask)

        label = index // self.num_template + self.objID_offset
        return img, label, mask


class MaskedMVImgDataset(MVImgDataset):
    def __init__(self, data_dir, json_path, transform=None, masked_img_transform=None, imsize=448, freq=3, num_template=24,objID_offset=0):
        super().__init__(data_dir, None, imsize, freq, num_template, objID_offset=objID_offset)
        self.masked_transform = masked_img_transform
        self.object_features = self.load_json_tensors(json_path)

    def load_json_tensors(self, json_path):
        with open(json_path, 'r') as file:
            feat_dict = json.load(file)

        object_features = torch.tensor(feat_dict['features'])
        object_features = torch.nn.functional.normalize(object_features, dim=1, p=2)
        return object_features

    def __getitem__(self, index):
        img, label, mask = super().__getitem__(index)
        masked_img = self.apply_mask(img, mask)

        # Apply masked transformations if any
        if self.masked_transform:
            masked_img = self.masked_transform(masked_img)

        # Retrieve the object feature tensor for this index
        dinov2_object_feature = self.object_features[index]

        return masked_img, label, dinov2_object_feature

    def apply_mask(self, img, mask):
        """Applies the mask to the image, making unmasked areas transparent."""
        # Convert the PIL mask to a binary mask
        mask_array = np.array(mask)
        binary_mask = (mask_array > 0).astype(np.uint8)

        # Apply the mask to each channel of the image
        img_array = np.array(img)
        masked_img_array = img_array * np.stack([binary_mask] * 3, axis=-1)

        # Convert the masked image back to a PIL Image
        masked_img = Image.fromarray(masked_img_array)
        return masked_img

class SAM6DBOPDataset(BaseDataset):
    def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0, freq=1):
        super().__init__(data_dir, transform, imsize, objID_offset, freq, file_pattern='rgb_*', mask_pattern='mask_*')


class BOPSceneDataset(Dataset):
    def __init__(self, data_dir, dataset, data=None, transform=None, imsize=None):

        with open(os.path.join(data_dir, 'proposals_on_' + dataset + '.json')) as f:
            proposal_json = json.load(f)

        cfg = dict()
        cfg['dataset'] = dataset
        cfg['data_dir'] = data_dir
        cfg['image_dir'] = None
        cfg['proposals'] = proposal_json
        cfg['scene_name'] = [dataset]  # scene list for Scene
        cfg['length'] = [len(data)]

        self.samples = data

        self.cfg = cfg
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img = self.samples[index]
        w, h = img.size

        if (self.imsize is not None) and (min(w, h) > self.imsize):
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        else:
            new_w = math.ceil(w / 14) * 14
            new_h = math.ceil(h / 14) * 14
            img = img.resize((new_w, new_h), Image.ANTIALIAS)

        if self.transform is not None:
            img = self.transform(img)
        return img, index

if __name__ == '__main__':

    img_size = 448
    # Define transformations to be applied to the images
    transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    # Create an instance of your custom dataset
    # instance_dataset = FewSOLRealObjects(data_dir='../FewSOL/data/real_objects', transform=transform, imsize=448)
    # Lmo = BOPDataset(data_dir='../datasets/lmo/test_video', transform=transform, imsize=448)
    # FewSOL_dataset = FewSOLRealObjects(data_dir='../FewSOL/data/real_objects', transform=transform, imsize=448)
    # InsDet_dataset = SimpleInstanceDataset(data_dir='../database/Objects', transform=transform, imsize=448)
    masked_ins_det_dataset = MaskedImageDataset(data_dir='../database/Objects', json_path='../obj_FFA/object_features_vitl14_reg.json', transform=transform, file_pattern='images/*g', mask_pattern='masks/*g')
    print(len(masked_ins_det_dataset))
    # print(masked_ins_det_dataset[0])

    # Create a data loader
    batch_size = 2
    # data_loader = DataLoader(InsDet_dataset, batch_size=batch_size, shuffle=False)
    data_loader = DataLoader(masked_ins_det_dataset, batch_size=batch_size, shuffle=False)

    # # Iterate over the data loader to get batches of data
    # for images, labels, masks in data_loader:
    #     # Your training/inference code here
    #     print(images.shape)  # Shape of the batch of images
    #     print(masks.shape)  # Shape of the batch of masks
    #     print(labels)  # Batch of corresponding labels
    #     break  # For demonstration, break after the first batch

    # # Iterate over the data loader to get batches of data
    for masked_images, labels, features in data_loader:
        # Your training/inference code here
        print(masked_images.shape)  # Shape of the batch of images
        print(features.shape)  # Shape of the batch of masks
        print(labels)  # Batch of corresponding labels
        break  # For demonstration, break after the first batch


"""
    Ignore the following code, it is for recording the original dataset classes
"""

# class FewSOLRealObjects(Dataset):
#     def __init__(self, data_dir, transform=None, imsize=448, objID_offset=100):
#         num_obj = []
#         image_dir = []
#         mask_dir = []
#         count = []
#
#         source_list = sorted(glob.glob(os.path.join(data_dir, '*')))
#         self.obj_map = {}
#         for i, source_dir in enumerate(source_list):
#             self.obj_map[source_dir.split('/')[-1]] = i + objID_offset
#
#         for _, source_dir in enumerate(source_list):
#             num_obj.append(source_dir.split('/')[-1].split('.')[0])
#             image_paths = sorted(glob.glob(os.path.join(source_dir, '*color.jpg')))
#             image_dir.extend(image_paths)
#             mask_paths = sorted(glob.glob(os.path.join(source_dir, '*label*')))
#             mask_dir.extend(mask_paths)
#             count.append(len(image_paths))
#
#         cfg = dict()
#         cfg['data_dir'] = data_dir
#         cfg['image_dir'] = image_dir
#         cfg['mask_dir'] = mask_dir
#         cfg['obj_name'] = num_obj  # object lists for Object
#         cfg['length'] = count
#
#         self.samples = cfg['image_dir']
#         self.sample_masks = cfg['mask_dir']
#
#         self.cfg = cfg
#         self.transform = transform
#         self.imsize = imsize
#
#     def __len__(self):
#         return len(self.samples)
#
#     def get_resized_mask(self, mask):
#         mask_size = self.imsize // 14
#         resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
#         resized_mask_numpy = np.array(resized_mask)
#         # show resized mask
#         # plt.imshow(resized_mask_numpy)
#         # plt.show()
#         resized_mask_numpy = resized_mask_numpy / 255.0
#         tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
#         tensor_mask[tensor_mask > 0.5] = 1.0
#         tensor_mask = tensor_mask.unsqueeze(0).long()  # .to(self.device)
#         if tensor_mask.sum() == 0:
#             tensor_mask = torch.ones_like(tensor_mask)
#         return tensor_mask
#
#     def __getitem__(self, index):
#         path = self.samples[index]
#         label = self.obj_map[path.split('/')[-2]]
#         # if self.sample_masks:
#         mask_path = self.sample_masks[index]
#         mask = Image.open(mask_path)
#         mask = mask.convert('L')
#         ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             img = img.convert('RGB')
#
#         w, h = img.size
#
#         if (self.imsize is not None) and (min(w, h) > self.imsize):
#             img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
#             mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)
#         else:
#             new_w = math.ceil(w / 14) * 14
#             new_h = math.ceil(h / 14) * 14
#             img = img.resize((new_w, new_h), Image.LANCZOS)
#             mask = mask.resize((new_w, new_h), Image.BILINEAR)
#
#         if self.transform is not None:
#             img = self.transform(img)
#         # mask = self.get_resized_mask(mask)  # transform mask to tensor
#         return img, label, mask

# class BOPDataset(Dataset):
#     def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0, freq=10):
#         num_obj = []
#         image_dir = []
#         mask_dir = []
#         count = []
#
#         source_list = sorted(glob.glob(os.path.join(data_dir, 'obj_*')))
#         self.obj_map = {}
#         for i, source_dir in enumerate(source_list):
#             self.obj_map[int(source_dir[-6:])] = int(source_dir[-6:])+ objID_offset
#
#         for _, source_dir in enumerate(source_list):
#             # num_obj.append(source_dir.split('/')[-1].split('.')[0])
#             obj_image_paths = sorted(glob.glob(os.path.join(source_dir, 'rgb','*g')))
#             # we only pick 16 images for each object from 160 templete images
#             image_paths = obj_image_paths[::freq]
#             image_dir.extend(image_paths)
#             obj_mask_paths = sorted(glob.glob(os.path.join(source_dir, 'mask','*g')))
#             mask_paths = obj_mask_paths[::freq]
#             mask_dir.extend(mask_paths)
#             count.append(len(image_paths))
#
#         cfg = dict()
#         cfg['data_dir'] = data_dir
#         cfg['image_dir'] = image_dir
#         cfg['mask_dir'] = mask_dir
#         cfg['obj_name'] = num_obj  # object lists for Object
#         cfg['length'] = count
#
#         self.samples = cfg['image_dir']
#         self.sample_masks = cfg['mask_dir']
#
#         self.cfg = cfg
#         self.transform = transform
#         self.imsize = imsize
#
#     def __len__(self):
#         return len(self.samples)
#
#     def get_resized_mask(self, mask):
#         mask_size = self.imsize // 14
#         resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
#         resized_mask_numpy = np.array(resized_mask)
#         # show resized mask
#         # plt.imshow(resized_mask_numpy)
#         # plt.show()
#         resized_mask_numpy = resized_mask_numpy / 255.0
#         tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
#         tensor_mask[tensor_mask > 0.5] = 1.0
#         tensor_mask = tensor_mask.unsqueeze(0).long()  # .to(self.device)
#         if tensor_mask.sum() == 0:
#             tensor_mask = torch.ones_like(tensor_mask)
#         return tensor_mask
#
#     def __getitem__(self, index):
#         path = self.samples[index]
#         label = self.obj_map[int(path.split('/')[-3][-6:])]
#         # if self.sample_masks:
#         mask_path = self.sample_masks[index]
#         mask = Image.open(mask_path)
#         mask = mask.convert('L')
#         ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             img = img.convert('RGB')
#
#         w, h = img.size
#
#         if (self.imsize is not None) and (min(w, h) > self.imsize):
#             img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
#             mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)
#         else:
#             new_w = math.ceil(w / 14) * 14
#             new_h = math.ceil(h / 14) * 14
#             img = img.resize((new_w, new_h), Image.LANCZOS)
#             mask = mask.resize((new_w, new_h), Image.BILINEAR)
#
#         if self.transform is not None:
#             img = self.transform(img)
#             mask = self.get_resized_mask(mask)  # transform mask to tensor
#         return img, label, mask
#
# class OWIDDataset(Dataset):
#     def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0, freq=4):
#         num_obj = []
#         image_dir = []
#         mask_dir = []
#         count = []
#
#         source_list = []
#         for s in sorted(glob.glob(os.path.join(data_dir, '*'))):
#             if s.split('/')[-1].isdigit():
#                 source_list.append(s)
#         print("object number: ", len(source_list))
#         self.obj_map = {}
#         for i, source_dir in enumerate(source_list):
#             self.obj_map[int(source_dir.split('/')[-1])] = int(source_dir.split('/')[-1]) + objID_offset
#
#         for _, source_dir in enumerate(source_list):
#             # num_obj.append(source_dir.split('/')[-1].split('.')[0])
#             obj_image_paths = sorted(glob.glob(os.path.join(source_dir, 'rgb','*g')))
#             # we only pick 16 images for each object from 160 templete images
#             image_paths = obj_image_paths[::freq]
#             image_dir.extend(image_paths)
#             obj_mask_paths = sorted(glob.glob(os.path.join(source_dir, 'mask','*g')))
#             mask_paths = obj_mask_paths[::freq]
#             mask_dir.extend(mask_paths)
#             count.append(len(image_paths))
#
#         cfg = dict()
#         cfg['data_dir'] = data_dir
#         cfg['image_dir'] = image_dir
#         cfg['mask_dir'] = mask_dir
#         cfg['obj_name'] = num_obj  # object lists for Object
#         cfg['length'] = count
#
#         self.samples = cfg['image_dir']
#         self.sample_masks = cfg['mask_dir']
#
#         self.cfg = cfg
#         self.transform = transform
#         self.imsize = imsize
#
#     def __len__(self):
#         return len(self.samples)
#
#     def get_resized_mask(self, mask):
#         mask_size = self.imsize // 14
#         resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
#         resized_mask_numpy = np.array(resized_mask)
#         # show resized mask
#         # plt.imshow(resized_mask_numpy)
#         # plt.show()
#         resized_mask_numpy = resized_mask_numpy / 255.0
#         tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
#         tensor_mask[tensor_mask > 0.5] = 1.0
#         tensor_mask = tensor_mask.unsqueeze(0).long()  # .to(self.device)
#         if tensor_mask.sum() == 0:
#             tensor_mask = torch.ones_like(tensor_mask)
#         return tensor_mask
#
#     def __getitem__(self, index):
#         path = self.samples[index]
#         label = self.obj_map[int(path.split('/')[-3][-6:])]
#         # if self.sample_masks:
#         mask_path = self.sample_masks[index]
#         mask = Image.open(mask_path)
#         mask = mask.convert('L')
#         ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             img = img.convert('RGB')
#
#         w, h = img.size
#
#         if (self.imsize is not None) and (min(w, h) > self.imsize):
#             img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
#             mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)
#         else:
#             new_w = math.ceil(w / 14) * 14
#             new_h = math.ceil(h / 14) * 14
#             img = img.resize((new_w, new_h), Image.LANCZOS)
#             mask = mask.resize((new_w, new_h), Image.BILINEAR)
#
#         if self.transform is not None:
#             img = self.transform(img)
#             mask = self.get_resized_mask(mask)  # transform mask to tensor
#         return img, label, mask
# class SAM6DBOPDataset(Dataset):
#     def __init__(self, data_dir, transform=None, imsize=448, objID_offset=0, freq=1):
#         num_obj = []
#         image_dir = []
#         mask_dir = []
#         count = []
#
#         source_list = sorted(glob.glob(os.path.join(data_dir, 'obj_*')))
#         self.obj_map = {}
#         for i, source_dir in enumerate(source_list):
#             self.obj_map[int(source_dir[-6:])] = int(source_dir[-6:])+ objID_offset
#
#         for _, source_dir in enumerate(source_list):
#             # num_obj.append(source_dir.split('/')[-1].split('.')[0])
#             obj_image_paths = sorted(glob.glob(os.path.join(source_dir, 'rgb_*')))
#             # we only pick 16 images for each object from 160 templete images
#             image_paths = obj_image_paths[::freq]
#             image_dir.extend(image_paths)
#             obj_mask_paths = sorted(glob.glob(os.path.join(source_dir, 'mask_*')))
#             mask_paths = obj_mask_paths[::freq]
#             mask_dir.extend(mask_paths)
#             count.append(len(image_paths))
#
#         cfg = dict()
#         cfg['data_dir'] = data_dir
#         cfg['image_dir'] = image_dir
#         cfg['mask_dir'] = mask_dir
#         cfg['obj_name'] = num_obj  # object lists for Object
#         cfg['length'] = count
#
#         self.samples = cfg['image_dir']
#         self.sample_masks = cfg['mask_dir']
#
#         self.cfg = cfg
#         self.transform = transform
#         self.imsize = imsize
#
#     def __len__(self):
#         return len(self.samples)
#
#     def get_resized_mask(self, mask):
#         mask_size = self.imsize // 14
#         resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
#         resized_mask_numpy = np.array(resized_mask)
#         # show resized mask
#         # plt.imshow(resized_mask_numpy)
#         # plt.show()
#         resized_mask_numpy = resized_mask_numpy / 255.0
#         tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
#         tensor_mask[tensor_mask > 0.5] = 1.0
#         tensor_mask = tensor_mask.unsqueeze(0).long()  # .to(self.device)
#         if tensor_mask.sum() == 0:
#             tensor_mask = torch.ones_like(tensor_mask)
#         return tensor_mask
#
#     def __getitem__(self, index):
#         path = self.samples[index]
#         label = self.obj_map[int(path.split('/')[-2][-6:])]
#         # if self.sample_masks:
#         mask_path = self.sample_masks[index]
#         mask = Image.open(mask_path)
#         mask = mask.convert('L')
#         ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             img = img.convert('RGB')
#         w, h = img.size
#
#         if (self.imsize is not None) and (min(w, h) > self.imsize):
#             img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
#             mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)
#         else:
#             new_w = math.ceil(w / 14) * 14
#             new_h = math.ceil(h / 14) * 14
#             img = img.resize((new_w, new_h), Image.LANCZOS)
#             mask = mask.resize((new_w, new_h), Image.BILINEAR)
#
#         if self.transform is not None:
#             img = self.transform(img)
#         # mask = self.get_resized_mask(mask)  # transform mask to tensor
#         return img, label, mask
# class MVImgDataset(Dataset):
#     def __init__(self, data_dir, transform=None, imsize=448, freq=3, num_template=24):
#         num_obj = []
#         image_dir = []
#         mask_dir = []
#         count = []
#
#         source_list = sorted(glob.glob(os.path.join(data_dir, '*', '*')))
#         print("object number: ", len(source_list))
#         # self.obj_map = {}
#         # for i, source_dir in enumerate(source_list):
#         #     self.obj_map[int(source_dir.split('/')[-1])] = int(source_dir.split('/')[-1])
#
#         for _, source_dir in enumerate(source_list):
#             mask_source_dir = source_dir.replace('/data/', '/mask/')
#             if not os.path.exists(mask_source_dir) or not os.path.isdir(mask_source_dir):
#                 continue
#             obj_image_paths = sorted(glob.glob(os.path.join(source_dir, 'images','[0-9][0-9][0-9].jpg')))
#             if len(obj_image_paths) < num_template: # the object has a small number of multiview images
#                 continue
#             elif len(obj_image_paths) >= freq * num_template:
#                 image_paths = obj_image_paths[::freq]
#             elif len(image_paths)  >= 2 * num_template:
#                 image_paths = obj_image_paths[::2] # change the frequency
#             else:
#                 image_paths = obj_image_paths
#
#             image_paths = image_paths[:num_template]
#             assert len(image_paths)==num_template, "The number of images should be equal to num_template"
#
#             mask_paths = []
#             mask_exist = True
#             for img_path in image_paths:
#                 words = img_path.replace('/data/', '/mask/')+'.png'
#                 words = words.split("/")
#                 mask_path = '/'.join(words[:-2]+[words[-1]])
#                 if not os.path.exists(mask_path) or not os.path.isfile(mask_path):
#                     mask_exist = False
#                 mask_paths.append(mask_path)
#             if not mask_exist:
#                 continue
#             # mask_paths = [img_path.replace('/data/', '/mask/')+'.png' for img_path in image_paths]
#             # mask_paths = obj_mask_paths[::freq]
#             image_dir.extend(image_paths)
#             mask_dir.extend(mask_paths)
#             count.append(len(image_paths))
#
#         cfg = dict()
#         cfg['data_dir'] = data_dir
#         cfg['image_dir'] = image_dir
#         cfg['mask_dir'] = mask_dir
#         cfg['obj_name'] = num_obj  # object lists for Object
#         cfg['length'] = count
#         print('valid object number: ', len(mask_dir) // num_template)
#         self.samples = cfg['image_dir']
#         self.sample_masks = cfg['mask_dir']
#
#         self.cfg = cfg
#         self.transform = transform
#         self.imsize = imsize
#         self.num_template = num_template
#
#     def __len__(self):
#         return len(self.samples)
#
#     def get_resized_mask(self, mask):
#         mask_size = self.imsize // 14
#         resized_mask = mask.resize((mask_size, mask_size), Image.BILINEAR)
#         resized_mask_numpy = np.array(resized_mask)
#         # show resized mask
#         # plt.imshow(resized_mask_numpy)
#         # plt.show()
#         resized_mask_numpy = resized_mask_numpy / 255.0
#         tensor_mask = torch.from_numpy(resized_mask_numpy.astype(np.float32))
#         tensor_mask[tensor_mask > 0.5] = 1.0
#         tensor_mask = tensor_mask.unsqueeze(0).long()  # .to(self.device)
#         if tensor_mask.sum() == 0:
#             tensor_mask = torch.ones_like(tensor_mask)
#         return tensor_mask
#
#     def __getitem__(self, index):
#         path = self.samples[index]
#         # label = self.obj_map[int(path.split('/')[-3][-6:])]
#         label = index // self.num_template
#         # if self.sample_masks:
#         mask_path = self.sample_masks[index]
#         mask = Image.open(mask_path)
#         mask = mask.convert('L')
#         ImageFile.LOAD_TRUNCATED_IMAGES = True
#
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             img = img.convert('RGB')
#
#         w, h = img.size
#
#         if (self.imsize is not None) and (min(w, h) > self.imsize):
#             img.thumbnail((self.imsize, self.imsize), Image.LANCZOS)
#             mask.thumbnail((self.imsize, self.imsize), Image.BILINEAR)
#         else:
#             new_w = math.ceil(w / 14) * 14
#             new_h = math.ceil(h / 14) * 14
#             img = img.resize((new_w, new_h), Image.LANCZOS)
#             mask = mask.resize((new_w, new_h), Image.BILINEAR)
#
#         if self.transform is not None:
#             img = self.transform(img)
#             mask = self.get_resized_mask(mask)  # transform mask to tensor
#         return img, label, mask