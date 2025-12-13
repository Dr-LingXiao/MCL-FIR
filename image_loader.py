import os
import csv
import json
import random
import pickle
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.autograd import Variable
import torch.nn.functional as F

Image.MAX_IMAGE_PIXELS = None


class TripletGenerator(object):
    def __init__(self, root, base_path, meta, task_id, CLIP_text, CLIP_text0, CLIP_text1):

        fnamelistfile = os.path.join(root, base_path, 'filenames_train.txt')

        self.fnamelist = []
        with open(fnamelistfile, 'r') as f:
            for fname in f:
                self.fnamelist.append(fname.strip())

        labelfile = os.path.join(root, base_path, 'label_train.txt')

        self.labels = []
        with open(labelfile, 'r') as f:
            for label in f:
                self.labels.append([int(i) for i in label.strip().split()])

        self.category = meta['ATTRIBUTES']
        self.category_num = meta['ATTRIBUTES_NUM']

        self.attributes = [i for i in range(len(self.category))]
        ################################
        self.CLIP_text = CLIP_text
        self.CLIP_text0 = CLIP_text0
        self.CLIP_text1 = CLIP_text1
        ####################################################
        self.nearest_indices = self.find_nearest_feature(self.CLIP_text)

        self.category_dict = {}
        self.task_id = task_id
        for c in self.category:
            self.category_dict[c] = []

        for i in range(len(self.labels)):
            label = self.labels[i]
            fname = self.fnamelist[label[0]]

            for j in range(len(label) // 2):
                self.category_dict[self.category[label[j * 2 + 1]]].append([fname, label[j * 2 + 2]])

    def find_nearest_feature(self, feature_list):
        num_rows = feature_list.size(0)

        nearest_feature_indices = []

        for i in range(num_rows):
            # 计算当前行与其他行的欧氏距离
            distances = torch.norm(feature_list - feature_list[i, :], dim=1)

            # 找到距离最远的特征所在的行
            nearest_index = torch.argmin(distances)
            nearest_feature_indices.append(nearest_index.item())

        return nearest_feature_indices

    def get_triplet(self, num_triplets):
        triplets = []

        for i in range(num_triplets):
            # if random.random() < 0.2:
            #     cate_r = self.nearest_indices[self.task_id]
            # else:
            cate_r = self.task_id

            cate_sub = random.randint(0, self.category_num[self.category[cate_r]] - 1)

            while True:
                a = random.randint(0, len(self.category_dict[self.category[cate_r]]) - 1)
                if self.category_dict[self.category[cate_r]][a][1] == cate_sub:
                    break

            while True:
                b = random.randint(0, len(self.category_dict[self.category[cate_r]]) - 1)
                if self.category_dict[self.category[cate_r]][b][1] != cate_sub:
                    break

            while True:
                c = random.randint(0, len(self.category_dict[self.category[cate_r]]) - 1)
                if a != c and self.category_dict[self.category[cate_r]][c][1] == cate_sub:
                    break

            triplets.append([self.category_dict[self.category[cate_r]][a],
                             self.category_dict[self.category[cate_r]][b],
                             self.category_dict[self.category[cate_r]][c],
                             cate_r,
                             self.CLIP_text[cate_r],
                             self.CLIP_text0[cate_r],
                             self.CLIP_text1[cate_r]])

        return triplets


class MetaLoader(object):
    def __init__(self, root, dataset):
        self.data = json.load(open(os.path.join(root, 'meta.json')))[dataset]

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)

        return cls.__instance


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TripletImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, base_path, num_triplets, task_id, CLIP_text, CLIP_text0, CLIP_text1, transform=None,
                 loader=default_image_loader):
        self.root = root
        self.base_path = base_path
        self.num_triplets = num_triplets
        self.task_id = task_id
        self.meta = MetaLoader(self.root, self.base_path)
        self.CLIP_text = CLIP_text[:8]
        self.CLIP_text0 = CLIP_text0[:8]
        self.CLIP_text1 = CLIP_text1[:8]
        self.triplet_generator = TripletGenerator(self.root, self.base_path, self.meta.data, self.task_id,
                                                  self.CLIP_text, self.CLIP_text0, self.CLIP_text1)
        self.triplets = self.triplet_generator.get_triplet(self.num_triplets)

       
        self.loader = loader
        self.transform = transform
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        
        self.transform1 = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomPerspective(distortion_scale=0.4, p=1),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        

    def __getitem__(self, index):
        path1 = self.triplets[index][0][0]
        path2 = self.triplets[index][1][0]
        path3 = self.triplets[index][2][0]
        c = self.triplets[index][3]
        text_c = self.triplets[index][4]
        text_c0 = self.triplets[index][5]
        text_c1 = self.triplets[index][6]

        if os.path.exists(os.path.join(self.root, self.base_path, path1)):
            img1 = self.loader(os.path.join(self.root, self.base_path, path1))
        else:
            return None

        if os.path.exists(os.path.join(self.root, self.base_path, path2)):
            img2 = self.loader(os.path.join(self.root, self.base_path, path2))
        else:
            return None

        if os.path.exists(os.path.join(self.root, self.base_path, path3)):
            img3 = self.loader(os.path.join(self.root, self.base_path, path3))
        else:
            return None

        if self.transform is not None:
            
            img11 = self.transform1(img1)
            img22 = self.transform1(img2)
            img33 = self.transform1(img3)
           
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3, c, text_c, text_c0, text_c1, img11, img22, img33
    def __len__(self):
        return len(self.triplets)

    def refresh(self):
        self.triplets = self.triplet_generator.get_triplet(self.num_triplets)


class ImageLoader(torch.utils.data.Dataset):

    def __init__(self, root, base_path, filenames_filename, split, cand_query, task_id, CLIP_text, CLIP_text0,
                 CLIP_text1, transform=None, loader=default_image_loader):
        ''' root:                   rootpath to data
            base_path:              indicate dataset, e.g., fashionAI
            filenames_filename:     file of image names
            split:                  valid or test
            cand_query:             candidate or query
        '''
        self.root = root
        self.base_path = base_path
        self.filenamelist = []
        self.task_id = task_id

        self.meta = MetaLoader(self.root, self.base_path)

        with open(os.path.join(self.root, self.base_path, filenames_filename)) as f:
            for line in f:
                self.filenamelist.append(line.rstrip('\n'))
        samples = []
        with open(os.path.join(self.root, self.base_path, cand_query + '_' + split + '.txt')) as f:
            for line in f:
                if int(line.split()[1]) in range(self.task_id + 1):
                    samples.append(
                        (line.split()[0], int(line.split()[1]), int(line.split()[2])))  # picid condition groundtruth
        np.random.shuffle(samples)
        self.samples = samples
        self.transform = transform
        self.loader = loader
        ################################

        self.CLIP_text = CLIP_text[:8]
        self.CLIP_text0 = CLIP_text0[:8]
        self.CLIP_text1 = CLIP_text1[:8]

        ####################################################

    def __getitem__(self, index):
        path, c, gdtruth = self.samples[index]

        if os.path.exists(os.path.join(self.root, self.base_path, self.filenamelist[int(path)])):
            img = self.loader(os.path.join(self.root, self.base_path, self.filenamelist[int(path)]))
            if self.transform is not None:
                img = self.transform(img)
            return img, c, self.CLIP_text0[c], self.CLIP_text1[c], gdtruth, \
            self.filenamelist[int(path)].strip().split('/')[-1].split('.')[
                0]  # imgdata, condition, groundtruth, imgid(for retrieval use)
        else:
            return None

    def __len__(self):
        return len(self.samples)

    def sample(self):
        # randomly sample two images for each category
        samples = []
        for attribute in range(len(self.meta.data['ATTRIBUTES'])):
            sub = [sample for sample in self.samples if sample[1] == attribute]
            sample = random.sample(sub, 2)
            samples.append(
                (os.path.join(self.root, self.base_path, self.filenamelist[int(sample[0][0])]), sample[0][1]))
            samples.append(
                (os.path.join(self.root, self.base_path, self.filenamelist[int(sample[1][0])]), sample[1][1]))

        return samples
