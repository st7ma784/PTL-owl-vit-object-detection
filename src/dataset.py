import json
import os
from collections import Counter
from pathlib import Path
import zipfile
from pySmartDL import SmartDL
import pytorch_lightning as pl
import time
import torch.utils.data
import torch
from PIL import Image
import torchvision
# import datasets.transforms as T
import torchvision.transforms.v2 as T2
import numpy as np
import yaml
from torch.utils.data import DataLoader
from transformers import OwlViTProcessor

TRAIN_ANNOTATIONS_FILE = "data/train.json"
TEST_ANNOTATIONS_FILE = "data/test.json"
LABELMAP_FILE = "data/labelmap.json"


def get_images_dir():
    with open("config.yaml", "r") as stream:
        data = yaml.safe_load(stream)["data"]

        return data["images_path"]

class COCODataModule(pl.LightningDataModule):

    def __init__(self, dir='.', batch_size=256,):
        super().__init__()
        self.data_dir = dir
        self.ann_dir=os.path.join(self.data_dir,"annotations")
        self.batch_size = batch_size
        self.splits={"train":[],"val":[],"test":[]}
        self.image_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.prepare_data()
        

    def train_dataloader(self, B=None):
        if B is None:
            B=self.batch_size 
        return torch.utils.data.DataLoader(self.train, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def val_dataloader(self, B=None):
        if B is None:
            B=self.batch_size
       
        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=True,num_workers=1, prefetch_factor=1, pin_memory=True,drop_last=True)
    def test_dataloader(self,B=None):
        if B is None:
            B=self.batch_size


        return torch.utils.data.DataLoader(self.val, batch_size=B, shuffle=True, num_workers=4, prefetch_factor=4, pin_memory=True,drop_last=True)
    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # # download data
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir,exist_ok=True)
        if not os.path.exists(self.ann_dir):
            os.makedirs(self.ann_dir,exist_ok=True)
        urls=[
                #'https://images.cocodataset.org/zips/test2017.zip',
                'https://images.cocodataset.org/zips/train2017.zip',
                'https://images.cocodataset.org/zips/val2017.zip',
                'https://images.cocodataset.org/annotations/annotations_trainval2017.zip'
                ]

        objs=[]
        for url in urls:
            #print("url:",url)
            name=str(url).split('/')[-1]
            location=self.data_dir # if name.startswith("annotations") else self.ann_dir
            #print("Location", location) #/Data/train2014.zip
            #time.sleep(5)
            #print('Downloading',url)
            obj=SmartDL(url,os.path.join(location,name),progress_bar=False, verify=False)
            obj.FileName=name
            if name.endswith(".zip"):
                name=name[:-4]
            if name.startswith("train"):
                self.splits['train'].append(name)
            elif name.startswith("val"):
                self.splits['val'].append(name)
            elif name.startswith("test"):
                self.splits['test'].append(name)
            if not os.path.exists(os.path.join(location,name)) and not (name.startswith("annotations") and os.path.exists(os.path.join(location,"annotations"))):
                print(os.path.join(location,name))
                objs.append(obj)
                obj.start(blocking=False,  )#There are security problems with Hostename 'images.cocodataset.org' and Certificate 'images.cocodataset.org' so we need to disable the SSL verification
        for obj in objs:
            while not obj.isFinished():
                time.sleep(5)
            if obj.isSuccessful():
                print("Downloaded: %s" % obj.get_dest())
            path = obj.get_dest()
            if path.endswith(".zip"):
                with zipfile.ZipFile(path, 'r') as zip_ref:
                    try:
                        zip_ref.extractall(self.data_dir)
                    except Exception as e:
                        print(e)
                        print("Error extracting zip" ,path)
                        continue        
                    for root, dirs, files in os.walk(zip_ref.namelist()[0]):
                        for file in files:
                            Path(os.path.join(root, file)).touch()
    def make_coco_transforms(self,image_set):


        normalize = T2.Compose([
            T2.ToTensor(),
            T2.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return T2.Compose([
                T2.Resize(800),
                T2.CenterCrop(800),
                #Note: the standard  lambda function here is not supported by pytorch lightning
            
                normalize,
        ])
    def make_mask_transforms(self,x):
        #mask is a one channel tensor which we wish to resize to 800,800

        return T2.Compose([
                T2.Resize(800),
                T2.CenterCrop(800),])

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        #print("Entered COCO datasetup")
        
    
        
        root = Path(self.data_dir)
        assert root.exists(), f'provided COCO path {root} does not exist'
        mode = 'instances'
        PATHS = {
            "train": (root  / "train2017", root / "annotations" / f'{mode}_train2017.json'),
            "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
            "test": (root /"test2017", root / "annotations" / f'image_info_test-dev2017.json'),
        }

        img_folder, ann_file = PATHS["train"]
        self.train = CocoDetection(img_folder, ann_file, transforms=self.make_coco_transforms("train"),imProcessor=self.image_processor)
        img_folder, ann_file = PATHS["val"]
        self.val = CocoDetection(img_folder, ann_file, transforms=self.make_coco_transforms("val"),imProcessor=self.image_processor)
        # img_folder, ann_file = PATHS["test"]
        # self.test=CocoDetection(img_folder, ann_file, transforms=self.make_coco_transforms("test"), return_masks=False)
        self.train_labelcounts=self.train.getlabelcounts()
        
        # scales must be in order
        scales = self.train_labelcounts
        self.scales = (np.round(np.log(scales.max() / scales) + 3, 1)).tolist()
        #creeate list of names of classes
        self.labels = [self.train.labelmap[i][0] for i in range(len(self.train.labelmap))]
    

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, imProcessor, transforms=None):
        super(CocoDetection, self).__init__(img_folder, ann_file, transforms=transforms)
        self.images_dir = img_folder
        self.image_processor = imProcessor
        self.labelmap = {i:(k,v) for i,(k, v) in enumerate(self.coco.cats.items())}
        self.labelmap[0] = (0,"background")



        with open(ann_file) as f:
            data = json.load(f)
            n_total = len(data)

        self.data = [{k: v} for k, v in data.items() if len(v)]
        print(f"Dropping {n_total - len(self.data)} examples due to no annotations")

        #calculate label counts
        #read all annotations
        #for each annotation, get the label
        #count the labels
        #make list of all categories
        categories = [cat["category_id"] for cat in self.coco.anns.values()]
        #convert to tensor
        self.train_labelcounts = torch.tensor(categories).bincount()
        #self.train_labelcounts = Counter()
        #fn =lambda ann: self.train_labelcounts.update([ann["category_id"]])
        #for ann in self.coco.anns.values():
        #    fn(ann)

    def getlabelcounts(self):
        return self.train_labelcounts

    def load_image(self, idx: int) -> Image.Image:
        url = list(self.data[idx].keys()).pop()
        path = os.path.join(self.images_dir, os.path.basename(url))
        image = Image.open(path).convert("RGB")
        return image, path

    def load_target(self, idx: int):
        annotations = list(self.data[idx].values())

        # values results in a nested list
        assert len(annotations) == 1
        annotations = annotations.pop()

        labels = []
        boxes = []
        for annotation in annotations:
            labels.append(annotation["label"])
            boxes.append(annotation["bbox"])

        return labels, boxes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        print(target)

        '''
        [{'segmentation': [[500.49, 473.53, 599.73, 419.6, 612.67, 375.37, 608.36, 354.88, 528.54, 269.66, 457.35, 201.71, 420.67, 187.69, 389.39, 192.0, 19.42, 360.27, 1.08, 389.39, 2.16, 427.15, 20.49, 473.53]],
          'area': 120057.13925, 
          'iscrowd': 0,
          'image_id': 9, 
          'bbox': [1.08, 187.69, 611.59, 285.84],
          'category_id': 51,
          'id': 1038967},
         {'segmentation': [[357.03, 69.03, 311.73, 15.1, 550.11, 4.31, 631.01, 62.56, 629.93, 88.45, 595.42, 185.53, 513.44, 230.83, 488.63, 232.99, 437.93, 190.92, 429.3, 189.84, 434.7, 148.85, 410.97, 121.89, 359.19, 74.43, 358.11, 65.8]],
           'area': 44434.751099999994,
            'iscrowd': 0,
            'image_id': 9, 
            'bbox': [311.73, 4.31, 319.28, 228.68], 
            'category_id': 51, 'id': 1039564}, 
            {'segmentation': [[249.6, 348.99, 267.67, 311.72, 291.39, 294.78, 304.94, 294.78, 326.4, 283.48, 345.6, 273.32, 368.19, 269.93, 385.13, 268.8, 388.52, 257.51, 393.04, 250.73, 407.72, 240.56, 425.79, 230.4, 441.6, 229.27, 447.25, 237.18, 447.25, 256.38, 456.28, 254.12, 475.48, 263.15, 486.78, 271.06, 495.81, 264.28, 498.07, 257.51, 500.33, 255.25, 507.11, 259.76, 513.88, 266.54, 513.88, 273.32, 513.88, 276.71, 526.31, 276.71, 526.31, 286.87, 519.53, 291.39, 519.53, 297.04, 524.05, 306.07, 525.18, 315.11, 529.69, 329.79, 529.69, 337.69, 530.82, 348.99, 536.47, 339.95, 545.51, 350.12, 555.67, 360.28, 557.93, 380.61, 561.32, 394.16, 565.84, 413.36, 522.92, 441.6, 469.84, 468.71, 455.15, 474.35, 307.2, 474.35, 316.24, 464.19, 330.92, 438.21, 325.27, 399.81, 310.59, 378.35, 301.55, 371.58, 252.99, 350.12]],
              'area': 49577.94434999999,
              'iscrowd': 0,
              'image_id': 9,
              'bbox': [249.6, 229.27, 316.24, 245.08], 
              'category_id': 56, 'id': 1058555},
         {'segmentation': [[434.48, 152.33, 433.51, 184.93, 425.44, 189.45, 376.7, 195.58, 266.94, 248.53, 179.78, 290.17, 51.62, 346.66, 16.43, 366.68, 1.9, 388.63, 0.0, 377.33, 0.0, 357.64, 0.0, 294.04, 22.56, 294.37, 56.14, 300.82, 83.58, 300.82, 109.08, 289.2, 175.26, 263.38, 216.9, 243.36, 326.34, 197.52, 387.03, 172.34, 381.54, 162.33, 380.89, 147.16, 380.89, 140.06, 370.89, 102.29, 330.86, 61.94, 318.91, 48.38, 298.57, 47.41, 287.28, 37.73, 259.51, 33.85, 240.14, 32.56, 240.14, 28.36, 247.57, 24.17, 271.46, 15.13, 282.11, 13.51, 296.96, 18.68, 336.34, 55.48, 391.55, 106.81, 432.87, 147.16], [62.46, 97.21, 130.25, 69.77, 161.25, 59.12, 183.52, 52.02, 180.94, 59.12, 170.93, 78.17, 170.28, 90.76, 157.05, 95.92, 130.25, 120.78, 119.92, 129.49, 102.17, 115.29, 64.72, 119.81, 0.0, 137.89, 0.0, 120.13, 0.0, 117.87]],
           'area': 24292.781700000007, 
           'iscrowd': 0,
            'image_id': 9,
           'bbox': [0.0, 13.51, 434.48, 375.12],
           'category_id': 51, 
           'id': 1534147}, 
        {'segmentation': [[376.2, 61.55, 391.86, 46.35, 424.57, 40.36, 441.62, 43.59, 448.07, 50.04, 451.75, 63.86, 448.07, 68.93, 439.31, 70.31, 425.49, 73.53, 412.59, 75.38, 402.92, 84.13, 387.71, 86.89, 380.8, 70.77]],
          'area': 2239.2924,
          'iscrowd': 0,
          'image_id': 9,
          'bbox': [376.2, 40.36, 75.55, 46.53],
          'category_id': 55,
           'id': 1913551},
        {'segmentation': [[473.92, 85.64, 469.58, 83.47, 465.78, 78.04, 466.87, 72.08, 472.84, 59.59, 478.26, 47.11, 496.71, 38.97, 514.62, 40.6, 521.13, 49.28, 523.85, 55.25, 520.05, 63.94, 501.06, 72.62, 482.6, 82.93]], 'area': 1658.8913000000007, 'iscrowd': 0, 'image_id': 9, 'bbox': [465.78, 38.97, 58.07, 46.67], 'category_id': 55, 'id': 1913746}, {'segmentation': [[385.7, 85.85, 407.12, 80.58, 419.31, 79.26, 426.56, 77.94, 435.45, 74.65, 442.7, 73.66, 449.95, 73.99, 456.87, 77.94, 463.46, 83.87, 467.74, 92.77, 469.39, 104.63, 469.72, 117.15, 469.39, 135.27, 468.73, 141.86, 466.09, 144.17, 449.29, 141.53, 437.1, 136.92, 430.18, 129.67]],
          'area': 3609.3030499999995,
          'iscrowd': 0,
          'image_id': 9,
          'bbox': [385.7, 73.66, 84.02, 70.51],
          'category_id': 55,
          'id': 1913856}, 
        {'segmentation': [[458.81, 24.94, 437.61, 4.99, 391.48, 2.49, 364.05, 56.1, 377.77, 73.56, 377.77, 56.1, 392.73, 41.14, 403.95, 41.14, 420.16, 39.9, 435.12, 42.39, 442.6, 46.13, 455.06, 31.17]],
          'area': 2975.276, 
          'iscrowd': 0,
          'image_id': 9,
          'bbox': [364.05, 2.49, 94.76, 71.07],
          'category_id': 55,
          'id': 1914001}]
        
        
        '''
        #search thisfor labels and boxes to return
        w, h = img.shape[-2:]
        metadata = {
            "width": w,
            "height": h,
            # "impath": path,
        }
        image = self.image_processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        labels,boxes=zip(*[(x["category_id"],x["bbox"]) for x in target])
        return image, torch.tensor(labels), torch.tensor(boxes), metadata
        # image_id = self.ids[idx]
        # target = {'image_id': image_id, 'annotations': target}
        # #print("target",target)
        # img, target = self.prepare(img, target)
        # mask=target.pop("masks")
        # if self._transforms is not None:
        #     img, target = self._transforms(img, target)
        # if self.mask_transform is not None:
        #     target["masks"]=self.mask_transform(mask)

        # summed_mask=torch.sum(target["masks"],dim=0).bool().int()
        
        
        # return img, target, self.tokenized_classnames,1-summed_mask

        # image, path = self.load_image(idx)
        # labels, boxes = self.load_target(idx)
       


def get_dataloaders(
    train_annotations_file=TRAIN_ANNOTATIONS_FILE,
    test_annotations_file=TEST_ANNOTATIONS_FILE,
):
    image_processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")

    train_dataset = OwlDataset(image_processor, train_annotations_file)
    test_dataset = OwlDataset(image_processor, test_annotations_file)

    with open(LABELMAP_FILE) as f:
        labelmap = json.load(f)

    train_labelcounts = Counter()
    for i in range(len(train_dataset)):
        train_labelcounts.update(train_dataset.load_target(i)[0])

    # scales must be in order
    scales = []
    for i in sorted(list(train_labelcounts.keys())):
        scales.append(train_labelcounts[i])

    scales = np.array(scales)
    scales = (np.round(np.log(scales.max() / scales) + 3, 1)).tolist()

    train_labelcounts = {}
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4
    )

    return train_dataloader, test_dataloader, scales, labelmap
