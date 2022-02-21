import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
from utils.utils import preprocess_input


class DataGenerator(data.Dataset):
    def __init__(self, txt_path, img_size):
        # 840
        self.img_size = img_size
        # './data/widerface/train/label.txt'
        self.txt_path = txt_path

        self.imgs_path, self.words = self.process_labels()

    def __len__(self):
        return len(self.imgs_path)

    def get_len(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        #-----------------------------------#
        #   打开图像，获取对应的标签
        #-----------------------------------#
        img         = Image.open(self.imgs_path[index])
        labels      = self.words[index]
        annotations = np.zeros((0, 43))

        if len(labels) == 0:
            return img, annotations

        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 43))
            #-----------------------------------#
            #   bbox 真实框的位置
            #-----------------------------------#
            annotation[0, 0] = label[1]  # x1
            annotation[0, 1] = label[2]  # y1
            annotation[0, 2] = label[1] + label[3]  # x2
            annotation[0, 3] = label[2] + label[4]  # y2

            #-----------------------------------#
            #   landmarks 人脸关键点的位置
            #-----------------------------------#
            for i in range(4, 40):
                annotation[0, i] = label[i+1]

            # 类别：1：无口罩人脸    2：戴口罩人脸
            annotation[0, 41+int(label[0])] = 1
            # 是否有关键点
            # if (annotation[0, 4]<0):
            #     annotation[0, 14] = -1
            # else:
            #     annotation[0, 14] = 1
            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)

        img, target = self.get_random_data(img, target, [self.img_size,self.img_size])

        img = np.array(np.transpose(preprocess_input(img), (2, 0, 1)), dtype=np.float32)
        return img, target

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, targes, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5):
        iw, ih  = image.size
        h, w    = input_shape
        box     = targes

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = w/h * self.rand(1-jitter,1+jitter)/self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 3.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        #------------------------------------------#
        #   色域扭曲
        #------------------------------------------#
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand()<.5 else 1/self.rand(1, sat)
        val = self.rand(1, val) if self.rand()<.5 else 1/self.rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]] = box[:, [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]]*nw/iw + dx
            box[:, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]] = box[:, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]]*nh/ih + dy
            if flip: 
                box[:, [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]] = w - box[:, [2,0,22,20,18,16,14,12,10,8,6,4,38,36,34,32,30,28,26,24]]
                box[:, [5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]]     = box[:, [23,21,19,17,15,13,11,9,7,5,39,37,35,33,31,29,27,25]]
            
            center_x = (box[:, 0] + box[:, 2])/2
            center_y = (box[:, 1] + box[:, 3])/2
        
            box = box[np.logical_and(np.logical_and(center_x>0, center_y>0), np.logical_and(center_x<w, center_y<h))]

            box[:, 0:40][box[:, 0:40]<0] = 0
            box[:, [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]][box[:, [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]]>w] = w
            box[:, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]][box[:, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]]>h] = h
            
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

        box[:,4:-1][box[:,-1]==-1]=0
        box[:, [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38]] /= w
        box[:, [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39]] /= h
        box_data = box
        return image_data, box_data
        
    def process_labels(self):
        imgs_path = []
        words = []
        f = open(self.txt_path,'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = self.txt_path.replace('label.txt','images/') + path
                imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)
        words.append(labels)
        return imgs_path, words

def detection_collate(batch):
    images  = []
    targets = []
    for img, box in batch:
        if len(box)==0:
            continue
        images.append(img)
        targets.append(box)
    images = np.array(images)
    return images, targets
