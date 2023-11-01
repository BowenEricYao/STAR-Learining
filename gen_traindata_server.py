import os

import torchvision
import torch

from PIL import Image


def class_split(label_file):
    labels = torch.load(label_file)
    start = 0
    ind=[]

    for i in labels:


        ind.append(torch.range(start,start+len(i)-1,dtype=torch.long))
        start=start+len(i)

    return ind
def gen_data(idx):
    if not os.path.exists('D:/SKDD/raw/train_data2.pt'):
        dir_list=os.listdir("D:/skindiseasedata/train"+idx)
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),torchvision.transforms.ToTensor()])
        labels=[]
        data=[]
        for i, dir in enumerate(dir_list):
            path=os.path.join("D:/skindiseasedata/train"+idx, dir)
            if os.path.isdir(path):
                path_temp=path
                sub_dir_list=os.listdir(path_temp)
                disease=[]
                disease_label=[]
                for j,sub_dir in enumerate(sub_dir_list):
                    sub_path=os.path.join(path,sub_dir)
                    image_list=os.listdir(sub_path)
                    patient=[]
                    patient_label = []

                    label=torch.zeros(2)
                    for img_name in image_list:
                        img=Image.open(os.path.join(sub_path,img_name))
                        imgs=transform(img)

                        label[0]=i
                        label[1]=j
                        patient_label.append(label)
                        patient.append(imgs)
                        print('Data has been done:[{0}/{1}]'.format(i,j))
                    disease.append(torch.stack(patient))
                    disease_label.append(patient_label)
                data.append(disease)
                labels.append(disease_label)
        torch.save(data,'D:/SKDD/raw/train_data2.pt')
        torch.save(labels,'D:/SKDD/raw/train_labels2.pt')

    if not os.path.exists('D:/SKDD/raw/test_data2.pt'):
        dir_list=os.listdir("D:\\skindiseasedata\\test2")
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),torchvision.transforms.ToTensor()])
        labels=[]
        data=[]
        for i, dir in enumerate(dir_list):
            path=os.path.join("D:\\skindiseasedata\\test2", dir)
            if os.path.isdir(path):
                path_temp=path
                sub_dir_list=os.listdir(path_temp)
                disease=[]
                disease_label=[]
                for j,sub_dir in enumerate(sub_dir_list):
                    sub_path=os.path.join(path,sub_dir)
                    image_list=os.listdir(sub_path)
                    patient=[]
                    patient_label = []

                    label=torch.zeros(2)
                    for img_name in image_list:
                        img=Image.open(os.path.join(sub_path,img_name))
                        imgs=transform(img)

                        label=dir+"/"+sub_dir
                        print(label)
                        patient_label.append(label)
                        patient.append(imgs)
                        print('Data has been done:[{0}/{1}]'.format(i,j))
                    disease.append(torch.stack(patient))
                    disease_label.append(patient_label)
                data.append(disease)
                labels.append(disease_label)
        torch.save(data,'D:/SKDD/raw/test_data2.pt')
        torch.save(labels,'D:/SKDD/raw/test_labels2.pt')



        train_class_index = class_split('D:/SKDD/raw/train_labels2.pt')

        test_class_index = class_split('D:/SKDD/raw/test_labels2.pt')


        torch.save(train_class_index, 'D:/SKDD/raw/train_class_index'+idx+'2.pt')
        torch.save(test_class_index, 'D:/SKDD/raw/test_class_index2.pt')
    else:
        train_class_index = class_split('D:/SKDD/raw/train_labels2.pt')

        test_class_index = class_split('D:/SKDD/raw/test_labels2.pt')

        torch.save(train_class_index, 'D:/SKDD/raw/train_class_index'+idx+'2.pt')
        torch.save(test_class_index, 'D:/SKDD/raw/test_class_index2.pt')
def gen_data_sub(idx):
    if not os.path.exists('D:/SKDD/raw/train_data3_sub.pt'):
        dir_list=os.listdir("D:/skindiseasedata/trainsub/train"+idx+"sub")
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),torchvision.transforms.ToTensor()])
        labels=[]
        data=[]
        for i, dir in enumerate(dir_list):
            path=os.path.join("D:/skindiseasedata/trainsub/train"+idx+"sub", dir)
            if os.path.isdir(path):
                path_temp=path
                sub_dir_list=os.listdir(path_temp)
                disease=[]
                disease_label=[]
                for j,sub_dir in enumerate(sub_dir_list):
                    sub_path=os.path.join(path,sub_dir)
                    image_list=os.listdir(sub_path)
                    patient=[]
                    patient_label = []

                    label=torch.zeros(2)
                    for img_name in image_list:
                        img=Image.open(os.path.join(sub_path,img_name))
                        imgs=transform(img)

                        label[0]=i
                        label[1]=j
                        patient_label.append(label)
                        patient.append(imgs)
                        print('Data has been done:[{0}/{1}]'.format(i,j))
                    disease.append(torch.stack(patient))
                    disease_label.append(patient_label)
                data.append(disease)
                labels.append(disease_label)
        torch.save(data,'D:/SKDD/raw/train_data3_sub.pt')
        torch.save(labels,'D:/SKDD/raw/train_labels3_sub.pt')

    if not os.path.exists('D:/SKDD/raw/test_data3_sub.pt'):
        dir_list=os.listdir("D:\\skindiseasedata\\test3_sub")
        transform=torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),torchvision.transforms.ToTensor()])
        labels=[]
        data=[]
        for i, dir in enumerate(dir_list):
            path=os.path.join("D:\\skindiseasedata\\test3_sub", dir)
            if os.path.isdir(path):
                path_temp=path
                sub_dir_list=os.listdir(path_temp)
                disease=[]
                disease_label=[]
                for j,sub_dir in enumerate(sub_dir_list):
                    sub_path=os.path.join(path,sub_dir)
                    image_list=os.listdir(sub_path)
                    patient=[]
                    patient_label = []

                    label=torch.zeros(2)
                    for img_name in image_list:
                        img=Image.open(os.path.join(sub_path,img_name))
                        imgs=transform(img)

                        label=dir+"/"+sub_dir
                        print(label)
                        patient_label.append(label)
                        patient.append(imgs)
                        print('Data has been done:[{0}/{1}]'.format(i,j))
                    disease.append(torch.stack(patient))
                    disease_label.append(patient_label)
                data.append(disease)
                labels.append(disease_label)
        torch.save(data,'D:/SKDD/raw/test_data3_sub.pt')
        torch.save(labels,'D:/SKDD/raw/test_labels3_sub.pt')



        train_class_index = class_split('D:/SKDD/raw/train_labels3_sub.pt')

        test_class_index = class_split('D:/SKDD/raw/test_labels3_sub.pt')


        torch.save(train_class_index, 'D:/SKDD/raw/train_class_index'+idx+'_sub.pt')
        torch.save(test_class_index, 'D:/SKDD/raw/test_class_index_sub.pt')
    else:
        train_class_index = class_split('D:/SKDD/raw/train_labels3_sub.pt')

        test_class_index = class_split('D:/SKDD/raw/test_labels3_sub.pt')

        torch.save(train_class_index, 'D:/SKDD/raw/train_class_index'+idx+'_sub.pt')
        torch.save(test_class_index, 'D:/SKDD/raw/test_class_index_sub.pt')