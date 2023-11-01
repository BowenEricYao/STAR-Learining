
import torch
import torch.nn.functional as F

from Model import GCN,GCN2,GCN1,GCN3,GCN3_metrics


import torch.utils.data

import torch

import torch.optim as optim

import argparse
import scipy.io as sio

import copy

import gen_traindata_server
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


temprature_inc_rate=10000
import os
import cv2
import torchvision
import torch
import scipy.io as sio
from PIL import Image
import numpy as np
from torch_geometric.data import Data, InMemoryDataset,DataLoader
from torch_geometric.data import Batch, Dataset



from tqdm import tqdm
from typing import Any, Callable, List, Optional, Tuple, Union, Dict, Iterable
import torch.utils.data
from torch import Tensor
from torch_geometric.data import Data
from torch.utils.data.distributed import DistributedSampler
from collections.abc import Mapping, Sequence
IndexType = Union[slice, Tensor, np.ndarray, Sequence]
from torch.utils.data import _utils,Sampler,SequentialSampler,BatchSampler,SubsetRandomSampler



def train(dataset, model,  optimizer, epoch):
    model.train()

    support_set1,query_set,s_id,q_id = trainloader(dataset)


    support1,query = support_set1.to(device),query_set.to(device)
    optimizer.zero_grad()
    loss,acc,closs,loss1,loss2= model(support=support1, query=query,test=False)

    loss.backward()
    optimizer.step()
    print('Train Epoch: {} \tLoss: {}\tAcc: {}\tCloss: {},Attention_loss:{},Set_loss:{}'.format(epoch,loss,acc,closs,loss1,loss2))
    return acc,loss1,loss2,closs

def test(dataset, model,support_all1,support_index1,t,temprature_P,attention_class):
    model.eval()
    attention_class=attention_class

    with torch.no_grad():

        tloss=[]
        tacc=[]
        tacc5=[]
        TP=0
        FP=0
        with tqdm(total=len(dataset) ) as pbar:
            for id,data in enumerate(dataset):

                query =  data.to(device)

                output = model( query=query, support_set_all1=support_all1,test=True,support_index1=support_index1,suport_target=t,temprature_P=temprature_P)
                y = torch.nn.functional.one_hot(query.y, num_classes=24)

            #
            #     # for data in data_list:
            #     #     data = data.to(rank)
            #     #     optimizer.zero_grad()
            #     #     output = model(data)
                loss = F.cross_entropy(output.unsqueeze(0), y.float())
                values, indices = output.max(0)
                _,indices5=output.topk(2)

            # values, indices = preds.max(1)
            #
                acc = (indices.squeeze() == query.y).float()
                acck=[]
                for i in indices5:
                    acck.append((i.squeeze() == query.y).float())
                acc5=torch.sum(torch.stack(acck))
                if indices.squeeze() == attention_class:
                    if indices.squeeze() == query.y:
                        TP+=1
                    else:
                        FP+=1

                tloss.append(loss)
                tacc.append(acc)
                tacc5.append(acc5)
                iter_out = "Acc:{},Acc5:{}".format(acc,acc5)
                pbar.set_description(iter_out)
                pbar.update(1)



    print('Test set: Average loss: {:.4f}, Accuracy: {:.2f},Accuracy5: {:.2f}\nTP:{},FP:{}'.format(
        torch.mean(torch.stack(tloss)),  torch.mean(torch.stack(tacc)), torch.mean(torch.stack(tacc5)),TP,FP))
    return TP,FP
def testout(dataset, model,support_all1,support_index1,t,attention_class):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(device)
    attention_class = attention_class

    with torch.no_grad():
        pre=[]
        gt=[]
        gt2=[]
        outcome=[]
        label=[]
        tacc2=[]
        tacc3 = []
        tacc4 = []
        tacc5 = []
        with tqdm(total=len(dataset) ) as pbar:
            for id,data in enumerate(dataset):

                query =  data.to(device)

                output = model( query=query, support_set_all1=support_all1,test=True,support_index1=support_index1,suport_target=t)
                y = torch.nn.functional.one_hot(query.y, num_classes=24)

            #
            #     # for data in data_list:
            #     #     data = data.to(rank)
            #     #     optimizer.zero_grad()
            #     #     output = model(data)
                values, indices = output.max(0)
                acc = (indices.squeeze() == query.y).float()
                _, indices2 = output.topk(2)
                _, indices3 = output.topk(3)
                _, indices4 = output.topk(5)
                _, indices5 = output.topk(10)
                acck = []
                for i in indices2:
                    acck.append((i.squeeze() == query.y).float())
                acc2 = torch.sum(torch.stack(acck))
                tacc2.append(acc2)
                acck = []
                for i in indices3:
                    acck.append((i.squeeze() == query.y).float())
                acc3 = torch.sum(torch.stack(acck))
                tacc3.append(acc3)
                acck = []
                for i in indices4:
                    acck.append((i.squeeze() == query.y).float())
                acc4 = torch.sum(torch.stack(acck))
                tacc4.append(acc4)
                acck = []
                for i in indices5:
                    acck.append((i.squeeze() == query.y).float())
                acc5 = torch.sum(torch.stack(acck))
                tacc5.append(acc5)


            # values, indices = preds.max(1)
            #
                acc = (indices.squeeze() == query.y).float()
                if acc==1.0:
                    pre.append(query.idx)
                if indices.squeeze()==attention_class:
                    gt.append(query.idx)
                    outcome.append(1)
                if indices.squeeze()!=attention_class:
                    gt.append(query.idx)
                    outcome.append(0)
                if query.y==attention_class:
                    gt2.append(indices.squeeze().item())
                    label.append(1)
                if query.y !=attention_class:
                    gt2.append(indices.squeeze().item())
                    label.append(0)
            print(torch.mean(torch.stack(tacc2)),torch.mean(torch.stack(tacc3)),torch.mean(torch.stack(tacc4)),torch.mean(torch.stack(tacc5)))


    return pre,gt,gt2,outcome,label

from torch_scatter import scatter
def featureout(dataset, model,train_index):
    model.eval()
    correct = 0
    ddp_loss = torch.zeros(3).to(device)
    with torch.no_grad():
        support_all1=[]
        support_all2 = []
        t=[]
        support_index1=[]
        support_index2 = []
        pp=[]
        pp2=[]

        mask = torch.eye(23,23, dtype=torch.bool)
        count1 = torch.tensor([0]).to(device)

        for i_id, list in enumerate(train_index):



            for j_id,id in enumerate(list):
                support=dataset[id]
                feature=model(support=support.to(device),feature=True)
                support_index1.append(torch.repeat_interleave(count1,feature.size(0)))
                support_all1.append(feature.detach())
                t.append(support.y)
                count1+=1




        support_all1= torch.cat(support_all1,dim=0)

        t=torch.stack(t)
        support_index1=torch.cat(support_index1,dim=0)



    return support_all1,support_index1,t

attention_class=23
idx=str(attention_class)+"_"+str(46)



def Train():
    np.random.seed(1234)
    gen_traindata_server.gen_data(idx)
    traindata = TrainDataset(root='D:/SKDD')
    traindata2 = TrainDataset(root='D:/SKDD')

    testdata = TestDataset(root='D:/SKDD')
    # testdata2 = TestDataset2(root='D:/SKDD')


    train_index = torch.load('D:/SKDD/raw/train_class_index'+idx+'2.pt')
    train_index2 = torch.load('D:/SKDD/raw/train_class_index'+idx+'2.pt')
    save_name="D:/SKDD/"+idx+".pt"
    best_model_path="D:/SKDD/server_weights_e41/"+str(attention_class)+"/best_model/best_model_"+idx+".pth"
    supoort_feature_path = "D:/SKDD/server_weights_e41/"+str(attention_class)+"/support_feature/support_feature_" + idx + ".pth"
    supoort_index_path = "D:/SKDD/server_weights_e41/"+str(attention_class)+"/support_index/support_index_" + idx + ".pth"
    supoort_target_path = "D:/SKDD/server_weights_e41/"+str(attention_class)+"/support_target/support_target_" + idx + ".pth"
    tpr=0.0
    fpr = 0.20

    model = GCN3(attention_class).to(device)


    tp=torch.zeros(1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    val_loss=100
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14,200])
    A_L=[]
    S_L=[]
    L=[]
    TP=[]
    FP=[]
    C_L=[]


    for epoch in range(100):


        acc,Attentionloss,loss2,closs=train(traindata, model, optimizer, epoch)
        scheduler.step(Attentionloss)
        support_all1, support_index1, t = featureout(traindata, model, train_index)
        tP, fP = test(testdata, model, support_all1, support_index1, t, temprature_P,attention_class)
        L.append(Attentionloss+loss2)
        A_L.append(Attentionloss)
        S_L.append(loss2)
        C_L.append(closs)
        TP.append(tP)
        FP.append(fP)
        if val_loss > Attentionloss and epoch>15:
            val_loss=Attentionloss
            torch.save(model.state_dict(), best_model_path)
            print("Epoch {}:tpr is {} fpr is {} Save weights in {}".format(epoch,tpr,fpr, best_model_path))
    model.load_state_dict(copy.deepcopy(torch.load(best_model_path, device)))
    support_all1,support_index1,t = featureout(traindata2, model, train_index2)
    torch.save(support_all1,supoort_feature_path)
    torch.save(support_index1, supoort_index_path)
    torch.save(t, supoort_target_path)
    out,gt,gt2,outcome, label = testout(testdata, model,support_all1,support_index1,t,attention_class)
    print(out,len(out),"\n",gt,len(gt),"\n",gt2,len(gt2))
    winsound.Beep(freq, delay)


class TrainDataset(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train_data2.pt', 'train_labels2.pt']

    @property
    def processed_file_names(self):
        return ['server\\'+idx+'\\train_predata2.pt']

    def download(self):
        return
        # Download to `self.raw_dir`.


    def process(self):
        # Read data into huge `Data` list.
        data_all = torch.load(self.raw_paths[0])
        target_all = torch.load(self.raw_paths[1])
        data_list=[]
        for i, disease in enumerate(tqdm(data_all)):
            for j, patient in enumerate(tqdm(disease)):
                start = torch.repeat_interleave(torch.arange(len(patient)), patient.size(0))

                end = torch.arange(len(patient)).repeat(patient.size(0))
                # start = torch.repeat_interleave(torch.tensor(0), patient.size(0))
                #
                # end = torch.arange(len(patient))
                edge_index=torch.stack([start,end],dim=0).long()
                target=torch.tensor(i,dtype=torch.long)
                idx = torch.tensor(j, dtype=torch.long)




                data=Data(x=patient,edge_index=edge_index,y=target,idx=j)



                data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
    def indices(self) -> Sequence:

        return range(self.len()) if self._indices is None else self._indices
    def __len__(self) -> int:
        r"""The number of examples in the dataset."""

        return len(self.indices())

    def __getitem__(
            self,
            idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)
    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0

class TestDataset(InMemoryDataset):
    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['test_data2.pt', 'test_labels2.pt']

    @property
    def processed_file_names(self):
        return ['test_predata2.pt']

    def download(self):
        return
        # Download to `self.raw_dir`.


    def process(self):
        # Read data into huge `Data` list.
        data_all = torch.load(self.raw_paths[0])
        target_all = torch.load(self.raw_paths[1])
        data_list=[]
        for i, disease in enumerate(tqdm(data_all)):
            for j, patient in enumerate(tqdm(disease)):
                start = torch.repeat_interleave(torch.arange(len(patient)), patient.size(0))

                end = torch.arange(len(patient)).repeat(patient.size(0))
                # start = torch.repeat_interleave(torch.tensor(0), patient.size(0))
                #
                # end = torch.arange(len(patient))

                edge_index=torch.stack([start,end],dim=0).long()
                target=torch.tensor(i,dtype=torch.long)
                idx = target_all[i][j][0]




                data=Data(x=patient,edge_index=edge_index,y=target,idx=idx)



                data_list.append(data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]


        data, slices = self.collate(data_list)

        torch.save((data, slices), self.processed_paths[0])
    def indices(self) -> Sequence:

        return range(self.len()) if self._indices is None else self._indices
    def __len__(self) -> int:
        r"""The number of examples in the dataset."""

        return len(self.indices())

    def __getitem__(
            self,
            idx: Union[int, np.integer, IndexType],
    ) -> Union['Dataset', Data]:
        r"""In case :obj:`idx` is of type integer, will return the data object
        at index :obj:`idx` (and transforms it in case :obj:`transform` is
        present).
        In case :obj:`idx` is a slicing object, *e.g.*, :obj:`[2:5]`, a list, a
        tuple, or a :obj:`torch.Tensor` or :obj:`np.ndarray` of type long or
        bool, will return a subset of the dataset at the specified indices."""
        if (isinstance(idx, (int, np.integer))
                or (isinstance(idx, Tensor) and idx.dim() == 0)
                or (isinstance(idx, np.ndarray) and np.isscalar(idx))):

            data = self.get(self.indices()[idx])
            data = data if self.transform is None else self.transform(data)
            return data

        else:
            return self.index_select(idx)
    def len(self) -> int:
        if self.slices is None:
            return 1
        for _, value in nested_iter(self.slices):
            return len(value) - 1
        return 0
def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):

        for key, value in node.items():

            for inner_key, inner_value in nested_iter(value):

                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node


def nested_iter(node: Union[Mapping, Sequence]) -> Iterable:
    if isinstance(node, Mapping):

        for key, value in node.items():

            for inner_key, inner_value in nested_iter(value):

                yield inner_key, inner_value
    elif isinstance(node, Sequence):
        for i, inner_value in enumerate(node):
            yield i, inner_value
    else:
        yield None, node
def trainloader2(data,num_ways=2,num_shot=5,num_query=1,drop_last=True,shuffle=False):
    sampler1=np.zeros(num_ways*(num_shot),dtype=np.int64)
    sampler2 = np.zeros(num_ways * (num_query), dtype=np.int64)
    sampler1_t=[]
    data_list=[]
    train_index=torch.load('D:/SKDD/raw/train_class_index_all2class.pt')

    for i,ind in enumerate(train_index):


        choice=np.random.choice(ind, num_shot+num_query,replace=False)
        sampler1[i * (num_shot):(i + 1) * (num_shot)]=choice[:-1]
        sampler2[i * (num_query):(i + 1) * (num_query)] =choice[-1]
        # sampler1[0:(num_shot+num_query)]=np.random.choice(ind,num_shot+num_query)
        # data_list.append(Batch.from_data_list(data.index_select(sampler1)))
    #     batch_sampler += BatchSampler(sampler, batch_size=num_ways, drop_last=drop_last)
    # for indices in batch_sampler:



    return Batch.from_data_list(data.index_select(sampler1)),Batch.from_data_list(data.index_select(sampler2)),sampler1,sampler2

def trainloader(data,num_ways=24,num_shot=5,num_query=1,drop_last=True,shuffle=False):
    # sampler1=np.zeros(num_ways*(num_shot),dtype=np.int64)
    sampler2 = np.zeros(num_ways * (num_query), dtype=np.int64)
    sampler1_t=[]
    data_list=[]
    train_index=torch.load('D:/SKDD/raw/train_class_index'+idx+'2.pt')

    for i,ind in enumerate(train_index):
        choice=np.random.choice(ind, int((len(ind)/2)+1),replace=False)
        for j in range(len(choice[:-1])):

            sampler1_t.append(choice[j])
        sampler2[i * (num_query):(i + 1) * (num_query)] =choice[-1]
        # sampler1[0:(num_shot+num_query)]=np.random.choice(ind,num_shot+num_query)
        # data_list.append(Batch.from_data_list(data.index_select(sampler1)))
    #     batch_sampler += BatchSampler(sampler, batch_size=num_ways, drop_last=drop_last)
    # for indices in batch_sampler:
    sampler1=np.asarray(sampler1_t)


    return Batch.from_data_list(data.index_select(sampler1)),Batch.from_data_list(data.index_select(sampler2)),sampler1,sampler2
def testloader(data,num_ways=24,num_shot=5,num_query=1,drop_last=True,shuffle=False):
    sampler1 = np.zeros(num_ways * (num_shot), dtype=np.int64)
    sampler2 = torch.nn
    data_list = []
    train_index = torch.load('D:/SKDD/raw/train_index2.pt')
    test_index = torch.load('D:/SKDD/raw/test_index2.pt')
    for i, ind in enumerate(train_index):
        choice = np.random.choice(ind, num_shot,replace=False,)
        sampler1[i * (num_shot):(i + 1) * (num_shot)] = choice[:]
        # sampler1[0:(num_shot+num_query)]=np.random.choice(ind,num_shot+num_query)
        # data_list.append(Batch.from_data_list(data.index_select(sampler1)))
    #     batch_sampler += BatchSampler(sampler, batch_size=num_ways, drop_last=drop_last)
    # for indices in batch_sampler:
    sampler2 = np.random.choice(test_index, (num_query))
    return Batch.from_data_list(data.index_select(sampler1)), Batch.from_data_list(data.index_select(sampler2))

if __name__ == '__main__':


    torch.manual_seed(args.seed)
    Train()

