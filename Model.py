from torch_scatter import scatter
from torch.nn import Sequential as Seq, Linear,Conv2d,CELU,Flatten,MaxPool2d, ReLU,Softmax,GELU
from torch_geometric.nn import MessagePassing
from torch import Tensor
import torch
import torch.nn as nn
from typing import Callable, List, Optional
import math
import torch.nn.functional as F
import snntorch as snn
class MEConv(MessagePassing):
    def __init__(self, in_channels, out_channels,features,out_features):
        super(MEConv,self).__init__(aggr='mean') #  "Max" aggregation.

        self.mlp = Seq(Conv2d(in_channels, 32,3,1,padding=1,bias=False),
                       nn.BatchNorm2d(32),
                       ReLU(),
                       MaxPool2d(2,2),
                       Conv2d(32, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Conv2d(out_channels, out_channels, 3, 1, padding=1,bias=False),
                       nn.BatchNorm2d(out_channels),
                       ReLU(),
                       MaxPool2d(2, 2),
                       Flatten())

        self.mlp2=Seq(Linear(4096, 4096,bias=False))
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-6)
    def initialize(self):  # 初始化模型参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal(m.weight.data)




    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        x=self.mlp(x)




        return self.propagate(edge_index, x=x)

    def message(self,x_j,x_i):
        Cos=self.cos(x_j,x_i)

        # return self.mlp2(torch.cat((x_j, x_i), dim=-1))
        return x_j,Cos

    def aggregate(self, inputs: Tensor, index: Tensor,
                  ptr: Optional[Tensor] = None,
                  dim_size: Optional[int] = None) -> Tensor:
        r"""Aggregates messages from neighbors as
        :math:`\square_{j \in \mathcal{N}(i)}`.

        Takes in the output of message computation as first argument and any
        argument which was initially passed to :meth:`propagate`.

        By default, this function will delegate its call to scatter functions
        that support "add", "mean" and "max" operations as specified in
        :meth:`__init__` by the :obj:`aggr` argument.
        """


        return scatter(inputs[0], index, dim=self.node_dim, dim_size=dim_size,
                       reduce=self.aggr),scatter(inputs[1], index, dim=0, dim_size=dim_size,
                       reduce=self.aggr)
    def update(self, inputs: Tensor,x) -> Tensor:
        return inputs[0],x,inputs[1]


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
def attention_loss_pre(attention_class):

    attention_bool=torch.zeros(24,dtype=torch.bool)
    attention_bool[attention_class]=True

    return attention_bool,~attention_bool
def attention_loss_pre_sub(attention_class):

    attention_bool=torch.zeros(6,dtype=torch.bool)
    attention_bool[attention_class]=True

    return attention_bool,~attention_bool
class GCN3(torch.nn.Module):
    def __init__(self,attention_class):
        super().__init__()
        self.conv1 = MEConv(3, 64,1024,1024)



        self.mlp=Seq(Linear(1024,1024),nn.ReLU(),Linear(1024,1))
        self.mlp2 = Seq(Linear(1024, 1024), nn.ReLU(), Linear(1024, 1))

        self.softmax = nn.Softmax(dim=-1)
        self.pdis=nn.PairwiseDistance(p=2)
        self.conv1.initialize()
        self.attention_class=attention_class
        self.attention_class_index, self.rest_class_index=attention_loss_pre(self.attention_class)
        self.attention_class_index, self.rest_class_index=self.attention_class_index.to(device), self.rest_class_index.to(device)
    def forward(self, support=None,query=None,support_set_all1=None,test=False,feature=False,support_index1=None,suport_target=None):
        if test is False and feature is False:
            support_x1, support_edge_index1 = support.x, support.edge_index

            query_x, query_edge_index = query.x, query.edge_index

            s_feature, s_mss_feature, s_cos1 = self.conv1(support_x1, support_edge_index1)
            # s_cat1 = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat1 = s_mss_feature
            s_cat1 = scatter(s_cat1, support.batch, dim=0, reduce='mean')




            q_feature1, p_mss_feature1,q_cos1 = self.conv1(query_x, query_edge_index)
            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            p_cat1 = scatter(p_cat1, query.batch, dim=0, reduce='mean')


            pdis1 = []


            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((s_cat1 - p_cat1[i, :]), 2).sum(1)))


                pdis1.append(a)




            pd1 = torch.stack(pdis1, dim=0)


            s_cos1=scatter(s_cos1, support.batch, dim=0, reduce='mean')



            cos_loss1=torch.pow(s_cos1 - 1, 2)

            cos_loss=torch.sum(cos_loss1)
            dis = []






            for i in range(pd1.size(0)):


                a = pd1[i,:]

                # a = center@q_x[i,:]
                a = scatter(a, support.y, dim=0, reduce='mean')


                dis.append(a)

            # x=self.classifier(torch.stack(dis,dim=0))
            output=torch.stack(dis, dim=0)



            label = torch.nn.functional.one_hot(query.y, num_classes=24)
            atten_loss= F.cross_entropy(output[self.attention_class_index,:], label[self.attention_class_index,:].float(), reduction='mean')
            rest_loss =F.cross_entropy(output[self.rest_class_index,:], label[self.rest_class_index,:].float(), reduction='mean')
            loss = atten_loss+rest_loss
            loss = loss + cos_loss

            values, indices = output.max(1)
            acc = torch.sum((indices.squeeze() == query.y).float())

            return loss,acc,cos_loss,atten_loss,rest_loss
        elif test is True and feature is False:
            query_x, query_edge_index = query.x, query.edge_index
            q_feature1, p_mss_feature1, _ = self.conv1(query_x, query_edge_index)

            # p_cat1 = torch.cat([q_feature1, p_mss_feature1], dim=-1)
            p_cat1 = p_mss_feature1
            support_set_all1=support_set_all1.to(device)


            pdis = []

            dd = []
            for i in range(p_cat1.size(0)):
                # a = center @ q_x[i, :]
                a = torch.exp(-(torch.pow((support_set_all1 - p_cat1[i, :]), 2).sum(1)))




                a = scatter(a, support_index1, dim=0, reduce='mean')

                pdis.append(a)
            pd = torch.mean(torch.stack(pdis, dim=0),dim=0)


            a = (pd)



            # a = center@q_x[i,:]
            a = scatter(a, suport_target.squeeze(), dim=0, reduce='mean')




            return a


        elif feature is True:

            support_x, support_edge_index = support.x, support.edge_index

            s_feature, s_mss_feature, s_cos = self.conv1(support_x, support_edge_index)
            # s_cat = torch.cat([s_feature, s_mss_feature], dim=-1)
            s_cat = s_mss_feature
            return s_cat
