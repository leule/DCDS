from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import scipy.io as io
import numpy as np


class CDS_Layer(object):
    def __init__(self, toll=0.0000007, max_it=1000, batch_size=None):
        super(CDS_Layer, self).__init__()
        # self.model = model
        # self.criterion = criterion
        # self.M = outputs
        self.l = batch_size
        self.x = Variable((torch.ones(self.l, 1) / self.l).cuda)

        self.toll = toll
        self.max_it = max_it


class CDS(CDS_Layer):
    def Dist_Comp(self, outputs):


        self.M = outputs
        self.k = self.M
        #self.M.requires_grad = False
        self.y = self.M
        n = self.k.size(0)
        m = self.y.size(0)
        d = self.k.size(1)

        self.k = self.k.unsqueeze(1).expand(n, m, d)
        self.y = self.y.unsqueeze(0).expand(n, m, d)
        self.dist = torch.pow(self.k - self.y, 2).sum(2)

        self.A = 1 - self.dist
        mi = torch.min(self.A)
        te = 0.001  # a parameter to set to the smallest (negative) value in the matrix
        self.A = mi + self.A + te
        # z = torch.diag(torch.tensor([float(0)]).expand(self.l)).cuda()
        # setting the diagonal to zero

        z = -1 * torch.diag(self.A)
        z = torch.diag(z.type(torch.float))
        self.A = z + self.A
        B_size = self.A.size(0)

        self.F_rank = Variable((torch.zeros(B_size, B_size).cuda()))
        #B_size = self.B_size
        for i in range(0,B_size):

            v = torch.ones(B_size,1)
            v[i] = 0
            ind = (v!=0).nonzero()


            ind = ind[:, 0]
            M=self.A[ind][:,ind]


            ########### save the mat as matlab file to cross

            EE=torch.eig(M)
            EE2 = EE[0]

            alpha = torch.max(EE2[:, 0])
            alpha = alpha + 0.99
            ident_M = torch.eye(B_size,B_size)
            ident_M[i,i] = 0
            ident_M= ident_M*float(alpha)
            self.A_2= self.A - ident_M.cuda()
            self.A_2=self.A_2+alpha
            self.A_2 = self.A_2/torch.max(self.A_2)
            # calling Replicator dynamics
            P_g_rank= self.Replicator()
            P_g_rank = P_g_rank.view(1,-1)
            self.F_rank[i,:] = P_g_rank

        return self.F_rank

    def Replicator(self):
        self.x = torch.ones(self.l, 1) / self.l
        A_f = Variable(self.A_2.cuda(), requires_grad=True)
        x = Variable(self.x.cuda(), requires_grad=True)

        ero = 2 * self.toll + 1
        if self.max_it:
           # print(self.max_it)
            self.max_it = self.max_it
        else:
            max_it = float('inf')

        count = int(0)

        x = x * (torch.mm(A_f, x))
        x = x / torch.sum(x)

        '''
                while ero > self.toll and count < self.max_it:
            x_old = self.x
            x_old = x_old.type(torch.cuda.FloatTensor)
            x = x * (torch.mm(A_f, x))
            x = x / torch.sum(x)
            ero = torch.sqrt(torch.sum(x - x_old))
            count = count + 1

        '''
        return x

# def collect_DS(x):
#    s = len(x)
#  b = torch.zeros()
#    x_final =
