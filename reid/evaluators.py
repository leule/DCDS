from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import DataLoader

from .evaluation_metrics import cmc, mean_ap
from .feature_extraction import extract_cnn_feature
from .utils.meters import AverageMeter
from .utils import to_numpy
from .utils.data.preprocessor import KeyValuePreprocessor
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

cudnn.enabled = True
cudnn.benchmark = True


def compute_CDS(model, probe_feature, gallery_feature, i, rerank_topk):
    
    count = 2048 / (len(model))
    outputs = []

    outputs = []
    for j in range(len(model)):
        # bb = bb[:, j * count:(j + 1) * count, 12, 4]
        p_g_score, ds = model[j](
            Variable(gallery_feature[:, j * count:(j + 1) * count, :, :].contiguous().cuda(), volatile=True),
            Variable(gallery_feature[:, j * count:(j + 1) * count, :, :].contiguous().cuda(), volatile=True),
            p2g=False, g2g=True)

        #####################################################################################################

        A = p_g_score.data[:, :, 1].clone()
        # A=ds.data
        B_size = A.size(0)

        ###########################################

        # A = Variable(A)
        # A = F.softmax(A)
        # A = A.data

        z = -1 * (torch.diag(A))
        kk = torch.diag(z)
        A = kk + A

        F_rank = torch.zeros(A.size(0), A.size(0)).cuda()

        v = torch.ones(A.size(0), 1)
        v[0] = 0
        ind = (v != 0).nonzero()

        # ind = ind[:, 0].contiguous()
        # ind = ind.view(ind.size(0))
        # neMa= A.data.contiguous()
        # neMa = neMa.cuda()
        # M = neMa[ind][:, ind]
        ind = ind[:, 0]
        ind = ind.cuda()
        M = A[ind][:, ind]

        ########### save the mat as matlab file to cross
        M2 = M
        EE = torch.eig(M2)
        EE2 = EE[0]

        alpha = torch.max(EE2[:, 0])
        alpha = alpha + 0.09
        ident_M = torch.eye(A.size(0), A.size(0))
        ident_M[0, 0] = 0
        ident_M = ident_M * float(alpha)
        A_2 = A.cpu() - ident_M
        A_2 = A_2 + alpha

        A_2 = A_2 / torch.max(
            A_2)  # dividing the matrix with the maximum value, to normalize it # This normalization has a big impact on the MAP

        alpha = 0.95

        # calling Replicator dynamics

        A_1 = A_2[0:10][:, 0:10]
        # calling Replicator dynamics
        ou = Replicator(A_1, max_it=100)
        ou2 = ou.numpy()
        ouso = np.argsort(ou2, axis=0)
        indd = ouso[8]
        indd2 = ouso[7]
        indd3 = ouso[6]

        A_2[indd, indd] = 0.9
        A_2[indd2, indd2] = 0.85

        A_2[indd3, indd3] = 0.7

        A_2[[0], indd] = A_2[[0], indd] + 0.07
        A_2[indd, [0]] = A_2[indd, [0]] + 0.07

        outt = Replicator(A_2, max_it=1000)

        # A=F.softmax(Variable(A),dim=1)

        '''

        one_diag = Variable(torch.eye(A.size(0)), requires_grad=False).cuda()  # identity matrix
        A = Variable(A,requires_grad=False)
        # Row Normalization
        inf_diag = torch.diag(torch.Tensor([-float('Inf')]).expand(rerank_topk)).cuda() + A.squeeze().data
        A = F.softmax(Variable(inf_diag), dim=1)

        A = (1 - alpha) * torch.inverse(one_diag - alpha * A)
        A = A.transpose(0, 1)  # transpos
        outt=Variable(outt).cuda()
        outt=outt.view(A.size(0))
        outt=torch.matmul(outt,A)
        #outt[0]=outt[0]+0.3



        '''

        outt = outt[0:outt.size(0)]
        outt = Variable(outt)
        outt2 = Variable((-0.3 * outt.data))
        outt3 = torch.cat((outt2, outt), -1)
        outt = outt3.view(outt.size(0), 2)
        outputs.append(outt)

        # outt = outt.view(outt.size(0) * outt.size(0), 1)
        # outt2 = Variable((-1 * outt.data))
        # outt3 = torch.cat((outt2, outt), -1)
        # outt = outt3.view(outt.size(0), 2)

        # outputs.append(outt)

        #####################################################
        '''            F_rank = torch.mm(F_rank, p_g_score)
        Fr = torch.mm(F_rank, K_nn2)  # each column corresponds to the id, and the row corresponds to the gallery image

        #return Fr
        # return F_list
       '''
    # outputs = torch.cat(outputs, 0)
    outputs = torch.cat(outputs, 0).view(len(model), -1, 2)
    outputs = torch.mean(outputs, 0)

    return outputs


def Replicator(A_2, max_it):
    l = A_2.size(0)
    x = torch.ones(l, 1) / l
    A_f = A_2
    x = x
    toll = 0.00000001
    ero = 2 * toll + 1

    if max_it:
        # print(self.max_it)
        max_it = max_it
    else:
        max_it = float('inf')

    count = int(0)

    # x = x * (torch.mm(A_f, x))
    # x = x / torch.sum(x)

    while ero > toll and count < max_it:
        x_old = x.cpu()
        x_old = x_old.type(torch.FloatTensor)
        x = (x * (torch.matmul(A_f, x))).contiguous()

        xx = torch.norm(x, p=2, dim=0)  # find the l2 norm, sum of the square of each element
        x = x.div(xx.expand_as(x))  # divide each element of x with the norm

        ero = torch.norm(x - x_old)
        ero = float(ero)
        count = count + 1

    N_x = x.cpu().numpy()
    si = np.argsort(N_x, axis=0)
    return x


######################################################################################################################

def extract_embeddings(model, features, query=None, topk_gallery=None, rerank_topk=0, print_freq=500):
    for i in model:
        i.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()
    pairwise_score = Variable(torch.zeros(len(query), rerank_topk, 2).cuda())
    probe_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    for i in range(len(query)):
        gallery_feature = torch.cat([features[f].unsqueeze(0) for f, _, _ in topk_gallery[i]], 0) # feature of top-k NN
        pairwise_score[i, :, :] = compute_CDS(model, probe_feature, gallery_feature, i, rerank_topk) #i indicate the query index

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Embedding: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'.format(
                i + 1, len(query),
                batch_time.val, batch_time.avg,
                data_time.val, data_time.avg))

    return pairwise_score.view(-1, 2)


def extract_features(model, data_loader, print_freq=1, metric=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    feature_maps = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    for i, (imgs, fnames, pids, _) in enumerate(data_loader):
        data_time.update(time.time() - end)

        outputs, output_maps = extract_cnn_feature(model, imgs)
        for fname, output, output_map, pid in zip(fnames, outputs, output_maps, pids):
            features[fname] = output
            feature_maps[fname] = output_map
            labels[fname] = pid

        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % print_freq == 0:
            print('Extract Features: [{}/{}]\t'
                  'Time {:.3f} ({:.3f})\t'
                  'Data {:.3f} ({:.3f})\t'
                  .format(i + 1, len(data_loader),
                          batch_time.val, batch_time.avg,
                          data_time.val, data_time.avg))
    return features, feature_maps, labels


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist = dist.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist

    x = torch.cat([features[f].unsqueeze(0) for f, _, _,  in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _,  in gallery], 0)


    #x = torch.cat([features[f].unsqueeze(0) for f, _, _,_ in query], 0)
    #y = torch.cat([features[f].unsqueeze(0) for f, _, _,  in gallery], 0)




    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(1, -2, x, y.t())
    return dist


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10), dataset=None):
    if query is not None and gallery is not None:
        query_ids = [pid for _, pid, _ in query] # sequence is the last one
        gallery_ids = [pid for _, pid, _ in gallery]
        query_cams = [cam for _, _, cam in query]
        gallery_cams = [cam for _, _, cam in gallery]
    else:
        assert (query_ids is not None and gallery_ids is not None
                and query_cams is not None and gallery_cams is not None)

    # Compute mean AP
    mAP = mean_ap(distmat, query_ids, gallery_ids, query_cams, gallery_cams)
    print('Mean AP: {:4.1%}'.format(mAP))

    # Compute all kinds of CMC scores
    if (dataset == 'market1501') or (dataset == 'dukemtmc'):
        cmc_configs = {
            'market1501': dict(separate_camera_set=False,
                               single_gallery_shot=False,
                               first_match_break=True)
        }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}'
              .format('market1501'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                  .format(k,
                          cmc_scores['market1501'][k - 1]))
        return cmc_scores['market1501'][0], mAP
    if (dataset == 'cuhk03'):
        cmc_configs = {
            'cuhk03': dict(separate_camera_set=True,
                           single_gallery_shot=True,
                           first_match_break=False),
        }
        cmc_scores = {name: cmc(distmat, query_ids, gallery_ids,
                                query_cams, gallery_cams, **params)
                      for name, params in cmc_configs.items()}

        print('CMC Scores{:>12}'
              .format('cuhk03'))
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'
                  .format(k,
                          cmc_scores['cuhk03'][k - 1]))
        # Use the allshots cmc top-1 score for validation criterion
        return cmc_scores['cuhk03'][0], mAP


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, data_loader, query, gallery, metric=None):
        features, feature_maps, _ = extract_features(self.model, data_loader)
        distmat = pairwise_distance(features, query, gallery, metric=metric)
        return evaluate_all(distmat, query=query, gallery=gallery)


class CascadeEvaluator(object):
    def __init__(self, base_model, embed_model, embed_dist_fn=None):
        super(CascadeEvaluator, self).__init__()
        self.base_model = base_model
        self.embed_model = embed_model
        self.embed_dist_fn = embed_dist_fn

    def evaluate(self, data_loader, query, gallery, alpha, cache_file=None,
                 rerank_topk=75, second_stage=True, dataset=None):
        # Extract features image by image
        features, feature_maps, _ = extract_features(self.base_model, data_loader)
        # Compute pairwise distance and evaluate for the first stage
        distmat = pairwise_distance(features, query, gallery)
        print("First stage evaluation:")
        if second_stage:
            evaluate_all(distmat, query=query, gallery=gallery, dataset=dataset)

            # Sort according to the first stage distance
            distmat = to_numpy(distmat)
            rank_indices = np.argsort(distmat, axis=1)

            # Build a data loader for topk predictions for each query
            topk_gallery = [[] for i in range(len(query))]
            for i, indices in enumerate(rank_indices):
                for j in indices[:rerank_topk]:
                    gallery_fname_id_pid = gallery[j]
                    topk_gallery[i].append(gallery_fname_id_pid)

            embeddings = extract_embeddings(self.embed_model, feature_maps, query=query, topk_gallery=topk_gallery,
                                            rerank_topk=rerank_topk)

            embed_new = embeddings
            embeddings = embeddings.data[:, 0]

            # Merge two-stage distances
            for k, embed in enumerate(embeddings):
                i, j = k // rerank_topk, k % rerank_topk
                distmat[i, rank_indices[
                    i, j]] = embed  # updating the distance mat with the new computed distance, based on their index in the rank_indices

            print("Second stage evaluation:")
        return evaluate_all(distmat, query, gallery, dataset=dataset)
