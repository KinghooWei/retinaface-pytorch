import torch
import torch.nn as nn
import torch.nn.functional as F

#------------------------------#
#   获得框的左上角和右下角
#------------------------------#
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,
                     boxes[:, :2] + boxes[:, 2:]/2), 1)

#------------------------------#
#   获得框的中心和宽高
#------------------------------#
def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,
                     boxes[:, 2:] - boxes[:, :2], 1)

#----------------------------------#
#   计算所有真实框和先验框的交面积
#----------------------------------#
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    #------------------------------#
    #   获得交矩形的左上角
    #------------------------------#
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    #------------------------------#
    #   获得交矩形的右下角
    #------------------------------#
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    #-------------------------------------#
    #   计算先验框和所有真实框的重合面积
    #-------------------------------------#
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    #-------------------------------------#
    #   返回的inter的shape为[A,B]
    #   代表每一个真实框和先验框的交矩形
    #-------------------------------------#
    inter = intersect(box_a, box_b)
    #-------------------------------------#
    #   计算先验框和真实框各自的面积
    #-------------------------------------#
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]

    union = area_a + area_b - inter
    #-------------------------------------#
    #   每一个真实框和先验框的交并比[A,B]
    #-------------------------------------#
    return inter / union  # [A,B]

def encode(matched, priors, variances):
    # 进行编码的操作
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # 中心编码
    g_cxcy /= (variances[0] * priors[:, 2:])
    
    # 宽高编码
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def encode_landm(matched, priors, variances):
    matched = torch.reshape(matched, (matched.size(0), 18, 2))
    priors_cx = priors[:, 0].unsqueeze(1).expand(matched.size(0), 18).unsqueeze(2)
    priors_cy = priors[:, 1].unsqueeze(1).expand(matched.size(0), 18).unsqueeze(2)
    priors_w = priors[:, 2].unsqueeze(1).expand(matched.size(0), 18).unsqueeze(2)
    priors_h = priors[:, 3].unsqueeze(1).expand(matched.size(0), 18).unsqueeze(2)
    priors = torch.cat([priors_cx, priors_cy, priors_w, priors_h], dim=2)

    # 减去中心后除上宽高
    g_cxcy = matched[:, :, :2] - priors[:, :, :2]
    g_cxcy /= (variances[0] * priors[:, :, 2:])
    g_cxcy = g_cxcy.reshape(g_cxcy.size(0), -1)
    return g_cxcy

def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def match(threshold, truths, priors, variances, labels, landms, loc_t, conf_t, landm_t, class_t, idx):
    #----------------------------------------------#
    #   计算所有的先验框和真实框的重合程度
    #----------------------------------------------#
    # (num_targets, num_priors)
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    #----------------------------------------------#
    #   所有真实框和先验框的最好重合程度
    #   best_prior_overlap [truth_box,1]
    #   best_prior_idx [truth_box,1]
    #----------------------------------------------#
    # (num_targets)
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)

    #----------------------------------------------#
    #   所有先验框和真实框的最好重合程度
    #   best_truth_overlap [1,prior]
    #   best_truth_idx [1,prior]
    #----------------------------------------------#
    # (num_priors)
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    num_priors = best_truth_idx.size(0)

    #----------------------------------------------#
    #   用于保证每个真实框都至少有对应的一个先验框
    #----------------------------------------------#
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)
    # 对best_truth_idx内容进行设置
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j

    #----------------------------------------------#
    # 获取每一个先验框对应的真实框(num_priors,4)
    #----------------------------------------------#
    matches = truths[best_truth_idx]            
    # 获取每一个先验框对应的置信度(num_priors)，如果重合程度小于threhold则认为是背景
    # (num_priors)
    conf = torch.Tensor(num_priors).fill_(1)
    conf[best_truth_overlap < threshold] = 0
    # 每一个先验框对应关键点(num_priors, 36)
    matches_landm = landms[best_truth_idx]
    # 获取每一个先验框对应的类别标签(num_priors, 3)
    # clazz = torch.LongTensor(num_priors, 3).fill_(0)
    # prior_labels = labels[best_truth_idx]
    # for i in range(num_priors):
    #     index = (torch.LongTensor([i, prior_labels[i]]),)
    #     clazz.index_put_(index, torch.LongTensor(1))
    # clazz[best_truth_overlap < threshold] = 0
    # 获取每一个先验框对应的类别(num_priors)，如果重合程度小于threhold则认为是背景
    clazz = labels[best_truth_idx]
    clazz[best_truth_overlap < threshold] = 0

    #----------------------------------------------#
    #   利用真实框和先验框进行编码
    #   编码后的结果就是网络应该有的预测结果
    #----------------------------------------------#
    loc = encode(matches, priors, variances)
    landm = encode_landm(matches_landm, priors, variances)

    #----------------------------------------------#
    #   [num_priors, 4]
    #----------------------------------------------#
    loc_t[idx] = loc
    #----------------------------------------------#
    #   [num_priors]
    #----------------------------------------------#
    conf_t[idx] = conf
    #----------------------------------------------#
    #   [num_priors, 10]
    #----------------------------------------------#
    landm_t[idx] = landm
    class_t[idx] = clazz


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, neg_pos, variance, cuda=True):
        super(MultiBoxLoss, self).__init__()
        #----------------------------------------------#
        #   对于retinaface而言num_classes等于3
        #----------------------------------------------#
        self.num_classes    = num_classes
        #----------------------------------------------#
        #   重合程度在多少以上认为该先验框可以用来预测
        #----------------------------------------------#
        # 0.35
        self.threshold      = overlap_thresh
        #----------------------------------------------#
        #   正负样本的比率
        #----------------------------------------------#
        # 7
        self.negpos_ratio   = neg_pos
        # [0.1, 0.2]
        self.variance       = variance
        self.cuda           = cuda

    # 对一个batch的样本计算损失
    def forward(self, predictions, priors, targets):
        #--------------------------------------------------------------------#
        #   取出预测结果的四个值：框的回归信息、置信度、人脸关键点的回归信息、类别信息
        #--------------------------------------------------------------------#
        loc_data, conf_data, landm_data, class_data = predictions
        #--------------------------------------------------#
        #   计算出batch_size和先验框的数量
        #--------------------------------------------------#
        num         = loc_data.size(0)
        num_priors  = (priors.size(0))

        #--------------------------------------------------#
        #   创建一个真实标签tensor进行处理
        #--------------------------------------------------#
        loc_t   = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 36)
        conf_t  = torch.Tensor(num, num_priors)
        class_t = torch.LongTensor(num, num_priors)

        # --------------------------------------------------#
        #   转化成Variable
        #   loc_t   (num, num_priors, 4)
        #   conf_t  (num, num_priors)
        #   landm_t (num, num_priors, 10)
        # --------------------------------------------------#
        zeros = torch.tensor(0.)
        if self.cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()
            class_t = class_t.cuda()
            zeros = zeros.cuda()

        # 遍历一个batch的样本
        for idx in range(num):
            # 获得真实框与标签
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, 40].data
            landms = targets[idx][:, 4:40].data

            # 获得先验框
            defaults = priors.data
            #--------------------------------------------------#
            #   利用真实框和先验框进行匹配。
            #   如果真实框和先验框的重合度较高，则认为匹配上了。
            #   该先验框用于负责检测出该真实框。
            #--------------------------------------------------#
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, class_t, idx)
            


        #------------------------------------------------------------------------#
        #   有人脸关键点的人脸真实框的标签为1，没有人脸关键点的人脸真实框标签为-1
        #   所以计算人脸关键点loss的时候pos1 = conf_t > zeros
        #   计算人脸框的loss的时候pos = conf_t != zeros
        #------------------------------------------------------------------------#
        # (batch, num_priors)
        pos = conf_t > zeros
        # (batch, num_priors, 36)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(landm_data)
        # (473, 10) 某一batch有473个有关键点的正样本
        landm_p = landm_data[pos_idx].view(-1, 36)
        landm_t = landm_t[pos_idx].view(-1, 36)
        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')

        # (batch, num_priors, 4)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # (473, 4) 某一batch有473个有真实框的正样本
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')



        #--------------------------------------------------#
        #   batch_conf  (num * num_priors, 2)
        #   loss_c      (num, num_priors)
        #--------------------------------------------------#
        # conf_t[pos] = 1
        # (batch*num_priors, 1)
        batch_conf = conf_data.view(-1, 1)
        # 这个地方是在寻找难分类的先验框
        # loss_conf = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_conf = batch_conf - conf_t.view(-1, 1).float()

        # 难分类的先验框不把正样本考虑进去，只考虑难分类的负样本
        # (batch*anchor, 1)
        loss_conf[pos.view(-1, 1)] = 0
        # (batch, anchor)
        loss_conf = loss_conf.view(num, -1)
        #--------------------------------------------------#
        #   loss_idx    (num, num_priors)
        #   idx_rank    (num, num_priors)
        #--------------------------------------------------#
        _, loss_idx = loss_conf.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        #--------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   num_pos     (num, )
        #   neg         (num, num_priors)
        #--------------------------------------------------#
        num_pos = pos.long().sum(1, keepdim=True)
        # 限制负样本数量
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        #--------------------------------------------------#
        #   求和得到每一个图片内部有多少正样本
        #   pos_idx   (num, num_priors, num_classes)
        #   neg_idx   (num, num_priors, num_classes)
        #--------------------------------------------------#
        # pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        # neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        
        # 选取出用于训练的正样本与负样本，计算loss
        # (3888, 2)
        # conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        conf_p = conf_data[(pos + neg).gt(0)].view(-1, 1)
        targets_weighted = conf_t[(pos+neg).gt(0)].view(-1, 1)
        # loss_conf = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        loss_conf = F.smooth_l1_loss(conf_p, targets_weighted, reduction='sum')


        class_p = class_data[(pos + neg).gt(0)].view(-1, 3)
        class_t = class_t[(pos+neg).gt(0)].view(-1, 1)
        # class_t = class_t[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        loss_c = F.cross_entropy(class_p, class_t.squeeze(), reduction='sum')

        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_conf /= N

        # num_pos_landm = pos1.long().sum(1, keepdim=True)
        # N1 = max(num_pos_landm.data.sum().float(), 1)
        loss_landm /= N
        loss_c /= N
        return loss_l, loss_conf, loss_landm, loss_c

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
