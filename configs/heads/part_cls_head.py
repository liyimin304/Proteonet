from .cls_head import ClsHead
import torch.nn as nn
import torch
import torch.nn.functional as F
from ..losses import *
from core.evaluations import Accuracy


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class BottleClassifier(nn.Module):

    def __init__(self, in_dim, out_dim, relu=True, dropout=True, bottle_dim=512):
        super(BottleClassifier, self).__init__()

        bottle = [nn.Linear(in_dim, bottle_dim)]
        bottle += [nn.BatchNorm1d(bottle_dim)]
        if relu:
            bottle += [nn.LeakyReLU(0.1)]
        if dropout:
            bottle += [nn.Dropout(p=0.5)]
        bottle = nn.Sequential(*bottle)
        bottle.apply(weights_init_kaiming)
        self.bottle = bottle

        classifier = [nn.Linear(bottle_dim, out_dim)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.bottle(x)
        x = self.classifier(x)
        return x

class PartClsHead(ClsHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 part_num,
                 init_cfg=dict(type='Normal', layer='Linear', std=0.01),
                 *args,
                 **kwargs):
        super(PartClsHead, self).__init__(init_cfg=init_cfg, *args, **kwargs)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.part_num = part_num
        if 'axu_loss' in kwargs:
            aux_loss = kwargs['axu_loss']
            self.compute_aux_loss = eval(aux_loss.pop('type'))(**aux_loss)
        else:
            self.compute_aux_loss = None

        if self.num_classes <= 0:
            raise ValueError(
                f'num_classes={num_classes} must be a positive integer')

        # classifier
        for i in range(part_num):
            name = 'classifier' + str(i)
            setattr(self, name, BottleClassifier(
                2048, self.num_classes, relu=True,
                dropout=False, bottle_dim=256)
                    )
        self.fc = nn.Linear(self.in_channels, self.num_classes)
        # embedding
        for i in range(part_num):
            name = 'embedder' + str(i)
            setattr(self, name, nn.Linear(2048, 256))

    def loss(self, score_list, gt_label, **kwargs):
        losses = dict()
        # compute loss

        global_score, part_scores = score_list
        avg_logits, part_score = part_scores

        num_samples = len(global_score)

        losses[f'loss_global'] = self.compute_loss(
            avg_logits, gt_label, avg_factor=num_samples, **kwargs)
        loss_all = losses[f'loss_global']
        # loss_all = 0
        for i in range(len(part_score)):
            logits_i = part_score[i]
            ide_loss_i = self.compute_loss(
                logits_i, gt_label, avg_factor=self.part_num, **kwargs)
            losses[f'loss_{i}'] = 1.0 / float(self.part_num) * ide_loss_i
            # losses[f'loss_{i}'] = ide_loss_i
            loss_all += losses[f'loss_{i}']

        if self.cal_acc:
            # compute accuracy
            with torch.no_grad():
                global_acc = self.compute_accuracy(global_score, gt_label)
                part_acc = self.compute_accuracy(avg_logits, gt_label)
                assert len(global_acc) == len(self.topk)
                losses['global_acc'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, global_acc)
                }
                losses['part_acc'] = {
                    f'top-{k}': a
                    for k, a in zip(self.topk, part_acc)
                }
        losses['loss'] = loss_all
        return losses

    def pre_logits(self, features):
        features_part, features_global = features
        feature_list = [features_global]

        for i in range(self.part_num):
            if self.part_num == 1:
                features_i = features_part
            else:
                features_i = torch.squeeze(features_part[:, :, i])
            feature_list.append(features_i)

        return feature_list

    def get_logits(self, feature_list):
        avg_logits = 0
        logits_list = []
        global_logits = self.fc(feature_list[0])

        part_feature_list = feature_list[1:]
        for i in range(len(part_feature_list)):
            features_i = part_feature_list[i]

            classifier_i = getattr(self, 'classifier'+str(i))
            logits_i = classifier_i(features_i)
            logits_list.append(logits_i)

            avg_logits += 1.0 / float(self.part_num) * logits_i
        return global_logits, logits_list, avg_logits

    def forward_train(self, x, gt_label):

        feature_list = self.pre_logits(x)
        global_logits, logits_list, avg_logits = self.get_logits(feature_list)
        losses = self.loss([global_logits, (avg_logits, logits_list)], gt_label)

        return losses


    def simple_test(self, features, softmax=True, post_process=False):

        feature_list = self.pre_logits(features)
        global_logits, logits_list, avg_logits = self.get_logits(feature_list)

        # cls_score = (global_logits + avg_logits) / 2
        # cls_score = 0.2 * global_logits + 0.8 * avg_logits
        cls_score = avg_logits

        if softmax:
            pred = (
                F.softmax(cls_score, dim=1) if cls_score is not None else None)
        else:
            pred = cls_score

        return pred