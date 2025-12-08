# chapter4-4-SSD/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features = self._make_layers()
    
    def _make_layers(self):
        # cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        return x


class L2Norm(nn.Module):
    def __init__(self, features, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(features))
        self.reset_parameters(scale)

    def reset_parameters(self, scale):
        nn.init.constant_(self.weight, scale)
    
    def forward(self, x):
        x = F.normalize(x, dim=1)
        scale = self.weight[None, :, None, None]
        return scale * x


class VGG16Extractor300(nn.Module):
    def __init__(self):
        super(VGG16Extractor300, self).__init__()
        self.features = VGG16()
        self.norm4 = L2Norm(512, 20)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)

        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

    def forward(self, x):
        hs = []
        h = self.features(x)
        hs.append(self.norm4(h))

        h = F.max_pool2d(h, kernel_size=2, stride=2, ceil_mode=True)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pool2d(h, kernel_size=3, stride=1, padding=1, ceil_mode=True)

        h = F.relu(self.conv6(h))
        h = F.relu(self.conv7(h))
        hs.append(h)

        h = F.relu(self.conv8_1(h))
        h = F.relu(self.conv8_2(h))
        hs.append(h)

        h = F.relu(self.conv9_1(h))
        h = F.relu(self.conv9_2(h))
        hs.append(h)

        h = F.relu(self.conv10_1(h))
        h = F.relu(self.conv10_2(h))
        hs.append(h)

        h = F.relu(self.conv11_1(h))
        h = F.relu(self.conv11_2(h))
        hs.append(h)
        return hs

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, neg_pos_ratio=3):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.neg_pos_ratio = neg_pos_ratio
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, pred_boxes, pred_scores, gt_boxes, gt_labels):
        """
        pred_boxes : [B, 8732, 4]
        pred_scores: [B, 8732, C]
        gt_boxes   : [B, max_gt, 4]   (不需要扩展到8732)
        gt_labels  : [B, max_gt]
        """

        batch_size = pred_boxes.size(0)
        num_anchors = pred_boxes.size(1)

        # -------------------------------------
        # 1. Match ground truth with anchors
        # -------------------------------------
        target_boxes = torch.zeros_like(pred_boxes)  # [B,8732,4]
        target_labels = torch.zeros(batch_size, num_anchors, dtype=torch.long, device=pred_boxes.device)

        for b in range(batch_size):
            if gt_boxes[b].size(0) == 0:
                continue

            iou = self.iou_jaccard(pred_boxes[b], gt_boxes[b])  # [8732, n_gt]
            best_gt_iou, best_gt_idx = iou.max(dim=1) # [8732], [8732]

            # positive anchors
            pos = best_gt_iou > 0.5

            target_boxes[b, pos] = gt_boxes[b][best_gt_idx[pos]]
            target_labels[b, pos] = gt_labels[b][best_gt_idx[pos]]

        pos_mask = target_labels > 0
        num_pos = pos_mask.sum(dim=1)

        # -------------------------------------
        # 2. Localization loss
        # -------------------------------------
        loc_loss = self.smooth_l1(pred_boxes, target_boxes).sum(dim=2)
        loc_loss = (loc_loss * pos_mask.float()).sum()

        # -------------------------------------
        # 3. Classification loss
        # -------------------------------------
        conf_loss = self.ce(pred_scores.view(-1, self.num_classes),
                            target_labels.view(-1))
        conf_loss = conf_loss.view(batch_size, num_anchors)

        # split positive and negative
        conf_loss_pos = conf_loss * pos_mask.float()
        conf_loss_neg = conf_loss.clone()
        conf_loss_neg[pos_mask] = -1

        # 困难样本挖掘 找出负样本中分类损失最大的部分作为困难样本 困难样本可以是任意类别(正样本只能是非背景类)
        num_neg = torch.clamp(self.neg_pos_ratio * num_pos, max=num_anchors - 1)

        conf_loss_neg_sorted, idx = conf_loss_neg.sort(dim=1, descending=True)
        hard_neg_mask = torch.zeros_like(conf_loss_neg)

        for b in range(batch_size):
            hard_neg_mask[b, idx[b, :num_neg[b]]] = 1

        conf_loss_neg_hard = (conf_loss * hard_neg_mask).sum()

        # sum all
        conf_loss = conf_loss_pos.sum() + conf_loss_neg_hard

        # -------------------------------------
        # 4. Normalize
        # -------------------------------------
        total_loss = (loc_loss + conf_loss) / num_pos.clamp(min=1).sum()

        return total_loss

    @staticmethod
    def iou_jaccard(a, b):
        """
        a: [8732,4] predicted boxes
        b: [n_gt,4] gt boxes
        """
        A = a.unsqueeze(1)  # [8732,1,4]
        B = b.unsqueeze(0)  # [1,n_gt,4]

        inter_xmin = torch.max(A[..., 0], B[..., 0])
        inter_ymin = torch.max(A[..., 1], B[..., 1])
        inter_xmax = torch.min(A[..., 2], B[..., 2])
        inter_ymax = torch.min(A[..., 3], B[..., 3])

        inter_w = (inter_xmax - inter_xmin).clamp(min=0)
        inter_h = (inter_ymax - inter_ymin).clamp(min=0)
        inter = inter_w * inter_h

        area_a = (A[..., 2] - A[..., 0]) * (A[..., 3] - A[..., 1])
        area_b = (B[..., 2] - B[..., 0]) * (B[..., 3] - B[..., 1])

        union = area_a + area_b - inter
        return inter / union.clamp(min=1e-6)

def get_baseAnchors():
    steps = SSD300.steps
    box_sizes = SSD300.box_sizes
    aspect_ratios = SSD300.aspect_ratios
    fm_sizes = SSD300.fm_sizes
    # 38*38*4 + 19*19*6 + 10*10*6 + 5*5*6 + 3*3*4 + 1*1*4 = 8732
    anchors = []
    for k, fm_size in enumerate(fm_sizes):
        for i in range(fm_size):
            for j in range(fm_size):
                f_k = 300 / steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = box_sizes[k] / 300
                anchors.append([cx, cy, s_k, s_k])

                s_k_prime = (box_sizes[k] * box_sizes[k + 1]) ** 0.5 / 300
                anchors.append([cx, cy, s_k_prime, s_k_prime])

                for ar in aspect_ratios[k]:
                    anchors.append([cx, cy, s_k * (ar ** 0.5), s_k / (ar ** 0.5)])
                    anchors.append([cx, cy, s_k / (ar ** 0.5), s_k * (ar ** 0.5)])
    return torch.tensor(anchors)

def get_predictedAnchors(loc_preds, baseAnchors, variances=[0.1, 0.2]):
    anchors = baseAnchors.unsqueeze(0).expand(loc_preds.size(0), -1, -1)
    cx = loc_preds[:, :, 0] * variances[0] * anchors[:, :, 2] + anchors[:, :, 0]
    cy = loc_preds[:, :, 1] * variances[0] * anchors[:, :, 3] + anchors[:, :, 1]
    w = torch.exp(loc_preds[:, :, 2] * variances[1]) * anchors[:, :, 2]
    h = torch.exp(loc_preds[:, :, 3] * variances[1]) * anchors[:, :, 3]

    boxes = torch.zeros_like(loc_preds)
    boxes[:, :, 0] = cx - w / 2
    boxes[:, :, 1] = cy - h / 2
    boxes[:, :, 2] = cx + w / 2
    boxes[:, :, 3] = cy + h / 2
    return boxes

class SSD300(nn.Module):
    steps = [8, 16, 32, 64, 100, 300]
    box_sizes = (30, 60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))
    fm_sizes = (38, 19, 10, 5, 3, 1)
    def __init__(self, num_classes):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        self.anchors = (4, 6, 6, 6, 4, 4)
        self.in_channels = (512, 1024, 512, 256, 256, 256)

        self.extractor = VGG16Extractor300()
        self.loc_layers = nn.ModuleList()
        self.conf_layers = nn.ModuleList()
        for i in range(len(self.in_channels)):
            self.loc_layers.append(
                nn.Conv2d(self.in_channels[i], self.anchors[i] * 4, kernel_size=3, padding=1)
            )
            self.conf_layers.append(
                nn.Conv2d(self.in_channels[i], self.anchors[i] * self.num_classes, kernel_size=3, padding=1)
            )
        
        self.anchors = get_baseAnchors()
    
    def forward(self, x):
        loc_preds, cls_preds = [], []
        xs = self.extractor(x)
        for i, x in enumerate(xs):
            loc_pred = self.loc_layers[i](x)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds.append(loc_pred.view(loc_pred.size(0), -1, 4))

            cls_pred = self.conf_layers[i](x)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds.append(cls_pred.view(cls_pred.size(0), -1, self.num_classes))
        
        loc_preds = torch.cat(loc_preds, dim=1)
        cls_preds = torch.cat(cls_preds, dim=1)
        predicted_anchors = get_predictedAnchors(loc_preds, self.anchors)
        return predicted_anchors, cls_preds


if __name__ == "__main__":
    x = torch.randn(1, 3, 300, 300)
    vgg16 = VGG16()
    output = vgg16(x)
    print("vgg16 Input Shape :", x.shape, "Output Shape :", output.shape)
    vgg16_extractor300 = VGG16Extractor300()
    outputs = vgg16_extractor300(x)
    print("vgg16_extractor300 Input Shape :", x.shape)
    for i, out in enumerate(outputs):
        print(f"Output {i} Shape :", out.shape)
    
    ssd300 = SSD300(num_classes=21)
    predicted_anchors, cls_preds = ssd300(x)
    print("SSD300 Input Shape :", x.shape)
    print("Location Predictions Shape :", predicted_anchors.shape)
    print("Class Predictions Shape    :", cls_preds.shape)
