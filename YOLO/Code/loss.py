from genericpath import exists
import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predctions, target):
        predctions = predctions.reshape(-1, self.S, self.S, self.C + self.B*5)

        iou_b1 = intersection_over_union(predctions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predctions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0) # bestbox를 이용해서 2개의 bbox중 gt와 iou가 높은 것을 선택함.
        exists_box = target[..., 20].unsqueeze(3) # Iobj_i 에 객체가 존재하면 1, 존재하지 않다면 0

        # for Box Coordinates
        box_predictions = exists_box * (
            (
                bestbox * predctions[..., 26:30]
                + (1-bestbox) * predctions[..., 21:25]
            )
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6) # 예측한 h, w 루트 연산 in paper
        )

        # (N, S, S, 25)
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4]) # gt의 h, w 루트 연산 in paper

        # flatten 전 : (N, S, S, 4) -> flatten 후 : (N * S, S, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2),
        )

        # For object loss
        pred_box = (
            bestbox * predctions[..., 25:26] + (1-bestbox) * predctions[..., 20:21]
        )

        # (N*S*S)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21] * iou_maxes) # confidence score를 이용해 loss 를 계산하기 위해 iou_maxes를 곱해준다.
        )

        # For No object Loss
        # flatten 전 (N, S, S, 1) -> flatten 후: (N, S*S)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predctions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predctions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # For class Loss
        # flatten 전 : (N, S, S, 20) -> flatten 후 (N*S*S, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * predctions[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = (
            self.lambda_coord * box_loss # First tow rows of loss in paer
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )