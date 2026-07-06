
import torch
import random
import numpy as np
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(0)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) :
        counter = 0
        loss=0
        for i in range(images.shape[0]):
            anchor_img = images[i]
            anchor_label = labels[i]

            pindex = torch.ones(images.shape[0], dtype=torch.bool)
            pindex[i] = 0
            pindex[labels != anchor_label] = 0
            if any(pindex) :
                positive_img = random.choice(images[pindex])
            else:
                continue

            nindex = torch.ones(images.shape[0], dtype=torch.bool)
            nindex[i] = 0
            nindex[labels == anchor_label] = 0
            negative_img = random.choice(images[nindex])

            distance_positive = self.calc_euclidean(anchor_img, positive_img)
            distance_negative = self.calc_euclidean(anchor_img, negative_img)

            counter = counter+1
            loss = loss + torch.relu(distance_positive - distance_negative + self.margin)
        return loss/counter