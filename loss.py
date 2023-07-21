import torch.nn as nn
import torch
from vae_decoder import vae_loss
from functools import partial

class Loss(nn.Module):
    def __init__(self, class_criterion_weight=1, decoder_weight=0,
                       certainty_weight=0, word_embdding_weight=0,
                       word_embdding_loss_param=None,
                       vae_loss_param=None
                       ):
        super().__init__()
        self.class_criterion = nn.CrossEntropyLoss()
        self.certainty_criterion = nn.MSELoss()
        # self.vae_criterion = nn.L1Loss()
        if decoder_weight:
            self.vae_criterion = partial(vae_loss, **vae_loss_param)
        

        # kw = {k: v for k,v in kw.items() if v is not None}
        if word_embdding_weight:
            if word_embdding_loss_param is None:
                word_embdding_loss_param = {}
            self.word_embdding_loss = TripletLoss(**word_embdding_loss_param)

        self.decoder_weight = decoder_weight
        self.certainty_weight = certainty_weight
        self.class_criterion_weight = class_criterion_weight
        self.word_embdding_weight = word_embdding_weight

    def forward(self, pred, target, img=None):
        l1 = l2 = l3 = l4 = 0

        class_prob = pred["class_prob"]

        if "decoded_image" in pred:
            decoded_image, mean, z_log_sigma2 = pred["decoded_image"]

        if "certainty_pred" in pred:
            certainty_pred = pred["certainty_pred"]
        
        if "embedding" in pred:
            embedding = pred["embedding"]

        if self.class_criterion_weight != 0:
            l1 = self.class_criterion(class_prob, target)

        if self.decoder_weight != 0:
            l2 = self.vae_criterion(decoded_image, img, mean, z_log_sigma2)

        if self.certainty_weight != 0:
            target = 1.0 * (target == class_prob.argmax(dim=1))
            target = target.squeeze(dim=-1)
            l3 = self.certainty_criterion(certainty_pred, target)
        
        if self.word_embdding_weight != 0:
            l4 = self.word_embdding_loss(embedding, target)

        return (self.class_criterion_weight * l1
                + self.decoder_weight * l2
                + self.certainty_weight * l3
                + self.word_embdding_weight * l4)


class TripletLoss(nn.Module):
    def __init__(self, label_embedding, margin=1.0, positive_aggregator="max"):
        super().__init__()

        self.label_embedding_matrix = label_embedding
        self.positive_aggregator = positive_aggregator
        self.margin = margin


    def forward(self, embeddings, labels):

        label_embeddings = self.label_embedding_matrix[labels] 

        dist_matrix = torch.cdist(embeddings.unsqueeze(0),
                                  label_embeddings.unsqueeze(0)).squeeze(0)
        
        dist_matrix = dist_matrix ** 2
        
        labels_matrix = (torch.cdist(labels.unsqueeze(1).unsqueeze(0) * 1.0,
                                    labels.unsqueeze(1).unsqueeze(0) * 1.0).squeeze() == 0) * 1.0

        positive_dists = torch.multiply(dist_matrix, labels_matrix)
        if self.positive_aggregator == "max":
            positive_dist = torch.max(positive_dists, dim=0).values
        elif self.positive_aggregator == "sum":
            positive_dist = torch.sum(positive_dists, dim=0)
        else:
            raise NotImplementedError("positive_aggregator should be one of the following: \"max\", \"sum\"")
        negative_dist  = torch.min(torch.multiply(dist_matrix, 1.0 - labels_matrix), dim=0).values
    
        loss = torch.sum(torch.clamp(positive_dist - negative_dist + self.margin, min=0))

        return loss