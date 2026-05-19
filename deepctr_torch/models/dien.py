"""
Author:
    Ze Wang, wangze0801@126.com

Reference:
    [1] Zhou G, Mou N, Fan Y, et al. Deep Interest Evolution Network for Click-Through Rate Prediction[J]. arXiv preprint arXiv:1809.03672, 2018. (https://arxiv.org/pdf/1809.03672.pdf)
"""

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .basemodel import BaseModel
from ..layers import *
from ..inputs import *


class DIEN(BaseModel):
    """Instantiates the Deep Interest Evolution Network architecture.

    :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
    :param history_feature_list: list,to indicate  sequence sparse field
    :param gru_type: str,can be GRU AIGRU AUGRU AGRU
    :param use_negsampling: bool, whether or not use negtive sampling
    :param alpha: float ,weight of auxiliary_loss
    :param use_bn: bool. Whether use BatchNormalization before activation or not in deep net
    :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
    :param dnn_activation: Activation function to use in DNN
    :param att_hidden_units: list,list of positive integer , the layer number and units in each layer of attention net
    :param att_activation: Activation function to use in attention net
    :param att_weight_normalization: bool.Whether normalize the attention score of local activation unit.
    :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
    :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param init_std: float,to use as the initialize std of embedding vector
    :param seed: integer ,to use as random seed.
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self,
                 dnn_feature_columns, history_feature_list,
                 gru_type="GRU", use_negsampling=False, alpha=1.0, use_bn=False, dnn_hidden_units=(256, 128),
                 dnn_activation='relu',
                 att_hidden_units=(64, 16), att_activation="relu", att_weight_normalization=True,
                 l2_reg_dnn=0, l2_reg_embedding=1e-6, dnn_dropout=0, init_std=0.0001, seed=1024, task='binary',
                 device='cpu', gpus=None):
        super(DIEN, self).__init__([], dnn_feature_columns, l2_reg_linear=0, l2_reg_embedding=l2_reg_embedding,
                                   init_std=init_std, seed=seed, task=task, device=device, gpus=gpus)

        self.item_features = history_feature_list
        self.use_negsampling = use_negsampling
        self.alpha = alpha
        self._split_columns()

        # structure: embedding layer -> interest extractor layer -> interest evolution layer -> DNN layer -> out

        # embedding layer
        # inherit -> self.embedding_dict
        input_size = self._compute_interest_dim()
        # interest extractor layer
        self.interest_extractor = InterestExtractor(input_size=input_size, use_neg=use_negsampling, init_std=init_std)
        # interest evolution layer
        self.interest_evolution = InterestEvolving(
            input_size=input_size,
            gru_type=gru_type,
            use_neg=use_negsampling,
            init_std=init_std,
            att_hidden_size=att_hidden_units,
            att_activation=att_activation,
            att_weight_normalization=att_weight_normalization)
        # DNN layer
        dnn_input_size = self._compute_dnn_dim() + input_size
        self.dnn = DNN(dnn_input_size, dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, use_bn,
                       init_std=init_std, seed=seed)
        self.linear = nn.Linear(dnn_hidden_units[-1], 1, bias=False)
        # prediction layer
        # inherit -> self.out

        # init
        for name, tensor in self.linear.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)

        self.to(device)








class InterestExtractor(nn.Module):
    def __init__(self, input_size, use_neg=False, init_std=0.001, device='cpu'):
        super(InterestExtractor, self).__init__()
        self.use_neg = use_neg
        self.gru = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        if self.use_neg:
            self.auxiliary_net = DNN(input_size * 2, [100, 50, 1], 'sigmoid', init_std=init_std, device=device)
        for name, tensor in self.gru.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)
        self.to(device)

    def forward(self, keys, keys_length, neg_keys=None):
        """
        Parameters
        ----------
        keys: 3D tensor, [B, T, H]
        keys_length: 1D tensor, [B]
        neg_keys: 3D tensor, [B, T, H]

        Returns
        -------
        masked_interests: 2D tensor, [b, H]
        aux_loss: [1]
        """
        pass



class InterestEvolving(nn.Module):
    __SUPPORTED_GRU_TYPE__ = ['GRU', 'AIGRU', 'AGRU', 'AUGRU']

    def __init__(self,
                 input_size,
                 gru_type='GRU',
                 use_neg=False,
                 init_std=0.001,
                 att_hidden_size=(64, 16),
                 att_activation='sigmoid',
                 att_weight_normalization=False):
        super(InterestEvolving, self).__init__()
        if gru_type not in InterestEvolving.__SUPPORTED_GRU_TYPE__:
            raise NotImplementedError("gru_type: {gru_type} is not supported")
        self.gru_type = gru_type
        self.use_neg = use_neg

        if gru_type == 'GRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=False)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AIGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = nn.GRU(input_size=input_size, hidden_size=input_size, batch_first=True)
        elif gru_type == 'AGRU' or gru_type == 'AUGRU':
            self.attention = AttentionSequencePoolingLayer(embedding_dim=input_size,
                                                           att_hidden_units=att_hidden_size,
                                                           att_activation=att_activation,
                                                           weight_normalization=att_weight_normalization,
                                                           return_score=True)
            self.interest_evolution = DynamicGRU(input_size=input_size, hidden_size=input_size,
                                                 gru_type=gru_type)
        for name, tensor in self.interest_evolution.named_parameters():
            if 'weight' in name:
                nn.init.normal_(tensor, mean=0, std=init_std)


    def forward(self, query, keys, keys_length, mask=None):
        """
        Parameters
        ----------
        query: 2D tensor, [B, H]
        keys: (masked_interests), 3D tensor, [b, T, H]
        keys_length: 1D tensor, [B]

        Returns
        -------
        outputs: 2D tensor, [B, H]
        """
        pass
