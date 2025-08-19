from .module import Module
from .. import functional as F
from ...core import Tensor


class Loss(Module):
    '''损失函数基类'''

    def __init__(self, reduction='mean') -> None:
        super().__init__()
        self.reduction = reduction
        assert self.reduction in {'mean', 'sum'}

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        raise NotImplementedError


class MSELoss(Loss):

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.mse_loss(y_pred, y_true, reduction=self.reduction)


class NLLLoss(Loss):

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.nll_loss(y_pred, y_true, reduction=self.reduction)


class CrossEntropyLoss(Loss):

    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        return F.cross_entropy_loss(y_pred, y_true, reduction=self.reduction)
