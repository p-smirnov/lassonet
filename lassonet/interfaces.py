from itertools import islice
from abc import ABCMeta, abstractmethod, abstractstaticmethod
from dataclasses import dataclass
from functools import partial
from typing import List
from warnings import warn
from copy import copy

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import torch
from scipy.stats import pearsonr, spearmanr

from .model import LassoNet


def abstractattr(f):
    return property(abstractmethod(f))


@dataclass
class HistoryItem:
    lambda_: float
    state_dict: dict
    val_loss: float
    regularization: float
    selected: torch.BoolTensor
    n_iters: int
    train_loss: float
    validation_metrics: tuple


class BaseLassoNet(BaseEstimator, metaclass=ABCMeta):
    def __init__(
            self,
            *,
            hidden_dims=(100,),
            eps_start=1,
            lambda_start=None,
            path_multiplier=1.02,
            M=10,
            optim=None,
            n_iters=(1000, 100),
            patience=(100, 10),
            tol=0.99,
            val_size=0.1,
            device=None,
            verbose=0,
            random_state=None,
            torch_seed=None,
            lr=1e-3,
    ):
        """
        Parameters
        ----------
        hidden_dims : tuple of int, default=(100,)
            Shape of the hidden layers.
        eps_start : float, default=1
            Sets lambda_start such that it has a strength comparable to the
            loss of the unconstrained model multiplied by eps_start.
        lambda_start : float, default=None
            First value on the path.
        path_multiplier : float
            Multiplicative factor (:math:`1 + \\epsilon`) to increase
            the penalty parameter over the path
        M : float, default=10.0
            Hierarchy parameter.
        optim : torch optimizer or tuple of 2 optimizers, default=None
            Optimizer for initial training and path computation.
            Default is Adam(lr=1e-3), SGD(lr=1e-3, momentum=0.9).
        n_iters : int or pair of int, default=(1000, 100)
            Maximum number of training epochs for initial training and path computation.
            This is an upper-bound on the effective number of epochs, since the model
            uses early stopping.
        patience : int or pair of int, default=10
            Number of epochs to wait without improvement during early stopping.
        val_size : float, default=0.1
            Proportion of data to use for early stopping.
        device : torch device, default=None
            Device on which to train the model using PyTorch.
            Default: GPU if available else CPU
        verbose : int, default=0
        random_state
            Random state for cross-validation
        torch_seed
            Torch state for model random initialization

        """

        self.hidden_dims = hidden_dims
        self.eps_start = eps_start
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier
        self.M = M
        if optim is None:
            optim = (
                partial(torch.optim.Adam, lr=lr),
                partial(torch.optim.SGD, lr=lr, momentum=0.9),
            )
        if isinstance(optim, torch.optim.Optimizer):
            optim = (optim, optim)
        self.optim_init, self.optim_path = optim
        if isinstance(n_iters, int):
            n_iters = (n_iters, n_iters)
        self.n_iters_init, self.n_iters_path = n_iters
        if isinstance(patience, int):
            patience = (patience, patience)
        self.patience_init, self.patience_path = patience
        self.tol = tol
        self.val_size = val_size
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.verbose = verbose

        self.random_state = random_state
        self.torch_seed = torch_seed

        self.model = None

    @abstractmethod
    def _convert_y(self, y) -> torch.TensorType:
        """Convert y to torch tensor"""
        raise NotImplementedError

    @abstractstaticmethod
    def _output_shape(cls, y):
        """Number of model outputs"""
        raise NotImplementedError

    @abstractattr
    def criterion(cls):
        raise NotImplementedError

    def _init_model(self, X, y):
        """Create a torch model"""
        output_shape = self._output_shape(y)
        if self.torch_seed is not None:
            torch.manual_seed(self.torch_seed)
        self.model = LassoNet(
            X.shape[1],
            *self.hidden_dims,
            output_shape,
        ).to(self.device)

    def _cast_input(self, X, y=None):
        X = torch.FloatTensor(X).to(self.device)
        if y is None:
            return X
        y = self._convert_y(y)
        return X, y

    def fit(self, X, y):
        """Train the model.
        Note that if `lambda_` is not given, the trained model
        will most likely not use any feature.
        """
        self.path(X, y)
        return self

    @abstractmethod
    def validation_metrics(self, X_val, y_val):
        raise NotImplementedError

    def _train(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            epochs,
            lambda_,
            optimizer,
            patience=None,
    ):
        model = self.model

        def validation_loss():
            with torch.no_grad():
                model.eval()
                return (
                        self.criterion(model(X_val), y_val).item()
                        + lambda_ * model.regularization().item()
                )

        def train_loss():
            with torch.no_grad():
                model.eval()
                return (
                        self.criterion(model(X_train), y_train).item()
                        + lambda_ * model.regularization().item()
                )

        best_obj = validation_loss()
        epochs_since_best_obj = 0

        for epoch in range(epochs):

            def closure():
                optimizer.zero_grad()
                loss = self.criterion(model(X_train), y_train)
                loss.backward()
                return loss

            model.train()
            optimizer.step(closure)
            if lambda_:
                model.prox(lambda_=lambda_ * optimizer.param_groups[0]["lr"], M=self.M)

            obj = validation_loss()
            if obj < self.tol * best_obj:
                best_obj = obj
                epochs_since_best_obj = 0
            else:
                epochs_since_best_obj += 1
            if patience is not None and epochs_since_best_obj == patience:
                break

        return lambda_, epoch + 1, obj, train_loss(), self.validation_metrics(X_val, y_val)

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractstaticmethod
    def _lambda_max(X, y):
        raise NotImplementedError

    def path(self, X_train, y_train, X_val=None, y_val=None, lambda_=None, lambda_path=None) -> List[HistoryItem]:
        """Train LassoNet on a lambda_ path.
        The path is defined by the class parameters:
        start at `eps * lambda_max` and increment according
        to `path_multiplier` or `n_lambdas`.
        The path will stop when no feature is being used anymore.


        'X_val' and 'y_val' are optional parameters allowing a
        validation set to be passed in. If omitted, a validation
        set will be randomly created by reserving 'model.val_size'
        proportion of the training data.

        The optional `lambda_` argument will also stop the path when
        this value is reached.
        """
        if (X_val is None or y_val is None):
            if (y_val is not None):
                warn("y_val passed in without passing in X_val. Ignoring y_val.")
            if (X_val is not None):
                warn("X_val passed in without passing in y_val. Ignoring X_val.")
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=self.val_size)
        if lambda_ is None:
            lambda_ = float('inf')

        X_train, y_train = self._cast_input(X_train, y_train)
        X_val, y_val = self._cast_input(X_val, y_val)

        hist = []

        def register(hist, lambda_, n_iters, val_loss, train_loss, validation_metrics):
            hist.append(
                HistoryItem(
                    lambda_=lambda_,
                    state_dict=self.model.cpu_state_dict(),
                    val_loss=val_loss,
                    regularization=self.model.regularization().item(),
                    selected=self.model.input_mask(),
                    n_iters=n_iters,
                    train_loss=train_loss,
                    validation_metrics=validation_metrics,
                )
            )

        if self.model is None:
            self._init_model(X_train, y_train)

        register(
            hist,
            *self._train(
                X_train,
                y_train,
                X_val,
                y_val,
                lambda_=0,
                epochs=self.n_iters_init,
                optimizer=self.optim_init(self.model.parameters()),
                patience=self.patience_init,
            ),
        )
        if self.verbose:
            print(
                f"Initialized dense model in {hist[-1].n_iters} epochs, "
                f"val loss {hist[-1].val_loss:.2e}, "
                f"regularization {hist[-1].regularization:.2e}"
            )
        if self.lambda_start is not None:
            current_lambda = self.lambda_start
        else:
            # don't take hist[-1].regularization into account!
            current_lambda = self.eps_start * hist[-1].val_loss
        optimizer = self.optim_path(self.model.parameters())


        def path_step(current_lambda):
            register(
                hist,
                *self._train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    lambda_=current_lambda,
                    epochs=self.n_iters_path,
                    optimizer=optimizer,
                    patience=self.patience_path,
                ),
            )
            last = hist[-1]
            if len(hist) > 1:
                prev = hist[-2]
                # compute loss of previous solution at current lambda
                prev_loss = prev.train_loss + (-prev.lambda_ + current_lambda) * prev.regularization
                if prev_loss < last.train_loss:
                    # use previous solution with lower loss for warm start
                    self.model.load_state_dict(prev.state_dict)
                    last = copy(prev)
                    hist.pop()
                    last.train_loss = prev_loss
                    hist.append(last)

            if self.verbose:
                print(
                    f"Lambda = {current_lambda:.2e}, "
                    f"selected {self.model.selected_count()} features "
                    f"in {last.n_iters} epochs"
                )
                print(
                    f"val_loss (excl regularization) "
                    f"{last.val_loss - last.lambda_ * last.regularization:.2e}, "
                    f"regularization {last.regularization:.2e}"
                )

            return(current_lambda * self.path_multiplier)


        if lambda_path is not None:
            for current_lambda in lambda_path:
                path_step(current_lambda)
        else:
            while self.model.selected_count() != 0 and current_lambda < lambda_:
                current_lambda=path_step(current_lambda)


        self.feature_importances_ = self._compute_feature_importances(hist)
        """When does each feature disappear on the path?"""

        return hist

    @staticmethod
    def _compute_feature_importances(path: List[HistoryItem]):
        """When does each feature disappear on the path?

        Parameters
        ----------
        path : List[HistoryItem]

        Returns
        -------
            feature_importances_
        """

        current = path[0].selected.clone()
        ans = torch.full(current.shape, float("inf"))
        for save in islice(path, 1, None):
            lambda_ = save.lambda_
            diff = current & ~save.selected
            ans[diff.nonzero().flatten()] = lambda_
            current &= save.selected
        return ans

    def load(self, state_dict):
        if self.model is None:
            output_shape, input_shape = state_dict["skip.weight"].shape
            self.model = LassoNet(
                input_shape,
                *self.hidden_dims,
                output_shape,
            ).to(self.device)

        self.model.load_state_dict(state_dict)


class LassoNetRegressor(
    RegressorMixin,
    MultiOutputMixin,
    BaseLassoNet,
):
    """Use LassoNet as regressor"""

    def _convert_y(self, y):
        y = torch.FloatTensor(y).to(self.device)
        if len(y.shape) == 1:
            y = y.view(-1, 1)
        return y

    @staticmethod
    def _output_shape(y):
        return y.shape[1]

    @staticmethod
    def _lambda_max(X, y):
        n_samples, _ = X.shape
        return torch.tensor(X.T.dot(y)).abs().max().item() / n_samples

    criterion = torch.nn.MSELoss(reduction="mean")

    def predict(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X))
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def validation_metrics(self, X_val, y_val):
        with torch.no_grad():
            self.model.eval()
            preds = self.model(X_val).cpu().numpy()
            y_val = y_val.cpu()
            return (
                (pearsonr(preds.reshape(-1), y_val.reshape(-1)), spearmanr(preds.reshape(-1), y_val.reshape(-1)), r2_score(preds, y_val))
            )


class LassoNetClassifier(
    ClassifierMixin,
    BaseLassoNet,
):
    """Use LassoNet as classifier"""

    def _convert_y(self, y) -> torch.TensorType:
        y = torch.LongTensor(y).to(self.device)
        assert len(y.shape) == 1, "y must be 1D"
        return y

    @staticmethod
    def _output_shape(y):
        return (y.max() + 1).item()

    @staticmethod
    def _lambda_max(X, y):
        n = len(y)
        d = LassoNetClassifier._output_shape(y)
        y_bin = torch.full((n, d), False)
        y_bin[torch.arange(n), y] = True
        return LassoNetRegressor._lambda_max(X, y_bin)

    criterion = torch.nn.CrossEntropyLoss(reduction="mean")

    def predict(self, X):
        with torch.no_grad():
            ans = self.model(self._cast_input(X)).argmax(dim=1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def predict_proba(self, X):
        with torch.no_grad():
            ans = torch.softmax(self.model(self._cast_input(X)), -1)
        if isinstance(X, np.ndarray):
            ans = ans.cpu().numpy()
        return ans

    def validation_metrics(self, X_val, y_val):
        return ((None,))


def lassonet_path(X, y, task, **kwargs):
    """
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Training data
    y : array-like of shape (n_samples,) or (n_samples, n_outputs)
        Target values
    task : str, must be "classification" or "regression"
        Task
    hidden_dims : tuple of int, default=(100,)
        Shape of the hidden layers.
    eps_start : float, default=1
        Sets lambda_start such that it has a strength comparable to the
        loss of the unconstrained model multiplied by eps_start.
    lambda_start : float, default=None
        First value on the path.
    path_multiplier : float
        Multiplicative factor (:math:`1 + \\epsilon`) to increase
        the penalty parameter over the path
    M : float, default=10.0
        Hierarchy parameter.
    optim : torch optimizer or tuple of 2 optimizers, default=None
        Optimizer for initial training and path computation.
        Default is Adam(lr=1e-3), SGD(lr=1e-3, momentum=0.9).
    n_iters : int or pair of int, default=(1000, 100)
        Maximum number of training epochs for initial training and path computation.
        This is an upper-bound on the effective number of epochs, since the model
        uses early stopping.
    patience : int or pair of int, default=10
        Number of epochs to wait without improvement during early stopping.
    val_size : float, default=0.1
        Proportion of data to use for early stopping.
    device : torch device, default=None
        Device on which to train the model using PyTorch.
        Default: GPU if available else CPU
    verbose : int, default=0
    random_state
        Random state for cross-validation
    torch_seed
        Torch state for model random initialization
    """
    if task == "classification":
        model = LassoNetClassifier(**kwargs)
    elif task == "regression":
        model = LassoNetRegressor(**kwargs)
    else:
        raise ValueError('task must be "classification" or "regression"')
    return model.path(X, y)
