#
import numpy as np
import torch
import torch.nn as nn

import random
from imblearn.over_sampling import SMOTE


from scipy.optimize import fmin_l_bfgs_b
from torch.optim import Optimizer
from functools import reduce

from sklearn.metrics import log_loss

debug = False # True # 
if debug:
    import pdb

eps=np.finfo(float).eps

class LBFGSScipy(Optimizer):
    """Wrap L-BFGS algorithm, using scipy routines.
    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).
    .. warning::
        Right now CPU only
    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.
    Arguments:
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-5).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
    """

    def __init__(self, params, max_iter=20, max_eval=None,
                 tolerance_grad=1e-5, tolerance_change=1e-9, history_size=10,
                 ):
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        defaults = dict(max_iter=max_iter, max_eval=max_eval,
                        tolerance_grad=tolerance_grad, tolerance_change=tolerance_change,
                        history_size=history_size)
        super(LBFGSScipy, self).__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("LBFGS doesn't support per-parameter options "
                             "(parameter groups)")

        self._params = self.param_groups[0]['params']
        self._numel_cache = None

        self._n_iter = 0
        self._last_loss = None

    def _numel(self):
        if self._numel_cache is None:
            self._numel_cache = reduce(lambda total, p: total + p.numel(), self._params, 0)
        return self._numel_cache

    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                view = p.data.new(p.data.numel()).zero_()
            elif p.grad.data.is_sparse:
                view = p.grad.data.to_dense().view(-1)
            else:
                view = p.grad.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _gather_flat_params(self):
        views = []
        for p in self._params:
            if p.data.is_sparse:
                view = p.data.to_dense().view(-1)
            else:
                view = p.data.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def _distribute_flat_params(self, params):
        offset = 0
        for p in self._params:
            numel = p.numel()
            # view as to avoid deprecated pointwise semantics
            p.data = params[offset:offset + numel].view_as(p.data)
            offset += numel
        assert offset == self._numel()

    def step(self, closure):
        """Performs a single optimization step.
        Arguments:
            closure (callable): A closure that reevaluates the model
                and returns the loss.
        """
        assert len(self.param_groups) == 1

        group = self.param_groups[0]
        max_iter = group['max_iter']
        max_eval = group['max_eval']
        tolerance_grad = group['tolerance_grad']
        tolerance_change = group['tolerance_change']
        history_size = group['history_size']

        def wrapped_closure(flat_params):
            """closure must call zero_grad() and backward()"""
            flat_params = torch.from_numpy(flat_params)
            self._distribute_flat_params(flat_params)
            loss = closure()
            self._last_loss = loss
            loss = loss.data
            flat_grad = self._gather_flat_grad().numpy()
            return loss, flat_grad

        def callback(flat_params):
            self._n_iter += 1

        initial_params = self._gather_flat_params()

        
        fmin_l_bfgs_b(wrapped_closure, initial_params, maxiter=max_iter,
                      maxfun=max_eval,
                      factr=tolerance_change / eps, pgtol=tolerance_grad, epsilon=1e-08,
                      m=history_size,
                      callback=callback)

def label_switching(y, alphasw=0.0, betasw=0.0):

    # alphasw: Label Switching Rate from Majority to Minority class
    # betasw: Label Switching Rate from Minority to Majority class

    ysw=1*y;

    idx1=np.where(y==+1)[0]
    l1=len(idx1)
    bet_1=int(round(l1*betasw))
    idx1_sw=np.random.choice(idx1,bet_1, replace=False)
    ysw[idx1_sw]=-y[idx1_sw]

    idx0=np.where(y==-1)[0]
    l0=len(idx0)
    alph_0=int(round(l0*alphasw))
    idx0_sw=np.random.choice(idx0,alph_0, replace=False)
    ysw[idx0_sw]=-y[idx0_sw]

    return ysw

def compute_weights(targets_train, RB = 1, IR = 1, mode = 'Normal'):
    # RB define la cantidad de reequilibrado final
    # Si RB = IR => No se reequilibra. Si RB = 1, es un reequilibrado full.
    
    weights = np.ones_like(targets_train) # .astype('float')
    if mode == 'Small':
        weights[np.where(targets_train<=0)[0]] = RB/IR
    else:
        weights[np.where(targets_train>0)[0]] = RB # IR/RB
    
    return torch.from_numpy(weights)


def train_LBFGS_scipy(x, y, model, loss_fn, optim, weights, num_epochs=10):
    
    dataset = torch.utils.data.TensorDataset(x, y)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset),
                                              shuffle=False, num_workers=0)

    """
    final_loss = None
    prev_loss = None  # To store the loss from the previous epoch
    """
    
    for epoch in range(num_epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            def closure():
                optim.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels.view(len(dataset), 1), weights.view(len(dataset), 1))
                loss.backward()
                return loss

            optim.step(closure)

            """
            # Get the final loss from the closure
            final_loss = closure().item()

            if prev_loss is not None:
                # Calculate the loss change
                loss_change = abs(final_loss - prev_loss)
                
                # Check if the loss change is below the tolerance
                if loss_change < optim.param_groups[0]['tolerance_change']:
                    if debug:
                        print(f"Stopping early at epoch {epoch + 1} due to minimal loss improvement.")
                    return model

            prev_loss = final_loss  # Update previous loss for next comparison

            if debug:
                print(f"Epoch {epoch + 1}, Training Loss: {final_loss:.4f}")
            """
    return model


def train_grad(trainloader, y_train, model, loss_fn, optimizer, weights, input_size, epochs, expert_idx):
    """
    Train the model using gradient-based optimization.
    """
    for epoch in range(epochs):
        for batch_idx, (X_batch, y_batch) in enumerate(trainloader):
            # Ensure X_batch uses the correct data type (float32)
            X_batch = X_batch[:, :input_size].float()  # Convert to float32
            y_batch = y_batch.view(-1, 1).float()      # Convert to float32
            
            # Move data to the correct device
            device = next(model.parameters()).device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            batch_weights = weights[batch_idx * len(X_batch):(batch_idx + 1) * len(X_batch)].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch, batch_weights)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
    #print("Training Results - loss: {:.2f}, aux_loss: {:.3f}".format(loss.item(), aux_loss.item()))
    return model


def weighted_mse_loss(inputs, target, weights=None):
    if isinstance(target,np.ndarray):
        target=torch.from_numpy(target)
    if isinstance(inputs,np.ndarray):
        inputs=torch.from_numpy(inputs)
    if weights==None:
        weights = torch.ones_like(inputs)
    return 0.5 * torch.sum((weights*((inputs - target)) ** 2)) # , torch.sum(weights))
    
    
def weighted_bce_loss(inputs, target, weights=None):
    loss_bce = nn.BCELoss(weight=weights)
    # Convert inputs and targets to float to avoid the dtype mismatch
    inputs_01 = torch.abs(0.5 * (inputs + 1)).float()  # Ensure float type
    targets_01 = (0.5 * (target + 1)).float()          # Ensure float type

    return loss_bce(inputs_01, targets_01)


def weighted_bce_logit_loss(inputs, target, weights):
    loss_bce = nn.BCEWithLogitsLoss(weight=weights)

    return loss_bce(inputs, target)

def f1_loss(predict, target, weights=None):
    
    if isinstance(target,np.ndarray):
        target=torch.from_numpy(target)
    target = 0.5*(target+1)
    if isinstance(predict,np.ndarray):
        predict=torch.from_numpy(predict)
    predict = torch.clip(0.5*(predict+1),0,1) 
    
    loss = 0
    lack_cls = target.sum(dim=0) == 0
    if lack_cls.any():
        loss += nn.BCEWithLogitsLoss(
            predict[:, lack_cls], target[:, lack_cls], weight=weights)

    tp = predict * target
    tp = tp.sum(dim=0)
    
    fp = predict * (1 - target)
    fp = fp.sum(dim=0)
    
    fn = ((1 - predict) * target)
    fn = fn.sum(dim=0)
    
    tn = (1-predict)*(1-target)
    tn = tn.sum(dim=0)
    
    soft_f1_class1 = 2*tp / (2*tp + fn + fp + 1e-8)
    soft_f1_class0 = 2*tn / (2*tn + fn + fp + 1e-8)
    cost_class1 = 1 - soft_f1_class1 # reduce 1 - soft-f1_class1 in order to increase soft-f1 on class 1
    cost_class0 = 1 - soft_f1_class0 # reduce 1 - soft-f1_class0 in order to increase soft-f1 on class 0
    cost = 0.5 * (cost_class1 + cost_class0) # take into account both class 1 and class 0
    macro_cost = cost.mean() # average on all labels
    
    return (macro_cost + loss)


# Base class to handle common initialization logic
class BaseAsymmetricMLP(nn.Module):
    def __init__(self, input_size, hidden_size, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.init_weights()

    def init_weights(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(weight_init)
                

class AsymmetricMLP1(BaseAsymmetricMLP):
    def __init__(self, input_size, hidden_size, alpha, beta):  # Only keep hidden_size, alpha, beta
        super().__init__(input_size, hidden_size, alpha, beta)  # Pass only necessary parameters
        self.hidden0 = nn.Linear(input_size, hidden_size)  # Use input size from the previous layer
        self.out = nn.Linear(hidden_size, 1)  # Directly define the output layer
        
        # Convert model parameters to float64 (double precision)
        self.hidden0 = self.hidden0.to(torch.float64)
        self.out = self.out.to(torch.float64)

    def forward(self, x):
        # Keep input data as float64
        x = x.to(torch.float64)
        o = torch.tanh(self.hidden0(x))
        z = self.out(o)
        return torch.where(z < 0, torch.tanh(z) * (1 - 2 * self.alpha), torch.tanh(z) * (1 - 2 * self.beta))


class AsymmetricMLP2(BaseAsymmetricMLP):
    def __init__(self, input_size, hidden_size, alpha, beta):  # Only keep hidden_size, alpha, beta
        super().__init__(input_size, hidden_size, alpha, beta)  # Pass only necessary parameters
        self.hidden0 = nn.Linear(input_size, hidden_size)  # Use input size from the previous layer
        self.out = nn.Linear(hidden_size, 1)  # Directly define the output layer
        
        # Convert model parameters to float64 (double precision)
        self.hidden0 = self.hidden0.to(torch.float64)
        self.out = self.out.to(torch.float64)

    def forward(self, x):
        o = torch.tanh(self.hidden0(x))
        z = self.out(o)
        return torch.where(z < 0, torch.tanh(z / (1 - 2 * self.alpha)) * (1 - 2 * self.alpha),
                                 torch.tanh(z / (1 - 2 * self.beta)) * (1 - 2 * self.beta))
    
class LSEnsemble(nn.Module):
    def __init__(self, hidden_size, num_experts, alpha=0, beta=0, Q_RB_C=1, 
                 Q_RB_S=1, n_epoch=1):
        super(LSEnsemble, self).__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.alpha = alpha
        self.beta = beta

        # Initialize the list to hold experts (will be populated later)
        self.experts = nn.ModuleList()
        self.loss_fn_e = weighted_mse_loss # weighted_bce_loss # 
      
        self.Q_RB_C = Q_RB_C
        self.Q_RB_S = Q_RB_S
        
        C10=1 # False Positive (FP)
        C00=0 # True Negative (TN)
        C01=1 # False Negative (FN)   : QC=1/5
        C11=0 # True Positive (TP)

        self.QC=float(C10-C00)/float(C01-C11)
        
        self.n_epoch = n_epoch

    def initialize_experts(self, input_size):
        # Create experts with the actual input size
        self.experts = nn.ModuleList([AsymmetricMLP1(input_size, self.hidden_size, self.alpha, self.beta) 
                                      for _ in range(self.num_experts)])

    # This function generates distinct data for each expert using SMOTE
    def generate_experts_data(self, x, y, RB_optimization=True, RB_each_expert = True, Q_RB_S=0):
        x_np = x.cpu().numpy()
        y_np = y.cpu().numpy()
        
        # Check if y_np contains only 0 and 1
        if np.all(np.isin(y_np, [0, 1])):
            self.bin_format = 0
            # Convert binary labels from 0 and 1 to +1 and -1
            y_np = np.where(y_np == 0, -1, 1)
        elif np.all(np.isin(y_np, [-1, 1])):
            self.bin_format = -1
        
        N0_tr = len(np.where(y_np==-1)[0])
        N1_tr = len(np.where(y_np==1)[0])
        
        P0_tr = float(N0_tr)/float(N0_tr+N1_tr)
        P1_tr = float(N1_tr)/float(N0_tr+N1_tr)
        
        QP_tr = P0_tr/P1_tr
        self.QP_tr = QP_tr
    
        if RB_each_expert and RB_optimization: 
            for expert in self.experts:
                if RB_optimization and Q_RB_S != 0:
                    Q_RB_S = max(Q_RB_S, 1)  # Ensure Q_RB is at least 1
                    self.QP_tr = QP_tr / Q_RB_S
                    try:
                        smote = SMOTE(random_state=random.randint(1, 100), sampling_strategy=Q_RB_S/self.QP_tr)
                        X_RB, y_RB = smote.fit_resample(x_np, y_np)
                        ind_ord = np.arange(len(y_RB))
                        ind_rnd = np.random.choice(ind_ord, size=len(y_RB), replace=False)
                        X_RB, y_RB = X_RB[ind_rnd,:], y_RB[ind_rnd]
                    except ValueError as e:
                        # print(f"SMOTE error: {e}, using original data.")
                        X_RB, y_RB = x_np, y_np  # Fallback to original data
                else:
                    X_RB, y_RB = x_np, y_np
                    self.Q_RB_S = self.QP_tr
                
                # Label Switching
                if self.alpha == 0 and self.beta == 0:
                    y_RB_SW = torch.from_numpy(y_RB)
                else:
                    targets_sw = torch.from_numpy(label_switching(y_RB, self.alpha, self.beta))
                    beta_sw = (1 - 2 * self.beta) * torch.ones_like(targets_sw).float()
                    alpha_sw = -(1 - 2 * self.alpha) * torch.ones_like(targets_sw).float()
                    y_RB_SW = torch.where(targets_sw > 0, beta_sw, alpha_sw)
                
                X_RB_SW = X_RB
        
                # Convert to torch tensors and store them in each expert's properties
                expert.X = torch.from_numpy(X_RB_SW).float().to(x.device)
                expert.y = y_RB_SW.to(y.device)
        else:
            if RB_optimization and Q_RB_S != 0:
                Q_RB_S = max(Q_RB_S, 1)  # Ensure Q_RB is at least 1
                self.QP_tr = Q_RB_S / Q_RB_S
                try:
                    smote = SMOTE(random_state=random.randint(1, 100), sampling_strategy=Q_RB_S/self.QP_tr)
                    X_RB, y_RB = smote.fit_resample(x_np, y_np)
                except ValueError as e:
                    # print(f"SMOTE error: {e}, using original data.")
                    X_RB, y_RB = x_np, y_np  # Fallback to original data
            else:
                X_RB, y_RB = x_np, y_np
                self.Q_RB_S = self.QP_tr
                
            for expert in self.experts:
                # Label Switching
                if self.alpha == 0 and self.beta == 0:
                    y_RB_SW = torch.from_numpy(y_RB)
                else:
                    targets_sw = torch.from_numpy(label_switching(y_RB, self.alpha, self.beta))
                    beta_sw = (1 - 2 * self.beta) * torch.ones_like(targets_sw).float()
                    alpha_sw = -(1 - 2 * self.alpha) * torch.ones_like(targets_sw).float()
                    y_RB_SW = torch.where(targets_sw > 0, beta_sw, alpha_sw)
                X_RB_SW = X_RB
        
                # Convert to torch tensors and store them in each expert's properties
                expert.X = torch.from_numpy(X_RB_SW).float().to(x.device)
                expert.y = y_RB_SW.to(y.device)

    def fit(self, x_train, y_train, sample_weight=None, epochs=50, batch_size=256, lbfgs=False):
        """
        Fit the ensemble model using training data.
    
        Parameters:
        - x_train: Tensor of shape (n_samples, n_features) for training data.
        - y_train: Tensor of shape (n_samples,) for training labels.
        - sample_weight: Optional tensor of shape (n_samples,) for sample weights.
        - epochs: Number of training epochs.
        - batch_size: Size of batches for training.
        - lbfgs: Boolean indicating whether to use LBFGS optimizer.
        """
        # Initialize input_size
        self.input_size = x_train.shape[1]  # Assuming x_train is a 2D tensor
        self.initialize_experts(self.input_size)
        
        X_train_torch = torch.from_numpy(x_train)
        y_train_torch = torch.from_numpy(y_train.astype(int))
        
        self.generate_experts_data(X_train_torch, y_train_torch, 
                                   RB_optimization=False,
                                   RB_each_expert=False,
                                   Q_RB_S=self.Q_RB_S)
        
    
        # Optional: Set sample weights to experts if provided
        if sample_weight is not None:
            sample_weight_torch = torch.from_numpy(sample_weight)
            self.fit_expert_model(sample_weight_torch, epochs=self.n_epoch, batch_size=batch_size, lbfgs=lbfgs)
        else:
            # If no sample weights are provided, call fit_expert_model with default weights
            # default_weights = torch.ones(y_train.shape[0]).float().to(x_train.device)
            default_weights = torch.ones(y_train.shape[0]).to(X_train_torch.device)
            self.fit_expert_model(default_weights, epochs=self.n_epoch, batch_size=batch_size, lbfgs=lbfgs)
            
    # Train experts using their stored data
    def fit_expert_model(self, w_train, epochs=50, batch_size=50, lbfgs=True):
        # weights = torch.ones((self.experts[0].y.shape[0], self.num_experts)).float().to(w_train.device)
        weights = torch.ones((self.experts[0].y.shape[0], self.num_experts)).to(w_train.device)
        
        # Compute weights for each expert
        for i, expert in enumerate(self.experts):
            weights[:, i] = compute_weights(expert.y, RB=self.Q_RB_C, IR=1, mode='Normal') * w_train # Scale weights by training weights

        # Training loop for experts
        if lbfgs:
            for i, expert in enumerate(self.experts):
                optim_LBFGS_scipy = LBFGSScipy(expert.parameters(), max_iter=150, max_eval=150,
                                               tolerance_grad=1e-04, tolerance_change=10e6*eps, history_size=10)
                train_LBFGS_scipy(expert.X, expert.y, expert, self.loss_fn_e, optim_LBFGS_scipy, weights[:, i], epochs)
        else:
            for i, expert in enumerate(self.experts):
                dataset = torch.utils.data.TensorDataset(expert.X, expert.y)
                trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
                
                optim_RMS = torch.optim.RMSprop(expert.parameters(), lr=0.0001)
                train_grad(trainloader, expert.y, expert, self.loss_fn_e, optim_RMS, weights[:, i], self.input_size, epochs, i)

        # return self
    
    # Get outputs from all experts
    def get_expert_outputs(self, x):
        expert_outputs = self.experts[0](x)
        aux_outputs = torch.zeros_like(expert_outputs)
        for i in range(self.num_experts-1):
            aux_outputs = self.experts[i+1](x)
            expert_outputs = torch.cat((expert_outputs, aux_outputs), 1)
        return expert_outputs

    # Predict outputs of each expert
    def predict_expert_outputs(self, x):
        with torch.no_grad():
            return self.get_expert_outputs(x) # .double())

    def forward(self, x):
        """
        This method performs the forward pass by calculating the predictions from the experts
        and returning the averaged prediction (o_pred) across experts.
        """
    
        # Convert input to tensor
        x_torch = torch.from_numpy(x).float()
    
        # Get expert outputs and compute their average
        expert_outputs = self.predict_expert_outputs(x_torch)
        o_pred = expert_outputs.mean(dim=1)  # Average over experts
    
        return o_pred
    
    def predict(self, x):
        """
        Final prediction averaging over experts and applying threshold to obtain class labels.
        """
        QR_tr = self.QP_tr * (self.Q_RB_C + self.Q_RB_S) / (self.Q_RB_C * self.Q_RB_S)
        Q_tr = self.QC * self.QP_tr
    
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
    
        # Apply thresholding to get the final class labels
        y = np.ones_like(o_pred)
        y[o_pred < (2 * (self.alpha + (1 - self.alpha - self.beta) * (Q_tr / (Q_tr + QR_tr))) - 1)] = self.bin_format
        
        return y
    
    def predict_proba(self, x):
        """
        Returns the predicted probability (o_pred) without applying threshold logic.
        This method is equivalent to predict but returns o_pred instead of y.
        """
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
        
        return o_pred
    # Method for GridSearchCV compatibility
    def get_params(self, deep=True):
        return {
            'hidden_size': self.hidden_size,
            'num_experts': self.num_experts,
            'alpha': self.alpha,
            'beta': self.beta,
            'Q_RB_C': self.Q_RB_C,
            'Q_RB_S': self.Q_RB_S,
            'n_epoch': self.n_epoch
        }

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self