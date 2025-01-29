#
import numpy as np
import torch
import torch.nn as nn

import random
from imblearn.over_sampling import SMOTE


from scipy.optimize import fmin_l_bfgs_b
from torch.optim import Optimizer
from functools import reduce

# from sklearn.metrics import log_loss

# debug = False # True # 
# if debug:
#     import pdb

eps=np.finfo(float).eps

# from torch.utils.data import DataLoader, WeightedRandomSampler, SubsetRandomSampler, TensorDataset


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
    
    return torch.from_numpy(weights) # /np.sum(weights))


 
# Custom DataLoader function with modes
def create_dataloader(X, y, weights, batch_size, mode='random'):
    """
    Create a DataLoader with the specified batching mode.

    Parameters:
    - X: Features (torch.Tensor).
    - y: Labels (torch.Tensor).
    - weights: Sample weights (torch.Tensor).
    - batch_size: Number of samples per batch.
    - mode: Batch generation mode ('random', 'class_equitative', 'representative').

    Returns:
    - A DataLoader instance.
    """
    # Combine X, y, and weights into a single dataset
    dataset = torch.utils.data.TensorDataset(X, y, weights)

    if mode == 'random':
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    elif mode in ['class_equitative', 'representative']:
        y_int = y.to(torch.int64).numpy()
        class_counts = np.bincount(y_int)
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[y_int]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(y), replacement=True)
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    else:
        raise ValueError("Invalid mode. Choose 'random', 'class_equitative', or 'representative'.")
        
def train_model(x, y, model, loss_fn, optimizer, weights, num_epochs=10, batch_size=None, mode='random', lbfgs=False, debug=False):
    """
    Train a model using either gradient-based optimizers or LBFGS.
    
    Parameters:
    - x: Features (torch.Tensor).
    - y: Labels (torch.Tensor).
    - model: PyTorch model to train.
    - loss_fn: Loss function (callable).
    - optimizer: Optimizer instance (torch.optim or LBFGSScipy).
    - weights: Sample weights (torch.Tensor).
    - num_epochs: Maximum number of epochs (int).
    - batch_size: Batch size for DataLoader (int).
    - mode: Sampling mode for DataLoader ('random', 'class_equitative', etc.).
    - lbfgs: If True, use LBFGS optimization logic (bool).
    - debug: If True, print debug information.
    """
    # Default to full dataset if batch_size is not specified
    # debug = True
    
    if batch_size is None:
        batch_size = len(x)
    
    trainloader = create_dataloader(x, y, weights, batch_size=batch_size, mode=mode)

    for epoch in range(num_epochs):
        
        """
        if not lbfgs:
            # Optionally add a learning rate scheduler
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        """
        for batch_idx, (features, labels, batch_weights) in enumerate(trainloader):
            features = features.float()
            labels = labels.view(-1, 1)
            batch_weights = batch_weights.float()
            
            if lbfgs:
                # Define closure for LBFGS optimizer
                def closure():
                    optimizer.zero_grad()
                    outputs = model(features)
                    loss = loss_fn(outputs, labels, batch_weights)
                    loss.backward()
                    return loss
                
                # Perform optimization step
                optimizer.step(closure)
                
                # Optionally retrieve the loss value explicitly
                loss = closure()
            else:
                # For standard optimizers
                optimizer.zero_grad()
                outputs = model(features)
                loss = loss_fn(outputs, labels, batch_weights)
                loss.backward()
                optimizer.step()
    
            # Debug logging
            if debug and batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.5f}")
        """       
        if not lbfgs:
            # Step the scheduler after each epoch
            scheduler.step()
        """
    return model

def weighted_mse_loss(inputs, target, weights=None):
    if isinstance(target,np.ndarray):
        target=torch.from_numpy(target)
    if isinstance(inputs,np.ndarray):
        inputs=torch.from_numpy(inputs)
    if weights==None:
        weights = torch.ones_like(inputs)

    # Compute weighted MSE
    weighted_diff = weights * (inputs - target) ** 2
    return 0.5 * torch.sum(weighted_diff) #  / torch.sum(weights)
    
    
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
    def __init__(self, input_size, hidden_size, alpha, beta):  
        super().__init__(input_size, hidden_size, alpha, beta)  # Pass only necessary parameters
        self.hidden0 = nn.Linear(input_size, hidden_size)  # Use input size from the previous layer
        self.out = nn.Linear(hidden_size, 1)  # Directly define the output layer

        # Convert model parameters to float64 (double precision)
        self.hidden0 = self.hidden0.to(torch.float64)
        self.out = self.out.to(torch.float64)

    def forward(self, x):
        # Keep input data as float64
        x = x.to(torch.float64)
        o = torch.tanh(self.hidden0(x)) # torch.relu(self.hidden0(x)) # 
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
                 Q_RB_S=1, n_epoch=1, n_batch=1, lbfgs=True, mode='random'):
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
        self.n_batch = n_batch
        self.lbfgs = lbfgs
        self.mode = mode

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

    def fit(self, x_train, y_train, sample_weight=None):
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
        
        # Convert x_train and y_train to torch tensors if they are NumPy arrays
        if isinstance(x_train, np.ndarray):
            x_train = torch.from_numpy(x_train).float()
        if isinstance(y_train, np.ndarray):
            y_train = torch.from_numpy(y_train.astype(int))
        
        # Ensure the data is on the same device as the model
        device = next(self.parameters()).device
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        self.generate_experts_data(x_train, y_train, 
                                   RB_optimization=False,
                                   RB_each_expert=False,
                                   Q_RB_S=self.Q_RB_S)
        
        # Optional: Set sample weights to experts if provided
        if sample_weight is not None:
            if isinstance(sample_weight, np.ndarray):
                sample_weight = torch.from_numpy(sample_weight).float()
            sample_weight = sample_weight.to(device)
            self.fit_expert_model(sample_weight, epochs=self.n_epoch, batch_size=self.n_batch, lbfgs=self.lbfgs)
        else:
            # If no sample weights are provided, call fit_expert_model with default weights
            default_weights = torch.ones(y_train.shape[0]).float().to(device)
            self.fit_expert_model(default_weights, epochs=self.n_epoch, batch_size=self.n_batch, lbfgs=self.lbfgs)
        
        return self
                
    def fit_expert_model(self, w_train, epochs=50, batch_size=256, lbfgs=True):
        """
        Train experts using their stored data with either LBFGS or RMSprop optimization.
    
        Parameters:
        - w_train: Training weights (tensor).
        - epochs: Number of training epochs (int).
        - batch_size: Batch size for training (int).
        - lbfgs: Whether to use the LBFGS optimizer (bool).
    
        Returns:
        - self: The updated model after training.
        """
        # Initialize weights for each expert
        weights = torch.ones((self.experts[0].y.shape[0], self.num_experts)).to(w_train.device)
    
        # Compute weights for each expert
        for i, expert in enumerate(self.experts):
            weights[:, i] = compute_weights(expert.y, RB=self.Q_RB_C, IR=1, mode='Normal') * w_train  # Scale by training weights
    
        # Training loop for experts
        if lbfgs:
            # Use LBFGS optimizer for each expert
            for i, expert in enumerate(self.experts):
                # Configure the LBFGS optimizer
                optim_LBFGS_scipy = LBFGSScipy(
                    expert.parameters(),
                    max_iter=150,
                    max_eval=150,
                    tolerance_grad=1e-04,
                    tolerance_change=10e6 * eps,  # `eps` is assumed to be defined outside
                    history_size=10
                )
    
                # Train the model using LBFGS and the integrated data loader functionality
                train_model(
                    expert.X,  # Input data for this expert
                    expert.y,  # Labels for this expert
                    expert,  # Model to train (expert)
                    self.loss_fn_e,  # Loss function
                    optim_LBFGS_scipy,  # Optimizer (LBFGS)
                    weights[:, i],  # Sample weights for this expert
                    num_epochs=epochs,  # Number of epochs
                    batch_size=None, # batch_size,  # Batch size
                    mode=self.mode, #'representative', # 'class_equitative',  # Mode for DataLoader
                    lbfgs = lbfgs, 
                    debug=False  # Enable debug information if necessary
                )
        else:
            # Main training loop with gradient-based optimizers (e.g., RMSprop)
            for i, expert in enumerate(self.experts):
                # Define gradient-based optimizer (RMSprop in this case)
                optim_RMS = torch.optim.RMSprop(expert.parameters(), lr=0.001)
                # Define a more powerful optimizer (Adam)
                # optim_Adam = torch.optim.Adam(expert.parameters(), lr=0.001)
                # AdamW improves on Adam by incorporating weight decay (useful for regularization).
                # optim_AdamW = torch.optim.AdamW(expert.parameters(), lr=0.01, weight_decay=1e-2)
                # SGD with Momentum
                # optim_SGD = torch.optim.SGD(expert.parameters(), lr=0.1, momentum=0.9)
                # Adagrad adapts the learning rate based on the frequency of updates.
                # optim_Adagrad = torch.optim.Adagrad(expert.parameters(), lr=0.01)
                # Adadelta is an improvement over Adagrad, addressing its decaying learning rate problem.
                # optim_Adadelta = torch.optim.Adadelta(expert.parameters(), lr=0.01)
                
                # Train the expert using gradient-based optimizer
                train_model(  # Use the unified training function
                    expert.X,  # Input data for this expert
                    expert.y,  # Labels for this expert
                    expert,  # Model to train (expert)
                    self.loss_fn_e,  # Loss function
                    optim_RMS,  # Optimizer (RMSprop)
                    # optim_Adam, # Optimizer (Adam)
                    # optim_AdamW, # Optimizer (AdamW)
                    # optim_SGD, # SGD with Momentum
                    # optim_Adagrad, # Adagrad
                    # optim_Adadelta, # Adadelta
                    weights[:, i],  # Sample weights for this expert
                    num_epochs=epochs,  # Number of epochs
                    batch_size=batch_size,  # Batch size
                    mode= self.mode, # 'class_equitative',  # 'random', #'class_equitative',  #'representative', # Choose appropriate mode
                    lbfgs = lbfgs, 
                    debug=False  # Enable debug information if necessary
                )
    
        return self
    
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
        
        return o_pred.numpy()  # Forward method now gives output in the 0 to 1 range
    
    
    def predict(self, x):
        """
        Final prediction averaging over experts and applying threshold to obtain class labels.
        """
        QR_tr = 2*0.5 * self.QP_tr * (self.Q_RB_C + self.Q_RB_S) / (self.Q_RB_C * self.Q_RB_S)
        # QR_tr -= 1
        Q_tr = self.QC * self.QP_tr
    
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
    
        # Apply thresholding to get the final class labels
        y = np.ones_like(o_pred)
        eta_th = (2 * (self.alpha + (1 - self.alpha - self.beta) * (Q_tr / (Q_tr + QR_tr))) - 1)
        y[o_pred < eta_th] = self.bin_format
        
        return y.astype(int)
    
    def predict_proba(self, x):
        """
        Returns the predicted probability (o_pred) without applying threshold logic.
        This method is equivalent to predict but returns o_pred instead of y.
        """
        # Get the averaged expert predictions (o_pred)
        o_pred = self.forward(x)
        
        # Normalize to be between 0 and 1 (for proba outputs)
        o_pred_01 = (o_pred + 1) / 2
        
        return o_pred_01.numpy().astype(int)  # Convert back to numpy if necessary

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