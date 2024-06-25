import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib as mplt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
from sys import path

def Normalization(x):  
    '''
    The input is original data and the output is normalized values. 
    Both input and output are tensors.
    '''
    Num = x.shape[0]
    sym_num = Num
    mean_power = torch.sum(torch.mul(x, x)) / sym_num
    p = mean_power ** 0.5
    x = x / p          
    return x

def mkdir(path):
    '''
    Determine if a folder exists. If it does not exist, create it as a folder.
    '''
    folder = os.path.exists(path)
    if not folder:                  
        os.makedirs(path)            

def awgn(x, snr):
    '''
    Gaussian additive white noise channel simulation. The input is the original data x 
    and snr, and the output is the value through the channel, both of which are tensors.
    '''
    Num = x.shape[0]
    snr_db = snr                         # dB
    snr_w  = 10 ** (snr_db / 10)         # W
    n_power = 1 / snr_w
    # gaussian noise
    noise = ((n_power / 2) ** 0.5) * torch.randn(x.shape, device = device) 
    channel_signal = x + noise.detach()
    return channel_signal

def norm1d(x, sig_p = 1, mean = 0):
    '''
    Realize the normalization function of one-dimensional data
    Parameters:
    x: array_like
    Input one-dimensional data array
    sig_p: float, optional
    Normalization data power value, default to 1
    mean: bool, optional
    Represents whether to do the 0-means normalization

    Returns:
    y: array_like
    Output one-dimensional normalized data array
    Raises:
    AttributeError
    When the input data format is not numpy or tensor, show that 'Data type is not supported'
    '''
    sym_num = x.shape[0]
    if type(x) == torch.Tensor:
        if mean:
            m = x.mean()
            y = x - m
        else:
            y = x
        mean_power = torch.sum((y.abs()**2)) / sym_num  
        scale = (mean_power / sig_p) ** 0.5
    elif type(x) == np.ndarray:
        if mean:
            m = np.mean(x)
            y = x - m
        else:
            y = x
        mean_power = np.sum((np.abs(y)**2)) / sym_num  
        scale = (mean_power / sig_p) ** 0.5
    else:
        raise AttributeError('Data type is not supported')
    return y / scale

class MINE(nn.Module):
    '''
    The structure of proposed enhanced mutual information neural estimator.
    '''
    def __init__(self, carry_rate=0.99):
        super(MINE, self).__init__()
        self.forward_pass = nn.Sequential(
            nn.Linear(4, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Linear(128, 1))
        self.carry_rate = carry_rate
        self.ema = None
    
    def joint_matrix(self,  x, y):
        joint = torch.cat((x,y), dim=-1)
        return self.forward_pass(joint).mean()

    def marginal_matrix(self, x, y):
        x_tiled = torch.stack([x] * batch_size_m, dim=0)
        y_tiled = torch.stack([y] * batch_size_m, dim=1)
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [batch_size_m * batch_size_m, -1])
        output = self.forward_pass(xy_pairs)
        output = torch.reshape(output, [batch_size_m, batch_size_m]).t()
        infs = torch.tensor([float('inf')] * batch_size_m).to(output.device)
        marginal = torch.logsumexp(output - infs.diag(), dim = (0, 1)) \
            - torch.log(torch.tensor(batch_size_m * (batch_size_m -1 ))).to(device) 
        
        if self.ema:
            exp_marginal = (output - infs.diag()).exp().sum() / batch_size_m / (batch_size_m - 1)
            self.ema = exp_marginal.detach() if self.ema is None else \
                    self.carry_rate * self.ema + \
                    (1 - self.carry_rate) * exp_marginal.detach()
        return marginal

    def joint_vetor(self, x, y):
        joint_pair = torch.cat((x,y), dim=-1)
        return self.forward_pass(joint_pair).mean()

    def marginal_vetor(self, x, y):
        marginal_pair = torch.cat((x,y[np.random.permutation(batch_size)]), dim=-1)

        output = self.forward_pass(marginal_pair)
        exp_marginal = output.exp().mean()
        marginal = torch.log(exp_marginal)
        if self.ema:
            self.ema = exp_marginal.detach() if self.ema is None else \
                    self.carry_rate * self.ema + \
                    (1 - self.carry_rate) * exp_marginal.detach()
        return marginal
        
    def forward(self, x, y):
        if ehanced:
            joint = self.joint_matrix(x, y)
            marginal = self.marginal_matrix(x[0:batch_size_m], y[0:batch_size_m])
        else:
            joint = self.joint_vetor(x, y)
            marginal = self.marginal_vetor(x, y)
        
        dv_loss = joint - marginal
        if ema:
            if self.ema is None:
                self.ema = dv_loss.detach()
            else:
                self.ema = self.carry_rate * self.ema + (1 - self.carry_rate) * dv_loss.detach()
            mine_loss = (1 / self.ema) * dv_loss - dv_loss
        else:
            mine_loss = - dv_loss
        return mine_loss, dv_loss.detach()

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

def train(epoch_num, snr, true_mi):   
    result_path = path + str(epoch_num) + '_' + str(snr) + 'dB/'
    mkdir(result_path)
    model = MINE().to(device)
    model.apply(weight_init)
    optim_mine = optim.Adam(model.parameters(),lr = lr_mine)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim_mine, epoch_num)
    loss_list = []
    mi_list = []
    epoch_list = []
    loss_temp = 0
    mi_temp = 0
    cycle_idx = 0
    for i in range(epoch_num):
        model.train()
        for j in range(batch_num):
            cycle_idx += 1
            optim_mine.zero_grad()
            tx_sym_int = torch.randint(0, qamnum, (batch_size,))
            tx_sym = qammap[tx_sym_int]
            rx_sym = awgn(tx_sym, snr)
            loss, mi_estimation = model(tx_sym, rx_sym)
            loss.backward()
            optim_mine.step()        
            loss_temp += - loss.item() / np.log(2)
            mi_temp += mi_estimation.item() / np.log(2)
        scheduler.step()
        
        if((i + 1) % print_epoch == 0):
            loss_temp = loss_temp / cycle_idx
            mi_temp = mi_temp / cycle_idx
            
            loss_list.append(loss_temp)
            mi_list.append(mi_temp)
            epoch_list.append((i+1))
            loss_temp = 0
            mi_temp = 0
            cycle_idx = 0

    loss_list = np.array(loss_list)
    mi_list = np.array(mi_list)
    epoch_list = np.array(epoch_list)

    plt.figure()
    plt.plot(epoch_list, mi_list)
    plt.plot(epoch_list, true_mi * np.ones_like(epoch_list))
    plt.savefig(result_path + 'loss.png') 
    plt.close('all')
    np.savez(result_path + 'loss', loss_list)
    np.savez(result_path + 'mi', mi_list)
    np.savez(result_path + 'epoch', epoch_list)
    
    # return loss_list, mi_list, epoch_list
    loss_temp = 0
    mi_temp = 0
    cycle_idx = 0
    with torch.no_grad():
        for i in range(10):
            model.eval()
            for j in range(batch_num):
                cycle_idx += 1
                tx_sym_int = torch.randint(0, qamnum, (batch_size,))
                tx_sym = qammap[tx_sym_int]
                rx_sym = awgn(tx_sym, snr)
                loss, mi_estimation = model(tx_sym, rx_sym)
                loss_temp += - loss.item() / np.log(2)
                mi_temp += mi_estimation.item() / np.log(2)
        loss_temp = loss_temp / cycle_idx
        mi_temp = mi_temp / cycle_idx
    np.save(result_path + 'loss_test.npz', np.array([loss_temp]))
    np.save(result_path + 'mi_test.npz', np.array([mi_temp]))    
    print('Epoch: ' + str(epoch_num) + '_SNR:' + str(snr) + " MI: " + str(mi_temp) +\
                    " Real MI: " + str(true_mi))
    return mi_temp

def mi_gh(constell, snr, prob = None):
    """
    Evaluation of mutual information using Gauss-Hermite quadrature
    This function evaluates the mutual information (MI) and the generalized
    mutual information (GMI) of the 2D (complex) constellation C over an
    AWGN channel with standard deviation of the complex-valued noise equal
    to sigma_n. Evaluation is performed using the Gauss-Hermite quadrature
    which allows fast numerical integration of the mutual information. For
    GMI calculation the bit mapping B is used.
    
    taken from A. Alvarado et al., "Achievable Information Rates
    for Fiber Optics: Applications and Computations", J. Lightw. Technol. 36(2) pp. 424-439
    https://dx.doi.org/10.1109/JLT.2017.2786351
    Parameters of Gauss-Hermite taken from: http://keisan.casio.com/exec/system/1281195844
    
    Parameters:
        constell   :=      Constellation [M x 1]
        bit_maping :=      Bit mapping [M x log2(M)]
        snr        :=      Signal-to-noise ratios (Es/No, dB unit) [N x 1]
        prob       :=      Probability of each constellation point (optional) [M x 1]
    
    Returns:
        MI      :=      Mutual information in bits [N x 1]
        GMI     :=      Generalized mutual information in bits [N x 1]

    February 2018 - Dario Pilori
    """
    M = constell.shape[0]
    constell = constell.reshape(-1,1)
    if prob == None:
        prob = np.ones((M, 1)) / M     # Uninform distribution
    # Calculate sigma_n
    sigma_n = np.sqrt(np.sum(prob * np.abs(constell)**2)) * 10 ** (- snr / 20)
    # Params for Gauss-Hermite quadrature
    x = np.array([
        -3.436159118837737603327,  
        -2.532731674232789796409,
        -1.756683649299881773451,
        -1.036610829789513654178,
        -0.3429013272237046087892,	
        0.3429013272237046087892,	
        1.036610829789513654178, 
        1.756683649299881773451,
        2.532731674232789796409,
        3.436159118837737603327]).reshape(-1,1)
    w = np.array([
        7.64043285523262062916*(10**-6),
        0.001343645746781232692202,
        0.0338743944554810631362,
        0.2401386110823146864165,
        0.6108626337353257987836,
        0.6108626337353257987836,
        0.2401386110823146864165,
        0.03387439445548106313617,
        0.001343645746781232692202,
        7.64043285523262062916*(10**-6)]).reshape(-1,1)    

    # Evaluate Mutual Information
    MI = 0.0
    for l in range(M):
        for m in range(x.shape[0]):
            MI = MI - prob[l] / np.pi * w[m] * np.sum(\
                w.reshape(-1) * np.log2(np.sum(np.transpose(prob) * np.exp(\
                - (np.abs(constell[l] - np.transpose(constell)) ** 2 - \
                    2 * sigma_n * np.real((x + 1j * x[m]) * (constell[l] - np.transpose(constell))))\
                /sigma_n ** 2), axis = 1)))

    return MI[0]

def plot_mi(mi_results, mask):
    mi_results = np.abs(mi_results - mi_true.reshape((-1, 1)))

    vmin = 0
    vmax = 0.01
    norm = mplt.colors.Normalize(vmin=vmin, vmax=vmax)

    sns.set_theme(style="white")
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2,\
        hspace = 0.2, wspace = 0.2, \
        width_ratios=[1, 0.1], height_ratios = [1, 0.1])

    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    
    sns.heatmap(mi_results, cmap=cmap, square=True, \
        cbar = True, linewidths=.5, ax = ax0, \
            xticklabels = epoch, yticklabels = snr, \
                annot=False, vmin=vmin, vmax=vmax, mask = mask,\
                        cbar_ax=ax2, \
                            cbar_kws={"orientation": "horizontal"})
    ax0.set_title(r'|$\Delta$ MI|')
    ax0.set_xlabel('Epoch Num')
    ax0.set_ylabel('SNR Value')
    
    cmap_mi = sns.cubehelix_palette(as_cmap=True)
    sns.heatmap(mi_true.reshape((-1, 1)), cmap=cmap_mi, \
        cbar = False, linewidths=.5, ax = ax1, \
            xticklabels = False, yticklabels = False,\
                annot=True, fmt=".2f")
    ax1.set_title('True MI')

    plt.colorbar(mplt.cm.ScalarMappable(norm = norm, cmap=cmap), cax=ax2, \
    orientation = 'horizontal', label = r'|$\Delta$ MI| Value' )
    plt.savefig(path + 'mi_results.png', dpi = 800)    
    plt.close('all')

#-----------------------Neural network related parameters--------------------
lr_mine = 0.01
batch_size = 10000
batch_size_m = 1000
batch_num = 300
print_epoch = 1
ema = 0
ehanced = 1
#-----------------------------other parameters-------------------------------
caclu_with_gpu = 1
device = torch.device("cuda:0" if (torch.cuda.is_available() and caclu_with_gpu > 0) else "cpu")
qamnum = 256
qammap_np = np.loadtxt('./constellation/' + str(qamnum) + 'qam.txt')
qammap = Normalization(torch.from_numpy(qammap_np)).to(device).float()
if ehanced:
    path = './results/EMINE/'
else:
    path = './results/MINE/'
path = './results/'
mkdir(path)

snr = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
epoch = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
snr_list_num = len(snr)
epoch_list_num = len(epoch)
mi_true = np.zeros_like(snr)+0.0

mi_results = mi_true.reshape((-1, 1)) + np.random.rand(snr_list_num, epoch_list_num) * 0.02 - 0.01
mask = np.zeros_like(mi_results)
plot_mi(mi_results, mask)

mi_results = np.zeros((snr_list_num, epoch_list_num)) 
mask = np.ones_like(mi_results, dtype=bool)
qammap_np = norm1d(qammap_np[:,0]+1j*qammap_np[:,1])

for i_snr in range(snr_list_num):
    mi_true[i_snr] = mi_gh(qammap_np, snr[i_snr].reshape((-1, 1)))
    for i_epoch in range(epoch_list_num):
        mi_results[i_snr, i_epoch] = train(epoch_num = epoch[i_epoch],\
            snr = snr[i_snr], true_mi = mi_true[i_snr])
        mask[i_snr, i_epoch] = False
        plot_mi(mi_results, mask)
if ehanced:
    np.savez(path + 'EMINE/mi_results.npz', mi_results)
else:
    np.savez(path + 'MINE/mi_results.npz', mi_results)
