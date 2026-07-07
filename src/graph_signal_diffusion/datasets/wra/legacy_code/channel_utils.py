import numpy as np
import numpy.matlib
from scipy.spatial import distance
from numpy import linalg as LA
import torch
import os


DELTA_T = 50  # time interval in ms
T_DEFAULT = 100  # default number of time slots

# Global parameters
# min_D_TxTx = 35 # minimum Tx-Tx distance
# min_D_TxRx = 10 # minimum Tx-Rx distance
# max_D_TxRx = 50 # maximum Tx-Rx distance
# shadowing = 7 # shadowing standard deviation
# f_c = 2.4e9 # carrier frequency (Hz)
# speed = 1. # receiver speed (m/s) # 1.0


def get_network_area(m, density_mode, base_R = 1000):

    density_n_users_per_km_squared = None
    # base_R = 1000 # 1000

    # set network area side length based on the density mode
    if density_mode == 'var':
        R = base_R # if R is None else args.R
        density_n_users_per_km_squared = m / (R ** 2 / 1e6)
    elif density_mode == 'fixed':
        if density_n_users_per_km_squared is None:
            R = base_R * np.sqrt(m / 20)

            density_n_users_per_km_squared = m / (R ** 2 / 1e6) # in users/km^2
        # else:
        #     R = 1e3 * np.sqrt(m / density_n_users_per_km_squared) # in meters
    else:
        raise Exception
    
    print(f"Network area side length (R) = {R}, realized density = {density_n_users_per_km_squared}, density mode = {density_mode}, number of users (m) = {m}")

    return R


def convert_P_max_and_noise_PSD(P_max_dBm, BW, noise_PSD_dBm):

    P_max = np.power(10, (P_max_dBm - 30) / 10) # maximum transmit power = 10 dBm
    N = noise_PSD_dBm - 30 + 10 * np.log10(BW) # Noise PSD = -174 dBm/Hz
    noise_var = np.power(10, N / 10) # noise variance

    return P_max, noise_var


def UDN_PL(D, alpha1=2):
    m, n = np.shape(D)
    L = np.zeros((m, n))
    k0 = 39
    a1 = alpha1
    a2 = 4
    db = 100

    CONST = 10 * np.log10(db ** (a2-a1))
    
    for i in range(m):
        for j in range(n):
            d = D[i,j] 
            if d <= db:
                L[i,j] = k0 + 10 * a1 * np.log10(d)
            else:
                L[i,j] = k0 + 10 * a2 * np.log10(d) - CONST
    return L


def long_term_fading(D_TxRx, shadowing, shadowing_loss = None, **kwargs):
    """
    Long-term channel fading model
    """
    return_all_data = kwargs.get('return_long_term_fading_data', False)

    path_loss = UDN_PL(D_TxRx)
    if shadowing_loss is None:
        shadowing_loss = shadowing * np.random.randn(*D_TxRx.shape) # shadowing in dB
    L = path_loss + shadowing_loss # Loss matrix in dB
    H_l = np.sqrt(np.power(10, -L / 10)) # large-scale fading matrix
    
    if return_all_data:
        return H_l, {"loss": L, "path_loss": path_loss, "shadowing_loss": shadowing_loss}
    
    else:
        return H_l, None


def short_term_fading(T, N, f_c, speed):
    """ Rayleigh fading model via sum of sinusoids
    from "Model of independent Rayleigh faders" by Z. Wu
    """
    t_vec = np.array(range(T)) / 1e3 * DELTA_T
    a_0 = np.pi / (2*N)
    w_M = 2 * np.pi * f_c * speed / 3e8
    
    N0 = N // 4
    alpha = a_0 + 2 * np.pi * np.array(range(N0)) / N
    theta = 2 * np.pi * np.random.uniform(0, 1, (N0, 1))
    theta_p = 2 * np.pi * np.random.uniform(0, 1, (N0, 1))
    
    I = np.cos(w_M * np.outer(np.cos(alpha) , t_vec) + np.matlib.repmat(theta, 1, len(t_vec)))
    Q = np.sin(w_M * np.outer(np.sin(alpha) , t_vec) + np.matlib.repmat(theta_p, 1, len(t_vec)))
    
    h = sum(I + 1j * Q) / np.sqrt(N0)
    
    return h


def load_channels_from_path(path):
    if isinstance(path, list):
        tx_rx_locs_H_l = []
        channel_gains = []
        for p in path:
            if os.path.exists(p):
                temp_tx_rx_locs_H_l, temp_channel_gains = torch.load(p)

                tx_rx_locs_H_l.extend(temp_tx_rx_locs_H_l)
                channel_gains.extend(temp_channel_gains)
            print("Loaded channel data from: ", p)
        print("Total number of channel data files loaded: ", len(tx_rx_locs_H_l))
        return tx_rx_locs_H_l, channel_gains

    else:
        if os.path.exists(path):
            return torch.load(path)
        


def save_channel_data_to_path(data, path, max_size = 256):

    if len(data) < max_size:
        torch.save(data, path)

    else:
        print("Data is too large, splitting into multiple files...")
        print("Data size: ", len(data))
        print("Max size: ", max_size)   
        print("Saving data to multiple files...")
        i = 0
        while len(data) > max_size:
            subdata = data[:max_size]
            # Get the beginning of the string path before the last ".json" string
            modified_path = path[:path.rfind('.json')] + f'-{i}.json'
            torch.save(subdata, modified_path)
            i += 1
            data = data[max_size:]
        
        torch.save(data, path)





def deploy_tx_rx_pairs(m, n, R, min_D_TxTx, min_D_TxRx, max_D_TxRx, shadowing):
    '''
    Deploy transmitter and receiver pairs.
    '''

    print(f"Deploying {m} transmitters and {n} receivers in a network area of size {R}x{R} with minimum Tx-Tx distance {min_D_TxTx}, minimum Tx-Rx distance {min_D_TxRx}, maximum Tx-Rx distance {max_D_TxRx}, and shadowing standard deviation {shadowing} dB.")

    # specify transmitter locations
    while True:
        locTx = np.random.uniform(0, R, (m, 2)) - R / 2
        D_TxTx = distance.cdist(locTx, locTx, 'euclidean')
        for Tx in range(m):
            D_TxTx[Tx, Tx] = float('Inf')
        if np.quantile(D_TxTx, 0) >= min_D_TxTx:
            break

    # specify receiver locations
    H_l = np.zeros((m, n))
    while True:
        phi = 2 * np.pi * np.random.uniform(size = (n,))
        r = np.sqrt(np.random.uniform(low=min_D_TxRx ** 2, high=max_D_TxRx ** 2, size=(n,)))
        locRx = np.clip(locTx + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1), a_min= -R / 2, a_max=R / 2)

        D_TxRx = distance.cdist(locTx, locRx, 'euclidean')
        if np.quantile(D_TxRx, 0) < min_D_TxRx:
            continue

        H_l, long_term_fading_dict = long_term_fading(D_TxRx, shadowing, return_long_term_fading_data=True)

        associations = (H_l == np.max(H_l, axis=0, keepdims=True))
        if min(np.sum(associations, axis=1)) > 0: # each transmitter has at least one associated reciever
            break

    return {'locTx': locTx, 'locRx': locRx, 'H_l': H_l, 'associations': associations, 'large_scale_fading_dict': long_term_fading_dict}



def create_channel_matrix_over_time(H_l, T, **kwargs):
    '''
    Create channel matrix over time.
    ''' 
    num_fading_paths = kwargs.get('num_fading_paths', 100)
    f_c = kwargs.get('f_c', 2.4e9)
    speed = kwargs.get('speed', 1.)
    disable_short_term_fading = kwargs.get('disable_short_term_fading', False)

    m, n = H_l.shape

    # if disable_short_term_fading:
    #     H = torch.ones((m, n, T), dtype = torch.float32)
    # else:
    if disable_short_term_fading:
        speed = 0.

    H = np.zeros((m, n, T), dtype=complex)
    for i in range(m):
        for j in range(n):
            # H[i, j] = short_term_fading(T, 100, f_c, speed)
            H[i, j] = short_term_fading(T, num_fading_paths, f_c, speed)

                      
    # H *= H_l
    H *= np.expand_dims(H_l, axis=2)
    
    # H_l = H_l.squeeze(-1)

    channel_gains = {'H': np.abs(H)**2,
                    'H_l': np.abs(H_l)**2
                    }
    
    return channel_gains


def normalize_matrix(A, normalization = 'matrix-norm', ord = None, axis = None, keepdims = False):
    if normalization == 'matrix-norm':
        A_norm = LA.norm(A, ord = ord, axis=axis, keepdims=keepdims)
    elif normalization == 'max-eigenvalue':
        # Compute the largest eigenvalue of A
        eigs = LA.eigvals(A)
        A_norm = np.max(np.abs(eigs))
    else:
        raise NotImplementedError
    
    return A_norm


def convert_channels(a, snr, conversion = 'correlation', normalization = 'matrix-norm', return_normalization_factor = False):
    # a_flattened = a[a > 0]
    a_flattened = a  # a_flattened[a_flattened < np.inf]
    # print("a_flattened: ", a_flattened)
    assert np.all(a_flattened > 0), "All elements of the channel matrix must be positive."

    if conversion == 'log':
        a_flattened_log = np.log(snr * a_flattened)
        a_norm = normalize_matrix(a_flattened_log, normalization=normalization)
        # a_norm = normalize_matrix(a_flattened_log, ord = float('inf'))
        a_log = np.log(snr * a)
        a_log[a == 0] = 0
        # return a_log / a_norm
    
    elif conversion == 'log-one-plus':
        a_flattened_log = np.log(1 + snr * a_flattened)
        a_norm = normalize_matrix(a_flattened_log, normalization=normalization)
        # a_norm = normalize_matrix(a_flattened_log, axis = None, ord = float('inf'), keepdims=False)
        a_log = np.log(1 + snr * a)
        a_log[a == 0] = 0
        # return a_log / a_norm
    
    elif conversion == 'correlation':
        a_log = np.log(1 + snr * a)
        a_norm = normalize_matrix(a_log, axis = 0, ord = float('inf'), keepdims = True)
        # a_log = a_log / a_norm
        # return a_log
    
    else:
        raise NotImplementedError

    if return_normalization_factor:
        return a_log / a_norm, a_norm
    else:
        return a_log / a_norm



# calculating rates
def calc_rates(p, gamma, h, noise_var):
    """
    calculate rates for a batch of b networks, each with m transmitters and n recievers
    inputs:
        p: bm x 1 tensor containing transmit power levels
        gamma: bn x 1 tensor containing user scheduling decisions
        h: b x (m+n) x (m+n) weighted adjacency matrix containing instantaneous channel gains
        noise_var: scalar indicating noise variance
        training: boolean variable indicating whether the models are being trained or not; during evaluation, 
        entries of gamma are forced to be integers to satisfy hard user scheudling constraints
        
    output:
        rates: bn x 1 tensor containing user rates
    """
    b = h.shape[0]
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]
    
    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
    
    rates = torch.log2(1 + signal / (noise_var + interference)).view(-1, 1)
    
    return rates


def ITLinQ(H_raw, Pmax, noise_var, PFs):
    '''
    https://github.com/navid-naderi/StateAugmented_RRM_GNN/blob/main/data_gen.py
    '''
    H = H_raw * Pmax / noise_var
    n = np.shape(H)[0]
    prity = np.argsort(PFs)[-1:-n-1:-1]
    flags = np.zeros(n)
    M = 10 ** 2.5
    eta = 0.5
    flags[prity[0]] = 1
    for pair in prity[1:]:
        SNR = H[pair,pair]
        INRs_in = [H[TP,pair] for TP in range(n) if flags[TP]]
        INRs_out = [H[pair,UE] for UE in range(n) if flags[UE]]
        max_INR_in = max(INRs_in)
        max_INR_out = max(INRs_out)
        if max(max_INR_in,max_INR_out) <= M * (SNR ** eta):
            flags[pair] = 1
    return flags * Pmax



def wmmse(H, Pmax, noise_var, num_iters=100):
    '''
    https://github.com/navid-naderi/Resilient_RRM_GNN/blob/main/baselines.py
    '''
    h2 = np.copy(H)
    h = np.sqrt(h2)
    m = H.shape[1]
    n = H.shape[0]
    v = np.ones((n,m))*np.sqrt(Pmax)/2
    # num_iters = 100 # number of iterations
    v2 = np.expand_dims(v ** 2, axis=2)

    u = (np.diagonal(h,axis1=1,axis2=2) * v) / (np.matmul(h2,v2)[:,:,0] + noise_var)
    w = 1 /(1 - u *np.diagonal(h,axis1=1,axis2=2) *v)
    # N = 1000
    for _ in np.arange(num_iters):

        u2 = np.expand_dims(u**2, axis=2)
        w2 = np.expand_dims(w, axis=2)
        v = (w *u *np.diagonal(h,axis1=1,axis2=2)) / (np.matmul(np.transpose(h2,(0,2,1)),(w2*u2)))[:,:,0]
        v = np.minimum(np.sqrt(Pmax),np.maximum(0,v))
        v2 = np.expand_dims(v**2,axis=2)
        u = (np.diagonal(h,axis1=1,axis2=2)*v)/ ( np.matmul(h2,v2)[:,:,0] + noise_var)
        w = 1 /(1 - u*np.diagonal(h,axis1=1,axis2=2)*v)
    p = v**2
    return p


# def wmmse(H, Pmax, noise_var):
#     h2 = np.copy(H)
#     h = np.sqrt(h2)
#     m = H.shape[1]
#     n = H.shape[0]
#     v = np.ones((n,m))*np.sqrt(Pmax)/2
#     T = 100
#     v2 = np.expand_dims(v ** 2, axis=2)

#     u = (np.diagonal(h,axis1=1,axis2=2) * v) / (np.matmul(h2,v2)[:,:,0] + noise_var)
#     w = 1 /(1 - u *np.diagonal(h,axis1=1,axis2=2) *v)
#     # N = 1000
#     # for t in np.arange(T):

#     u2 = np.expand_dims(u**2, axis=2)
#     w2 = np.expand_dims(w, axis=2)
#     v = (w *u *np.diagonal(h,axis1=1,axis2=2)) / (np.matmul(np.transpose(h2,(0,2,1)),(w2*u2)))[:,:,0]
#     v = np.minimum(np.sqrt(Pmax),np.maximum(0,v))
#     v2 = np.expand_dims(v**2,axis=2)
#     u = (np.diagonal(h,axis1=1,axis2=2)*v)/ ( np.matmul(h2,v2)[:,:,0] + noise_var)
#     w = 1 /(1 - u*np.diagonal(h,axis1=1,axis2=2)*v)

#     p = v**2
#     return p