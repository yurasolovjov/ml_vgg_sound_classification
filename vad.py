import numpy as np
import pandas 

def double_exp_average(X, sr, vad_win_average_begin = 0.060, vad_win_average_end = 0.060):
    nLen = X.shape[0]

    En = X**2

    Y = np.zeros(X.shape)
    Z = np.zeros(X.shape)
    Alpha = 1.0 - 1.0/(vad_win_average_begin*sr)
    Beta  = 1.0 - 1.0/(vad_win_average_end*sr)

    for i in range(0, nLen - 1, 1):
        Y[i+1] = Alpha*Y[i] + (1-Alpha)*En[i+1]

    for i in range(nLen - 1, 0, -1):
        Z[i-1] = Beta*Z[i] + (1-Beta)*Y[i-1]

    return Z

def double_exp_average_fast(X, sr, vad_win_average_begin = 0.060, vad_win_average_end = 0.060):

    En = X ** 2
    d = pandas.ewma(En, com=sr * vad_win_average_begin - 1, adjust=False)
    d = d[::-1]
    d[0] = 0
    out = pandas.ewma(d, com=sr * vad_win_average_end - 1, adjust=False)
    ff = np.copy(out[::-1])
    return ff


#best
def apply_vad(X, sr, alfa_t = 0.2, beta_t  = 0.2, percent_th = 100.0):
    
    if len(X) < 1:
        return X

    En = double_exp_average_fast(X, sr, alfa_t, beta_t)
    max_el = max(En)
    threshold = max_el/percent_th
    return X[En>threshold]
