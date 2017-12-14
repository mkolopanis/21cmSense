#! usr/bin/env python
from scipy.interpolate import interp2d
import numpy as np, re

'''
    Python module used to accompany 21cmSense.
'''

def load_noise_files(files=None,verbose=False,polyfit_deg=3,
    thermal_noise=True,kmax=None,kmin=None,num_ks=50):
    '''
    Loads input 21cmsense files and clips nans and infs

    polyfit_deg: define degree of polynomial used to fit and regrid noise (defualt 3)
    thermal_noise: set to True to use thermal noise else will use noise with
                   sample variance factored in (Defaulte True)

    kmax: maximum k-value to which to re-grid data (default file min)
    kmin: minimum k-value to which to re-grid data (default file max)
    num_k: number of k-values in new grid (default 50)

    The returned noise_values array masked outside of k-range in file
    returns freqs, ks, noise_values
    '''

    if files is None:
        if verbose: print 'No files Input'
        if verbose: print 'Exiting'
        return 0,'',''

    if not files:
        if verbose: print 'No Files Input'
        return 0,'',''

    #wrap single files for iteration
    flag=False
    if len(np.shape(files)) ==0: files = [files]; flag=True
    files.sort()
    re_f = re.compile('(\d+\.\d+)')#glob should load them in increasing freq order

    noises = []
    noise_ks = []
    noise_freqs = []

    for noisefile in files:
        #load thermal noise or sample variance noise
        if thermal_noise: noise = np.load(noisefile)['T_errs']
        else: noise = np.load(noisefile)['errs']
        noise_k = np.load(noisefile)['ks']

        bad = np.logical_or(np.isinf(noise),np.isnan(noise))
        noise = noise[np.logical_not(bad)]
        noise_k = noise_k[np.logical_not(bad)]

        #set up k range for re-gridding
        if kmin is None: kmin = np.min(noise_k)
        if kmax is None: kmax = np.max(noise_k)
        nk_grid = np.linspace(kmin,kmax,num_ks)

        #keep only the points that fall in our desired k range
        good_k = noise_k < nk_grid.max()
        noise = noise[noise_k<nk_grid.max()]
        noise_k = noise_k[noise_k<nk_grid.max()]

        if verbose: print noisefile,np.max(noise),

        #regrid data by fitting polynomial then evaluating
        #tmp_fit = np.polyfit(noise_k,noise,polyfit_deg)
        #noise = np.poly1d(tmp_fit)(nk_grid)

        #noise[ np.logical_or(nk_grid < noise_k.min()
        #    , nk_grid> noise_k.max())] = np.NaN #Nan regions outside of k-range

        noises.append(noise)
        #noise_ks.append(nk_grid)
        noise_ks.append(noise_k)

        small_name = noisefile.split('/')[-1].split('.npz')[0].split('_')[-1]
        f = float(re_f.match(small_name).groups()[0])*1e3 #sensitivity freq in MHz
        if verbose: print f
        noise_freqs.append(f)

    #from IPython import embed; embed()
    #noises = np.ma.masked_invalid(noises)
    if flag:
        noise_freqs = np.squeeze(noise_freqs)
        noise_ks = np.squeeze(noise_ks)
        noises = np.squeeze(noises)
    return noise_freqs, noise_ks, noises


def noise_interp2d(noise_freqs=None,noise_ks=None,noises=None,
        interp_kind='linear', verbose=False ,**kwargs):
    '''
    Builds 2d interpolator from loaded data, default interpolation: linear
    interpolator inputs  frequency (in MHz), k (in hMpci)
    '''
    if noise_freqs is None:
        if verbose: print 'Must Supply frequency values'
        return 0
    if noise_ks is None:
        if verbose: print 'Must supply k values'
        return 0
    if noises is None:
        if verbose: print 'Must supply Noise'
        return 0

    noise_k_range = [np.min(np.concatenate(noise_ks)),np.max(np.concatenate(noise_ks))]

    if np.min(noise_k_range) == np.max(noise_k_range):
        if verbose:
            print 'K range only contains one value'
            print 'Exiting'
        return 0

    nk_count = np.mean([len(myks) for myks in noise_ks])
    nks = np.linspace(noise_k_range[0],noise_k_range[1],num=nk_count)
    noises_interp = np.array([np.interp(nks,noise_ks[i],noises[i],
                    left=np.NaN, right=np.NaN) for i in range(len(noises))])

    NK,NF = np.meshgrid(nks,noise_freqs)
    noises_interp = np.ma.masked_invalid(noises_interp)

    #mask out Nans during interpolation
    NK1,NF1 = NK[~noises_interp.mask], NF[~noises_interp.mask]
    noises_interp1 = noises_interp[~noises_interp.mask]

    noise_interp = interp2d(NF1, NK1, noises_interp1, kind=interp_kind,
            fill_value=np.NaN ,**kwargs)

    return noise_interp

def masked_noise(noise_interp,freqs,ks):
    '''
    given a noise_interp2d object and freqs and ks,
    returns masked array of noise values in desired frequency and k range
    '''
    return np.ma.masked_equal( np.ma.masked_invalid(noise_interp(freqs,ks)),0)
