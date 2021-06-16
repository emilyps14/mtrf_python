# -*- coding: utf-8 -*-
"""
Python Port of Yulia Oganian's peakRate Matlab package:
https://github.com/yoganian/peakRate

This function detects peakRate events, as defined in Oganian & Chang, 2019,
A speech envelope landmark for syllable encoding in human superior temporal
gyrus. Science Advances, 5(11), eaay6279.

:Author: Ported to python by Emily P. Stephen
"""
import warnings

import numpy as np
from scipy.fft import fft,ifft
from scipy.signal import butter, filtfilt, find_peaks, resample_poly, lfilter
from scipy.ndimage import gaussian_filter


def find_peakRate(sound, soundfs, onsOff=None, envtype='loudness', envfs=100,
                  cleanup_flag=False):
    '''
    This function creates the speech envelope and a timeseries of discrete
    peakRate events, as used in Oganian & Chang, 2019,
    A speech envelope landmark for syllable encoding in human superior temporal
    gyrus. Science Advances, 5(11), eaay6279.

    Original code: https://github.com/yoganian/peakRate
    (c) Yulia Oganian, Oct 2019
    yulia.oganian@ucsf.edu

    Ported to python by Emily P. Stephen 2020

    Parameters
    ----------
    sound : ndarray
        (T,), sound waveform
    soundfs : int
        sampling frequency of sound
    onsOff : list or None
        (2,) times of stimulus onset and offset in sound
        onsOff=None or onsOff=[None,None]: uses the beginning until end of the signal
        onsOff=[None,endtime] : uses beginning until endtime
        onsOff=[starttime,None] : uses starttime until end or signal
    envtype : str
        'loudness' (default), or 'broadband': specific loudness envelope or broadband envelope
    envfs : int
        the sampling rate to use for the envelope calculation
        Note that the peakRate values will be lower, since it is measured in units per sample
    cleanup_flag : bool
        if set to True, landmark series will be cleaned up to contain only a single
        peakRate event in each envelope cycle, defined as envelope
        trough-to-trough (default: False)


    Returns
    -------
    env - amplitude envelope of input
    peakRate - discrete time series of peakRate events in envelope (change in envelope per sample)
    peakEnv - discrete time series of peakEnv events
    '''

    # %% initialize
    # need to know when signal contains speech to remode landmark events
    # outside speech.
    if onsOff is None:
        onsOff= [None, None]

    # %% get envelope
    if envtype=='loudness': # specific loudness
            env = Syl_1984_Schotola(sound, soundfs, envfs)
    elif envtype=='broadband': # broadband envelope
            rectsound  = np.abs(sound)
            [b,a] = butter(2, 10/(soundfs/2))
            cenv = filtfilt(b, a, rectsound)
            if envfs!=soundfs:
                downsenv = resample_poly(cenv, envfs, soundfs)
                env = downsenv
            else:
                env = cenv
            env[env<0] = 0
    else:
        raise RuntimeError(f'envtype {envtype} not recognized')

    # %% get landmarks in envelope

    # vector marking speech onset and offset times.
    onsOffenv = np.zeros([2, len(env)])
    if onsOff[0] is None:
        onsOffenv[0,0] = 1
    else:
        onsOffenv[0,np.ceil(onsOff[0]*envfs).astype(int)] = 1
    if onsOff[1] is None:
        onsOffenv[1,-1] = 1
    else:
        onsOffenv[1,np.round(onsOff[1]*envfs).astype(int)] = 1

    allTS, varNames = find_landmarks(env, onsOffenv, cleanup_flag)
    peakEnv = allTS[3,:]
    peakRate = allTS[5,:]

    return env, peakRate, peakEnv


def Syl_1984_Schotola(p,fs,sr=100):
    '''
    Specific loudness function for envelope extraction

    This identifies vowel nuclei according to the method of Schotola (1984).
    The calculation of modifed total loudness is identical to that of
    Syl_1979_Zwicker_etal_v1.m (e.g., Zwicker et al. 1979), but some
    additional rules/improvements are made here. This is continuation of work
    by Ruske and Schotola (1978, 1981), but this is the best-specified.

    Parameters
    ----------
    p : ndarray
        (N,): speech waveform, p(t)
    fs : int
        sample rate (Hz)
    sr : int
        target sample rate

    Returns
    -------
    Nm : ndarray
        (N*sr/fs,): modified total loudness, Nm(t), used to identify VN and B


    written by Eric Edwards
    adopted by Yulia Oganian, yulia.oganian@ucsf.edu
    ported to python by Emily P Stephen
    '''

    N = len(p)
    tN = np.arange(N)/fs
    T = 1/fs

    # Loudness functions will be sampled at 100 Hz
    N1 = np.fix(N*sr/fs).astype(int)
    t1 = np.arange(N1)/sr

    # Critical-band filters are applied in freq-domain
    p = fft(p,N)
    frqs = np.arange(N/2)*(fs/N) #FFT positive freqs
    nfrqs = len(frqs)

    # Set up for cricial-band filter bank (in freq-domain)
    z = 13*np.arctan(.76*frqs/1000) + 3.5*np.arctan(frqs/7500)**2 #Bark (Zwicker & Terhardt 1980)
    z = z[1:nfrqs-1] #to set DC and Nyquist to 0

    # Set up for simple (RC) smoothing with 1.3-ms time-constant
    tau = 0.0013
    r = np.exp(-T/tau)
    b1 = [1-r]
    a1 = np.poly([r])

    F = np.zeros(N,dtype=np.double) #will hold critical-band filter shape
    czs = np.arange(22)+1
    nczs = len(czs)
    Nv = np.zeros([N1, nczs],dtype=np.double) #will hold the specific loudnesses
    for ncz,cz in enumerate(czs):
        F[1:nfrqs-1] = 10**(.7-.75*((z-cz)-.215)-1.75*(0.196+((z-cz)-.215)**2))
        # F = F./sum(F)
        Lev = np.real(ifft(p*F,N))
        Lev = Lev**2 #square-law rectification (Vogel 1975)
        Lev = lfilter(b1,a1,Lev) #smoothing (1.3-ms time-constant)
        Lev = np.flip(lfilter(b1,a1,np.flip(Lev)))
        # the last line makes zero-phase smoothing, comment out to leave causal

        Lev = np.log(Lev) #logarithmic nonlinearity this is now "excitation level"
        # The "specific loudness", Nv(t), is made by "delog" of .5*Lev(t)
        if N1!=N:
            Nv[:,ncz] = np.interp(t1,tN,np.exp(.5*Lev))
        else:
            Nv[:,ncz] = np.exp(.5*Lev)
    Nv = np.maximum(0,Nv) #in case negative values, set to 0

    # Nm is the total modified loudness, Nm(t). This is LPFed by a 3-point
    # triangular filter, n=26 times (I do 13 fwd and 13 backwd for zero-phase),
    # which results in ~Gaussian smoothing with sig~=100 ms.
    gv = np.ones(nczs,dtype=np.double) #weights
    gv[czs<3] = 0
    gv[czs>19] = -1
    Nm = np.dot(Nv,gv)
    if sr==100:
        # original method
        b = np.ones(3,dtype=np.double)/3
        n = 13
        for nn in range(n):
            Nm = filtfilt(b,1,Nm)
    else:
        # ES 10/2/20: For other sample rates, it works better to use an actual
        # Gaussian filter. The effective std above is closer to 42ms (I fit a
        # Gaussian to the equivalent filter to get the value below)
        Nm = gaussian_filter(Nm, 0.0416333*sr, mode='constant')

    # REFERENCES:
    #
    # Bismarck Gv (1973). Vorschlag f�r ein einfaches Verfahren zur
    # Klassifikation station�rer Sprachschalle. Acustica 28(3): 186-7.
    #
    # Zwicker E, Terhardt E, Paulus E (1979). Automatic speech recognition using
    # psychoacoustic models. J Acoust Soc Am 65(2): 487-98.
    #
    # Ruske G, Schotola T (1978). An approach to speech recognition using
    # syllabic decision units. ICASSP: IEEE. 3: 722-5.
    #
    # Ruske G, Schotola T (1981). The efficiency of demisyllable segmentation in
    # the recognition of spoken words. ICASSP: IEEE. 6: 971-4
    #
    # Schotola T (1984). On the use of demisyllables in automatic word
    # recognition. Speech Commun 3(1): 63-87.

    return Nm



def find_landmarks(TS,  onsOff, cleanup_flag):
    # landmark detection in envelope

    # %% find discrete events in envelope

    # normalize envelope to between -1 and 1
    TS /= np.max(np.abs(TS))

    TS[np.where(onsOff[1,:]==1)[0][0]:] = 0 # zero out after offset
    TS[:np.where(onsOff[0,:]==1)[0][0]] = 0 # zero out before onset

    # first temporal derivative of TS
    diff_loudness = np.append(np.diff(TS),0)

    # %% discrete loudness
    # min
    minloc,_ = find_peaks(-TS)
    lmin = TS[minloc]
    minEnv = np.zeros(TS.shape)
    minEnv[minloc]=lmin
    # max
    maxloc,_ = find_peaks(TS)
    lmax = TS[maxloc]
    peakEnv = np.zeros(TS.shape)
    peakEnv[maxloc]=lmax

    # %% discrete delta loudness
    # min
    negloud = diff_loudness.copy()
    negloud[negloud>0] = 0
    minloc,_ = find_peaks(-negloud)
    lmin = negloud[minloc]
    minRate = np.zeros(TS.shape)
    minRate[minloc]=lmin
    # max
    posloud = diff_loudness.copy()
    posloud[posloud<0] = 0
    maxloc,_ = find_peaks(posloud)
    lmax = posloud[maxloc]
    peakRate = np.zeros(TS.shape)
    peakRate[maxloc]=lmax

    del negloud, posloud
    
    # %% complete loudness information
    allTS = np.stack([TS,
        diff_loudness,
        minEnv,
        peakEnv,
        minRate,
        peakRate],axis=0)

    # %% --------------- clean up

    if cleanup_flag:
        ## start with maxima in envelope
        cmaxloc = np.where(allTS[3,:]!=0)[0]
        cmax = allTS[3,cmaxloc]

        # initialize all other landmark variables
        cmin = np.full(cmaxloc.shape, np.nan)
        cminloc = np.full(cmaxloc.shape, np.nan)
        cminDt = np.full(cmaxloc.shape, np.nan)
        cminDtLoc = np.full(cmaxloc.shape, np.nan)
        cmaxDt = np.full(cmaxloc.shape, np.nan)
        cmaxDtLoc = np.full(cmaxloc.shape, np.nan)

        # --- define minima in envelope for each peak in envelope

        # first peak - getting cmin, cminloc
        cminloc[0] = 0
        cmin[0] = 0.001

        # remaining peaks
        for i in range(1,len(cmaxloc)):
            # find troughs between two consecutive peaks
            cExtrLoc = np.where(allTS[2,cmaxloc[i-1]:cmaxloc[i]]!=0)[0]
            cExtr = allTS[2, cmaxloc[i-1]+cExtrLoc]
            if len(cExtr)==1: # this is standard case - one min per peak
                cmin[i] = cExtr
                cminloc[i] = cmaxloc[i-1] + cExtrLoc
            elif len(cExtr) > 1: # if multiple troughs, use the lowest one should not happen ever.
                cmin[i] = np.min(cExtr)
                cl = np.argmin(cExtr)
                cminloc[i] = cExtrLoc[cl] + cmaxloc[i-1]
            elif len(cExtr)==0: # no minima in this window found by general algorithm define as lowest point between this and previous peak.
                cExtr = np.min(allTS[0, cmaxloc[i-1]:cmaxloc[i]])
                cExtrLoc = np.argmin(allTS[0, cmaxloc[i-1]:cmaxloc[i]])
                cminloc[i] = cExtrLoc + cmaxloc[i-1] + 2 # +2 required for consistency with Matlab code (I'm not sure why it's there, doesn't affect PeakRate or peakEnv -- ES 10/1/20)
                cmin[i] = cExtr

        ## # #  peakRate # # #
        for i in range(len(cmaxloc)):
            if i == 0: # first peak
                cExtrLoc = np.where(allTS[5,:cmaxloc[0]]!=0)[0]
                cExtr = allTS[5, cExtrLoc]
                prevloc = 0
            else: # remaining peaks
                cExtrLoc = np.where(allTS[5,cmaxloc[i-1]:cmaxloc[i]]!=0)[0]
                cExtr = allTS[5, cmaxloc[i-1]+cExtrLoc]
                prevloc = cmaxloc[i-1]

            if len(cExtr)==1:
                cmaxDt[i] = cExtr
                cmaxDtLoc[i] = cExtrLoc+prevloc
            elif len(cExtr)>1:
                cl = np.argmax(cExtr)
                cmaxDt[i] = cExtr[cl]
                cmaxDtLoc[i] = cExtrLoc[cl]+prevloc
            elif len(cExtr)==0:
                warnings.warn(f'no peakRate found in cycle {i} \n')

        ## # #  minRate # # #
        # all but last peaks
        for i in range(len(cmaxloc)-1):
            cExtrLoc = np.where(allTS[4,cmaxloc[i]:cmaxloc[i+1]]!=0)[0]
            cExtr = allTS[4, cmaxloc[i]+cExtrLoc]
            if len(cExtr)==1:
                cminDt[i] = cExtr
                cminDtLoc[i] = cExtrLoc+cmaxloc[i]
            elif len(cExtr)==0:
                warnings.warn(f'no rate trough in cycle {i} \n')
            elif len(cExtr)>1:
                cl = np.argmin(cExtr)
                cminDt[i] = cExtr[cl]
                cminDtLoc[i] = cExtrLoc[cl]+cmaxloc[i]

        # last peak
        peakId = len(cmaxloc)-1
        envelopeEnd = np.where(TS!=0)[0][-1]
        cExtrLoc = np.where(allTS[4,cmaxloc[-1]: envelopeEnd-1]!=0)[0]
        cExtr = allTS[4, cExtrLoc+cmaxloc[-1]]
        if len(cExtr)==1:
            cminDt[peakId] = cExtr
            cminDtLoc[peakId] = cExtrLoc+cmaxloc[-1]-1
        elif len(cExtr)>1:
            cl = np.argmin(cExtr)
            cminDt[peakId] = cExtr[cl]
            cminDtLoc[peakId] = cExtrLoc[cl]+cmaxloc[-1]
        elif len(cExtr)==0:
            warnings.warn(f'no minDtL in cycle {i} \n')

        ## detect silence in middle of utterance
        if sum(cmin==0)>0:
            warnings.warn('0 min found \n')

        ## combine all info
        cextVal = [cmin,cmax,cminDt,cmaxDt]
        cextLoc = [cminloc,cmaxloc,cminDtLoc,cmaxDtLoc]

        # redo allTS with cleaned values.
        for i,(locs,vals) in enumerate(zip(cextLoc,cextVal)):
            allTS[i+2,:]=0
            allTS[i+2, locs[~np.isnan(locs)].astype(int)] = vals[~np.isnan(vals)]
    varNames = {'Loudness', 'dtLoudness', 'minenv', 'peakEnv', 'minRate', 'peakRate'}
    return allTS, varNames

def find_peakRate_events(sound, soundfs, onsOff=None, envtype='loudness',
                         envfs=100, cleanup_flag=False):
    '''
    Wrapper for find_peakRate that computes the times of peakRate and peakEnv
    events using the temporal resolution of the original sound file. The
    "envfs" parameter is used to convert the peakRate slope values to the scale
    that is comparable to the original Oganian and Chang (2019) paper, i.e.
    change in normalized envelope per 10 ms.

    For change in envelope per second, use envfs=1
    For the slope in the scale of the sound file, use envfs=soundfs

    Emily P. Stephen 2020

    Parameters
    ----------
    sound : ndarray
        (T,), sound waveform
    soundfs : int
        sampling frequency of sound. The peakRate and peakEnv events will be
        detected at this frequency
    onsOff : list or None
        (2,) times of stimulus onset and offset in sound
        onsOff=None or onsOff=[None,None]: uses the beginning until end of the signal
        onsOff=[None,endtime] : uses beginning until endtime
        onsOff=[starttime,None] : uses starttime until end or signal
    envtype : str
        'loudness' (default), or 'broadband': specific loudness envelope or broadband envelope
    envfs : int
        the sampling rate to convert the peakRate values to
    cleanup_flag : bool
        if set to True, landmark series will be cleaned up to contain only a single
        peakRate event in each envelope cycle, defined as envelope
        trough-to-trough (default: False)


    Returns
    -------
    peakRateTimes - times of peakRate events (in seconds)
    peakRateValues - peakRate values (change in envelope per sample, based on envfs)
    peakEnvTimes - times of peakEnv events (in seconds)
    peakEnvValues - peakEnv values (normalized envelope)
    '''
    # Compute the peakRate at the original sampling frequency
    _, peakRate, peakEnv = find_peakRate(sound, soundfs, onsOff=onsOff,
                                         envtype=envtype,
                                         cleanup_flag=cleanup_flag,
                                         envfs=soundfs)

    # Convert
    pe_inds = np.where(peakEnv)[0]
    peakEnvTimes = pe_inds/soundfs
    peakEnvValues = peakEnv[pe_inds]

    pr_inds = np.where(peakRate)[0]
    peakRateTimes = pr_inds/soundfs
    peakRateValues = peakRate[pr_inds]*soundfs/envfs

    return peakRateTimes, peakRateValues, peakEnvTimes, peakEnvValues

