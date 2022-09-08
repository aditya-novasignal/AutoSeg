"""qCH SNR library

Houses modules for computation of Envelope Signal to Noise Ratio (SNR).
For details see Method for Modeling Residual Variance in Biomedical Signals
Applied to Transcranial Doppler Ultrasonography Waveforms
Kian Jalaleddini, Samuel G. Thorpe, Nicolas Canac, Amber Y. Dorn,
Corey M. Thibeault, Seth J. Wilk, Robert B. Hamilton
https://doi.org/10.1101/633669


# NOTES
# ----------------------------------------------------------------------------|


"""


# # Imports
# -----------------------------------------------------|
from copy import copy
import numpy as np
import scipy.signal as signal


# # SNR methods
# -----------------------------------------------------|
def snr(envelope, beats, **kwargs):
    """
    Calculate TCD envelope signal to noise ratio.

    Args:
        envelope(list): TCD envelope data
        beats(list): list of tuples in the form of (start, stop) beat indices
    Returns:
        (float): signal to noise ratio (decibel)
    """
    mean_beat = calc_mean_beat(envelope, beats)
    env_clean, env_residual = calc_env_residuals(envelope, beats, mean_beat,
                                                 **kwargs)
    return calc_snr(env_clean, env_residual, **kwargs)


def calc_env_residuals(env_noisy, beats, beat_model, **kwargs):
    """
    Calculate TCD envelope residuals.

    Residuals: difference between individual beats and the beat model.

    Args:
        env_noisy(list): TCD envelope data
        beats(list): list of tuples in the form of (start, stop) beat indices
        beat_model(list): Beat waveform model
        adjust(bool, optional): If true, then shift the beat_model to account
                                for the mean of individual beats
    Returns:
        (list, list): residuals, clean TCD envelope
    """
    adjust = kwargs.get("adjust", False)
    env_residual = np.array([])
    env_clean = np.array([])
    # Return empty lists if beat (start, stop) indices or beat model is empty
    if np.size(beats) and np.size(beat_model):
        for beat in beats:
            bt_vals = env_noisy[beat[0]:beat[1]]
            bt_len = beat[1] - beat[0]
            bt_model_resamp = signal.resample(beat_model, bt_len)
            offset = 0
            if adjust:
                offset = np.mean(bt_vals) - np.mean(bt_model_resamp)
            bt_residual = bt_vals - bt_model_resamp - offset
            env_residual = np.append(env_residual, bt_residual)
            env_clean = np.append(env_clean, bt_model_resamp)
        return env_clean, env_residual
    return [], []


def calc_power(signal_vals, remove_dc):
    """
    Calculate the power of a signal.

    Args:
        signal_vals(list): signal values
        remove_dc(bool): If true, the mean of the signal will be subtracted
                         prior to power calculation.
    """
    if remove_dc:
        signal_vals = copy(signal_vals - np.nanmean(signal_vals))
    return np.nansum(np.square(signal_vals))


def calc_snr(signal_vals, noise_vals, **kwargs):
    """
    Calculate signal-to-noise ratio.

    Args:
        signal_vals(list): Clean signal
        noise_vals(list): Noise signal
        remove_dc(bool, optional): If true, the mean values of signal and
                                   noise will be subtracted prior to SNR
                                   calculation.
    Returns:
        (float): signal-to-noise ratio in decibel (dB)
    """
    remove_dc = kwargs.get("remove_dc", True)
    signal_power = calc_power(signal_vals, remove_dc)
    noise_power = calc_power(noise_vals, remove_dc)
    # Power will always be greater than or equal to zero
    if signal_power == 0 and noise_power > 0:
        return -np.inf
    if signal_power == 0 and noise_power == 0:
        return np.nan
    if signal_power > 0 and noise_power == 0:
        return np.inf
    return 10 * np.log10(signal_power / noise_power)


def calc_mean_beat(envelope, beats):
    """
    Calculate the mean beat envelope waveform.

    Args:
        envelope(list): TCD envelope data
        beats(list): list of tuples in the form of (start, stop) beat indices
    Returns:
        (list): mean beat envelope


    This is achieved by:
    1) finding the median beat length
    2) normalizing the beat lengths:
        a) truncating beats longer than the median beat length
        b) padding beats shorter than the median beat length
    3) calculating the average of the normalized beats
    """
    if not np.size(beats):
        return []
    env_beats = [envelope[start:stop] for start, stop in beats]
    med_bt_len = int(np.median([len(beat) for beat in env_beats]))
    avg_beat = np.zeros(med_bt_len)
    for env_beat in env_beats:
        norm_beat = np.zeros(med_bt_len)
        end = min(len(env_beat), med_bt_len)
        norm_beat[:end] = env_beat[:end]
        if len(env_beat) < med_bt_len:
            norm_beat[end:] = env_beat[-1]
        avg_beat += norm_beat
    return list(avg_beat / len(beats))
