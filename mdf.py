""" moving difference filter beat detection algorithm

Algorithm for detecting arterial blood pressure pulse onset based on Zong et.
al. (An Open Source Algorithm to Detect Onset of Arterial Blood Pressure
Pulses). Uses a low-pass filter, moving difference filter, and decision rule
to find beat onsets.


# NOTES
# ----------------------------------------------------------------------------|


"""


# # Imports
# -----------------------------------------------------|
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import correlation


def moving_diff_filt(data, f_s, window):
    """Apply a windowed and weighted moving difference filter to the data. The
    moving difference filter is defined as follows:

    z_i = sum_{k=i-w}^{i} y_k - y_{k-1}
    # should be: z_i = y_i - y_{i-w} for all i >= w where 
    # w is window size computed by int(f_s * window)

    Parameters
    ----------
    window : int
        Window size in seconds
    """
    wind = int(f_s * window)
    mdf = np.zeros_like(data)
    mdf[wind:] = data[wind:] - data[:-wind]
    return mdf


def get_mean_beat(data, starts):
    """Get the average beat waveform.
    """
    # Create a list of start and stop indices for each beat
    beats = [data[starts[i]:starts[i + 1]] for i in range(len(starts) - 1)]
    # Get the size of the median beat
    size = int(np.median([len(beat) for beat in beats]))
    # Average all beats. Truncate beats that are longer than median and
    # pad beats that are shorter than median.
    avg_beat = np.zeros(size)
    for beat in beats:
        norm_beat = np.zeros(size)
        end = min(len(beat), size)
        norm_beat[:end] = beat[:end]
        if len(beat) < size:
            norm_beat[end:] = beat[-1]
        avg_beat += norm_beat
    mean_beat = avg_beat / len(beats)
    return mean_beat


def find_onset(onset_seg, peak_seg, thresh, last_onset=None):
    """Find the onset and peak positions in a segment of data by finding the
    valley which most closely precedes the detected peak such that the
    difference between the peak value and onset value is greater than some
    provided threshold.

    Parameters
    ----------
    onset_seg : 1D array-like
        Data segment to search for onset
    peak_seg : 1D array-like
        Data segment to search for peak
    thresh : float
        Difference between onset and peak must be greater than this threshold
    last_onset : float
        Characterizes the value of previous onsets
    """
    # qch_logger.info('Calculating beat onset.')
    # First locate the probable peak by finding the maximum of the segment
    peak_pos = np.argmax(peak_seg)
    # Grab valleys to use as candidate onsets
    valleys = detect_peaks(onset_seg, edge='falling', valley=True)
    # Since in general there will be multiple valleys, the logic below is
    # responsible for determining which of these valleys is most likely to
    # be the actual onset.
    # First, find valley closest to peak such that the difference between them
    # is greater than the provided threshold
    onset = None
    for i, valley in enumerate(valleys[::-1]):
        if peak_seg[peak_pos] - onset_seg[valley] > thresh:
            if onset is None:
                onset = valley
                if last_onset is None:
                    break
            # Also want to make sure that the new onset looks similar to onsets
            # immediately preceding it. So if the current onset is far from
            # previous onsets and a new valley is found that is much closer,
            # then mark it as the onset instead.
            elif (abs(last_onset - onset_seg[onset]) > thresh and
                  abs(last_onset - onset_seg[valley]) < 0.25 * thresh):
                onset = valley
    return onset, peak_pos


def get_mad(arr):
    """Get the median absolute deviation
    """
    med = np.median(arr)
    diff = np.abs(arr - med)
    med_dev = np.median(diff)
    return 1.4826 * med_dev


def get_max_peak(data):
    """Find all the peaks in a segment of data and return the peak with the
    highest value.
    """
    pks = detect_peaks(data, edge="rising")
    if len(pks) > 0:
        return pks[np.argmax(data[pks])]
    else:
        return None


def compare_beats(beat1, beat2, metric=correlation):
    """Compare two beats using their correlation distance. Beats can either be
    resampled and scaled or truncated in order to match their lengths for
    comparison.
    """
    arr1 = np.array(beat1)
    arr2 = np.array(beat2)
    idx = min(len(arr1), len(arr2))
    dist = metric(arr1[:idx], arr2[:idx])
    return dist


def handle_long_beats(beats, peaks, mdf_peaks, data, f_s, window,
                      thresh_max=0.6, thresh_min=0.35, refractory_period=0.2,
                      long_thresh=3.0):
    """Identify long beats based on outlier detection. Long beats likely
    contain missed beats, so use a refined local search within these beats to
    search for missed beats.
    """
    # Make copies of lists to modify
    beats_cp = list(beats)
    peaks_cp = list(peaks)

    # Calculate differences
    beats_arr = np.array(beats_cp)
    diffs = beats_arr[1:] - beats_arr[:-1]
    med_length = np.median(diffs)

    # Get the mean beat commenting because currently unused
    # mean_beat = get_mean_beat(data, beats_cp)

    # Method for determining a long beat
    upper_bound = med_length + long_thresh * get_mad(diffs)

    # Deal with long beats by applying global beat detection algorithm locally
    # but with an incrementally reduced threshold value.
    loong = np.where(diffs > upper_bound)[0]
    for beat in loong:
        # Search data segment from corresponding peak to start of next beat
        segment = data[peaks_cp[beat]:beats_cp[beat + 1]]
        seg_mdf = moving_diff_filt(segment, f_s, window)
        thresh_norms = np.linspace(thresh_min, thresh_max, 6)[::-1]
        for thresh_norm in thresh_norms:
            new_beats, new_peaks, _ = \
                decision_rule(seg_mdf, segment, f_s,
                              refractory_period=refractory_period,
                              thresh_abs=thresh_norm * np.mean(mdf_peaks))
            # Add the offset to the newly found beats and break out of loop
            if len(new_beats) > 0:
                new_beats = [idx + peaks_cp[beat] for idx in new_beats]
                new_peaks = [idx + peaks_cp[beat] for idx in new_peaks]
                break

        beats_cp += new_beats
        peaks_cp += new_peaks

    # Sort lists with newly added beats
    beats_cp.sort()
    peaks_cp.sort()
    return beats_cp, peaks_cp


def handle_short_beats(beats, peaks, data, corr_thresh=0.2, dist_cutoff=0.7,
                       short_thresh=3.5):
    """Identify short beats based on outlier detection. Short beats are likely
    due to noise artifacts and may need to be merged with one of their
    neighboring beats.
    """
    # Make copies of lists to modify
    beats_cp = list(beats)
    peaks_cp = list(peaks)

    # Calculate differences
    beats_arr = np.array(beats_cp)
    diffs = beats_arr[1:] - beats_arr[:-1]
    med_length = np.median(diffs)

    # Get the mean beat
    mean_beat = get_mean_beat(data, beats_cp)

    # Method for determining a short beat
    lower_bound = med_length - short_thresh * get_mad(diffs)

    # Deal with short beats by deleting a beat in such a way as to try to
    # minimize the difference between the local beat lengths and the median
    # beat length:
    # Examine short beats in reverse order. First try to see if combining with
    # proceeding beat makes sense, and do it if so. If not, see if it makes
    # sense to combine with preceeding beat. If not, do nothing.
    short = np.where(diffs < lower_bound)[0]
    for beat in short[::-1]:
        if beat + 2 <= len(beats_cp) - 1:
            after = beats_cp[beat + 2] - beats_cp[beat + 1]
        else:
            after = None
        length = beats_cp[beat + 1] - beats_cp[beat]
        if beat - 1 >= 0:
            before = beats_cp[beat] - beats_cp[beat - 1]
        else:
            before = None
        beat1 = data[beats_cp[beat]:beats_cp[beat + 1]]
        dist = compare_beats(beat1, mean_beat)
        if (after is not None and
           abs(length + after - med_length) < abs(after - med_length)):
            if dist > corr_thresh:
                del beats_cp[beat + 1]
                del peaks_cp[beat + 1]
        elif (before is not None and
              abs(length + before - med_length) < abs(before - med_length)):
            if dist > corr_thresh:
                del beats_cp[beat]
                del peaks_cp[beat]
        elif dist > dist_cutoff:  # Delete short, significantly different beat
            del beats_cp[beat]
            del peaks_cp[beat]
    return beats_cp, peaks_cp


def align_beats(data, beats, wind, thresh):
    """Align beat starts by shifting start on either side to maximize dot
    product with mean beat.
    """
    mean_beat = get_mean_beat(data, beats)
    aligned_beats = []
    before = [beats[0]] + list(np.diff(beats))
    after = list(np.diff(beats)) + [len(data) - beats[-1]]
    for start, bef, aft in zip(beats, before, after):
        # If beats are very short for some reason, window can overlap with
        # start of next beat. To prevent this, only test for alignment
        # extending 1/3 of way into previous/following beat.
        beg = max(0, start - min(wind, int(bef / 3)))
        end = min(len(data), start + min(wind, int(aft / 3)) + wind)
        if (end - beg) > wind:
            lag = get_lag(mean_beat[:wind], data[beg:end], start - beg)
        else:
            lag = 0
        if lag < thresh:
            aligned_beats.append(start)
        else:
            aligned_beats.append(start + lag)
    return aligned_beats


def get_lag(seg1, seg2, start):
    """Figure out the "lag time" between two signals using dot product as a
    measure of similarity
    """
    norm_seg1 = seg1 / norm(seg1)
    wind = len(norm_seg1)
    sim = []  # Similarity metric values
    for i in range(len(seg2) - wind):
        norm_seg2 = seg2[i:i + wind] / norm(seg2[i:i + wind])
        sim.append(np.dot(norm_seg1, norm_seg2))
    return np.argmax(sim) - start


def decision_rule(mdf, data, f_s, refractory_period=0.2, init_wind=10,
                  thresh_norm=0.6, thresh_abs=None, num_peaks=20,
                  onset_thresh=0.75):
    """Scan through the moving difference filter and identify windows where an
       onset is likely to be. Then search within this window for the beat onset

    Parameters
    ----------
    mdf : 1D array
        Data after applying the moving difference filter
    data : 1D array
        Low-pass filtered data
    f_s : float
        Sampling rate
    refractory_period: float
        Minimum time between successive beats
    init_wind : int
        The initial window to average over to determine the starting threshold
        given in seconds
    thresh_abs : float
        If not None, this will be used instead of an adaptive threshold
    """
    refractory_period = int(f_s * refractory_period)
    init_wind = min(int(f_s * init_wind), len(mdf))
    search_wind = refractory_period
    beats = []  # Positions of beat onsets
    peaks = []  # Positions of corresponding peaks
    # rises = []  # Height of peak relative to onset UNUSED
    mdf_peaks = []
    refract = 0
    crossed = True
    maxima = detect_peaks(mdf[:init_wind],
                          mph=3.0 * np.mean(mdf[:init_wind].clip(min=0)))
    if thresh_abs is None:
        init_thresh = thresh_norm * np.median(mdf[maxima])
        thresh = init_thresh
    else:
        thresh = thresh_abs
    for i, val in enumerate(mdf):
        if val < thresh:
            crossed = True
        if refract <= 0 and val > thresh and crossed:
            crossed = False
            start = max(i, 0)
            stop = min(i + search_wind, len(mdf))
            max_val = np.max(mdf[start:stop])
            # Search for onset between last peak and threshold crossing point
            last_peak = peaks[-1] if peaks else 0
            last_onset = np.mean(data[beats][-5:]) if len(beats) > 0 else None
            onset, peak = find_onset(data[last_peak:i], data[start:stop],
                                     onset_thresh * max_val,
                                     last_onset=last_onset)

            if onset is not None:
                beats.append(onset + last_peak)
                peaks.append(peak + start)
                mdf_peaks.append(max_val)
                refract = refractory_period
                if thresh_abs is None and len(mdf_peaks) >= 5:
                    thresh = thresh_norm * np.mean(mdf_peaks[-num_peaks:])

        refract -= 1
    return beats, peaks, mdf_peaks


def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising', kpsh=False,
                 valley=False):
    """
    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.4"
    __license__ = "MIT"

    Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    See this IPython Notebook [1]_.

    References
    ----------
    [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/
        DetectPeaks.ipynb

    Examples
    --------
    from detect_peaks import detect_peaks
    x = np.random.randn(100)
    # detect all peaks and plot data
    ind = detect_peaks(x, show=True)
    print(ind)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # set minimum peak height = 0 and minimum peak distance = 20
    detect_peaks(x, mph=0, mpd=20, show=True)

    x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    # detection of valleys instead of peaks
    detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    x = [0, 1, 1, 0, 1, 1, 0]
    # detect both edges
    detect_peaks(x, edge='both', show=True)

    x = [-2, 1, -2, 2, 1, 1, 3, 0]
    # set threshold = 2
    detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # deal with NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) &
                           (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) &
                           (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # deal with NaN's
    if indnan.size:
        ind = ind[np.in1d(ind, indnan, invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1],
                    x[ind] - x[ind + 1]]),
                    axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])
    return ind


def mdf(data, f_s, window=0.15, refractory_period=0.2, max_iter=10,
        thresh_norm=0.6, thresh_min=0.35, num_peaks=20, onset_thresh=0.75,
        corr_thresh=0.2, dist_cutoff=0.7, short_thresh=3.5, long_thresh=3.0):
    """Run the moving difference filter (MDF) algorithm.

    Parameters
    ----------
    f_s : float
        Sampling rate in Hz
    window : float
        Window size for moving difference filter in seconds
    """
    # qch_logger.info('Calculating moving difference filtered signal.')
    # Calculate moving difference filtered signal
    mdf = moving_diff_filt(data, f_s, window)

    # First pass through signal to find beats
    beats, peaks, mdf_peaks = decision_rule(mdf, data, f_s,
                                            refractory_period=refractory_period,
                                            init_wind=10,
                                            thresh_norm=thresh_norm,
                                            num_peaks=num_peaks,
                                            onset_thresh=onset_thresh)

    if len(beats) <= 1:
        return beats

    # Perform beat length analysis to identify likely false positives/negatives
    # Iterate until no more outliers are found or until max iterations reached
    beats2, peaks2 = handle_long_beats(beats, peaks, mdf_peaks, data, f_s,
                                       window, thresh_max=thresh_norm,
                                       thresh_min=thresh_min,
                                       refractory_period=refractory_period,
                                       long_thresh=long_thresh)
    beats2, peaks2 = handle_short_beats(beats2, peaks2, data,
                                        corr_thresh=corr_thresh,
                                        dist_cutoff=dist_cutoff,
                                        short_thresh=3.5)

    while max_iter > 0 and beats2 != beats and len(beats2) > 1:
        beats = beats2
        peaks = peaks2
        beats2, peaks2 = handle_long_beats(beats, peaks, mdf_peaks, data, f_s,
                                           window, thresh_max=thresh_norm,
                                           thresh_min=thresh_min,
                                           refractory_period=refractory_period,
                                           long_thresh=long_thresh)
        beats2, peaks2 = handle_short_beats(beats2, peaks2, data,
                                            corr_thresh=corr_thresh,
                                            dist_cutoff=dist_cutoff,
                                            short_thresh=3.5)

        max_iter -= 1

    # Align beats
    wind = int(refractory_period * f_s)
    beats = align_beats(data, beats, wind, 0.03 * f_s)

    return beats
