"""qCH beat detection method library

Houses modules for IIQR beat rejection algorithm, for details see
Objective assessment of beat quality in transcranial doppler measurement of
blood flow velocity in cerebral arteries K Jalaleddini, N Canac, SG Thorpe,
MJ O’Brien, M Ranjbaran, B Delay, et. al IEEE Transactions on Biomedical
Engineering 67 (3), 883-892


# NOTES
# ----------------------------------------------------------------------------|


"""


# # Imports
# -----------------------------------------------------|
from copy import copy
import numpy as np
from scipy.spatial.distance import euclidean


# # Main Beat Reject CLass
# -----------------------------------------------------|
class BeatReject:
    #  pylint: disable=too-few-public-methods
    """
    Class for beat rejection.

    Attributes:
        threshold(float): Inter-Quartile-Range percentile threshold
                          for outlier detection.
        reject_methods(list): List of strings for analyzer names
        env_detrend(bool): if True, de-trend envelope data
        time
        sample_rate
        beat_methods
    """

    def __init__(self, **params):
        """
        Initalize.

        Args:
            params (dictionary): Dictionary of experiment params as keywords.
        """
        self.sample_rate = params.get('sample_rate', 125.)
        self.reject_methods = params.get("reject_methods", ["cc"])
        self.detrend = params.get("detrend", False)
        self.threshold = params.get("threshold", 1.5)
        self.params = params

    def reject_beats(self, envelope, beats):
        """Reject poor quality beats.

        Args:
            env (list): TCD envelope data
            beats (list): list of tuples in the form of (start, stop)
                          beat indices
        Return:
            (Tuple): list of rejected and accepted beats
        """
        rej_idx = []
        for method in self.reject_methods:
            rej_idx.extend(self._reject_beat_method(envelope, beats, method))
        accept = [x for idx, x in enumerate(beats) if idx not in rej_idx]
        reject = [x for idx, x in enumerate(beats) if idx in rej_idx]
        return reject, accept

    def _reject_beat_method(self, env, all_beats, method):
        #  pylint: disable=too-many-locals,
        """
            Identify poor quality beats for a method using IIQR technique.

            Args:
                env (list): TCD envelope data
                all_beats (list): list of tuples in the form of (start, stop)
                beat indices
                method (string): method name.
            Return:
                list of indices of the rejected beat number
        """
        analyzer = BeatAnalyzerFactory.get_analyzer(method)
        accept_beats = copy(all_beats)
        nbl = NormalizeBeatLength(env, accept_beats, detrend=self.detrend)
        bts_eq_len = nbl.normalize_beats()
        all_beat_idx, all_reject_idx = range(bts_eq_len.shape[0]), []
        # # main IIQR loop
        continue_iqr = True
        while continue_iqr:
            # # get retained beats and indices
            keep_idx = np.setdiff1d(all_beat_idx, all_reject_idx)
            bts_eq = bts_eq_len[keep_idx, :]
            # # get rejection features
            features = self._get_features(env, analyzer, accept_beats, bts_eq)
            if np.isnan(features).all():
                return all_reject_idx
            # # Get new reject index candidate
            reject_idx, diff = self._get_rej_idx(features, analyzer['side'])
            # # check/reject/continue logic
            if diff > 0:
                beat_2_reject = accept_beats[reject_idx]
                del accept_beats[reject_idx]
                original_reject_idx = all_beats.index(beat_2_reject)
                all_reject_idx.append(original_reject_idx)
            else:
                continue_iqr = False
        return all_reject_idx

    def _get_features(self, env, analyzer, accept_beats, bts_eq):
        """ compute average beat waveform and return the rejection features
        """
        bt_avg = np.nanmean(bts_eq, axis=0)
        anlyzr_fcn = getattr(self, "_anlyzr_{}".format(analyzer["method"]))
        features = np.zeros(len(accept_beats))
        for bt_id, beat in enumerate(accept_beats):
            features[bt_id] = anlyzr_fcn(bt_orignl=env[beat[0]:beat[1]],
                                         bt_eq_len=bts_eq[bt_id],
                                         bt_avg=bt_avg)
        return features

    def _get_rej_idx(self, features, side):
        #  pylint: disable=assignment-from-no-return
        """ return the reject candidate for this iteration
        """
        lower_bound, upper_bound = \
            self._get_iqr_bounds(features, thresh=self.threshold, side=side)
        diff1, diff2 = features - upper_bound, lower_bound - features
        diff = np.maximum(diff1, diff2)
        reject_idx = np.nanargmax(diff)
        return reject_idx, diff[reject_idx]

    @staticmethod
    def _anlyzr_cross_correlation(**kargs):
        """
        Max cross-correlations between a beat and the average beat waveforms.

        Required keyword arguments:
            bt_eq_len (list): list of TCD envelope velocities for the beat
            bt_avg (list): list of TCD envelope velocity of the average beat
        Return:
            (float) Maximum cross correlation between the beat and average
                beat waveform
        """
        beat, avg = kargs["bt_eq_len"], kargs["bt_avg"]
        if len(beat) != len(avg):
            error_str = "Lengths of beat and average beat are different"
            raise Exception(error_str)
        beat_nrm, avg_nrm = np.linalg.norm(beat), np.linalg.norm(avg)
        avg = avg / avg_nrm if avg_nrm > 0 else avg
        beat = beat / beat_nrm if beat_nrm > 0 else beat
        return np.max(np.correlate(beat, avg))

    @staticmethod
    def _anlyzr_euclidean_distance(**kargs):
        """
        Euclidean distance between a beat and the average beat waveforms.

        Required keyword arguments:
            bt_eq_len (list): list of TCD envelope velocities for the beat
            bt_avg (list): list of TCD envelope velocity of the average beat
        Return:
            (float) Euclidean distance between the beat and the average beat
                waveform
        """
        beat = kargs["bt_eq_len"]
        avg = kargs["bt_avg"]
        return euclidean(beat, avg)

    def _anlyzr_hi_freq_noise(self, **kargs):
        """
        Ratio of high frequency power to low frequency power in a beat.

        Required keyword arguments:
            bt_orignl (list): list of TCD envelope velocities for the beat
        Return:
            (float) Ratio of the high frequency power to low frequency power
        """
        beat = kargs["bt_orignl"]
        if len(beat) < 5:
            return np.nan
        lpf_cut_off_freq = 15
        nyq_freq = self.sample_rate / 2.0
        if nyq_freq < lpf_cut_off_freq:
            error_str = 'Sampling frequency must be larger than 30 Hz'
            raise ValueError(error_str)
        beat_fft = np.abs(np.fft.rfft(beat))
        freq_axis = np.linspace(0, nyq_freq, len(beat_fft))
        freq_index_cutoff = np.min(np.where(freq_axis > lpf_cut_off_freq))
        hf_power = np.sum(beat_fft[freq_index_cutoff:])
        # excluding 0 Hz corresponding to DC for lf_power
        lf_power = np.sum(beat_fft[1:freq_index_cutoff])
        return hf_power / lf_power

    @staticmethod
    def _anlyzr_dia_var(**kargs):
        """
        Variance of the diastolic portion of a beat.

        Required keyword arguments:
            bt_orignl (list): list of TCD envelope velocities for the beat
        Return:
            (float) Variance of the diastolic portion of the beat
        """
        beat = kargs["bt_orignl"]
        win_len = len(beat)
        diastolic_win_start = int(win_len - win_len * 0.2)
        diastolic_env = beat[diastolic_win_start:]
        return np.var(diastolic_env)

    @staticmethod
    def _anlyzr_beat_len(**kargs):
        """
        Length of a beat.

        Required keyword arguments:
            bt_orignl (list): list of TCD envelope velocities for the beat
        Return:
            (int) Number of samples in a TCD beat
        """
        beat = kargs["bt_orignl"]
        return len(beat)

    @staticmethod
    def _get_iqr_bounds(features, thresh=1.5, side="both"):
        """
        Get upper and lower bounds for outlier detection.

        Args:
            features(list): Beat features calculated using the analyzers
            thresh(float): thresholds to in the interquartile range method to
            classify a point as outlier
            side(str): Side of the distribution to use for outliers:
                "low": Identifies outliers that are ''too small''
                "high": Identifies outliers that are ''too large''
                "both": Identifies outliers that are both ''too large'' and
                     ''too small''
        Return:
            (tuple): Tuple of form (lower bound, upper bound) in the IQR method
        """
        side = side.lower()
        if side not in ["both", "low", "high"]:
            error_str = f"Wrong side:{side}. Must be 'both', 'low' or 'high'"
            raise ValueError(error_str)
        if not np.isnan(features).all():
            thresh = float(thresh)
            low = side in ["both", "low"]
            high = side in ["both", "high"]
            quart_1, quart_3 = np.nanpercentile(features, [25, 75])
            iqr = quart_3 - quart_1
            lower_bound = quart_1 - (iqr * thresh) if low else \
                np.nanmin(features)
            upper_bound = quart_3 + (iqr * thresh) if high \
                else np.nanmax(features)
            return lower_bound, upper_bound
        return np.nan, np.nan


# # Beat Length Normalizer Class
# -----------------------------------------------------|
class NormalizeBeatLength:
    #  pylint: disable=too-few-public-methods
    """
    Normalize beat length in a TCD recording.

    Attributes:
        detrend(bool) : If True, data will be de-trended by removing
            an 8th order polynomial trend fitted to the data
        env(list): TCD envelope data
        beats(list): list of tuples in the form of (start, end)
            beat indices

    """
    def __init__(self, env, beats, detrend=False):
        """
        Initialize.

        Args:
            env (list): TCD envelope data
            beats (list): list of tuples in the form of
                          (start index, stop index) of beats
            detrend (bool): If True, de-trend envelope by removing an 8th
                            order power polynomials trend
        """
        if not isinstance(detrend, bool):
            error_str = "Detrend must be of type boolean"
            raise TypeError(error_str)
        n_samp = len(env)
        bt_idx = [x[0] for x in beats]
        bt_idx.extend([x[1] for x in beats])
        bt_max = max(bt_idx)
        bt_min = min(bt_idx)
        if (bt_max > n_samp) or (bt_min < 0):
            error_str = "Invalid beat indices"
            raise ValueError(error_str)
        self.detrend = detrend
        self.env = env
        self.beats = beats

    def normalize_beats(self):
        """
        Normalize beat lengths.

        Trend removal approach:
            Removes trend if data are non-stationary. Note non-stationarity is
            not necessarily undesirable (e.g. breath holding). So, detrending
            is needed to avoid unncessary triggering of beat rejection.
        """
        if self.detrend:
            x_env = np.arange(len(self.env))
            coefs = np.polyfit(x_env, self.env, 8)
            trend = np.polyval(coefs, x_env)
        else:
            trend = np.zeros_like(self.env)
        len_avg = self._get_beat_avg_len(self.beats)
        return self._eq_beat_len(self.beats, self.env - trend, len_avg)

    @staticmethod
    def _eq_beat_len(beats, env, bt_len):
        """
        Equalize beat lengths by truncating too long & padding too short beats.

        Args:
            beats(list): list of tuples in the form of (start, stop) indices
            env(list): TCD data envelope
            bt_len(int): length of desired beat length (samples)
        """
        num_beats = len(beats)
        # Get average beat length
        normalized_beats = np.zeros((num_beats, bt_len))
        # Normalize each beat len to the average beat len
        for idx, beat in enumerate(beats):
            this_bt_len = beat[1] - beat[0]
            if this_bt_len >= bt_len:
                # truncate beat if beat len longer than average beat len
                beat_normalized = env[beat[0]: beat[0] + bt_len]
            elif this_bt_len < bt_len:
                # hold last value if beat len shorter than average beat len
                beat_normalized = env[beat[0]: beat[0] + this_bt_len]
                this_bt_len = len(beat_normalized)
                beat_normalized = \
                    np.concatenate((beat_normalized,
                                    beat_normalized[-1] *
                                    np.ones(bt_len - this_bt_len)), 0)
            normalized_beats[idx, :] = beat_normalized
        return normalized_beats

    @staticmethod
    def _get_beat_avg_len(beats):
        """
        Average beat length based on the median of individual beat lengths.

        Args:
            beats(list): list of tuples in the form of (start, end) indices
        Returns:
            (int): average beat length
        """
        beat_len = [x[1] - x[0] for x in beats]
        return int(round(np.nanmedian(beat_len)))


# # Beat Analyzer Factory Class
# -----------------------------------------------------|
class BeatAnalyzerFactory:
    #  pylint:disable=too-few-public-methods
    """ Factory for beat rejection analyzers.
    """
    factory = \
        {
            "cc": {"method": "cross_correlation", "side": "low"},
            "ed": {"method": "euclidean_distance", "side": "high"},
            "len": {"method": "beat_len", "side": "both"},
            "hfn": {"method": "hi_freq_noise", "side": "high"},
            "dv": {"method": "dia_var", "side": "high"}}

    @classmethod
    def get_analyzer(cls, method):
        """
        Get analyzer info.

        Args:
            method(str): name of the rejection analyzer method
        Returns:
            (dict): dictionary with keys as method and side and values as
                method str name and outlier detection side
        """
        if method in cls.factory.keys():
            return cls.factory[method]
        error_str = f"{method} is not implemented"
        raise NotImplementedError(error_str)
