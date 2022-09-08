"""qCH beat detection method library

Houses methods for detecting arterial blood pressure pulse onset.
The primary mdf method uses a low-pass filter, moving difference filter, and
decision rule to find beat onsets. See details in Canac et al."Algorithm for
Reliable Detection of Pulse Onsets in Cerebral Blood Flow Velocity Signals."
Frontiers in Neurology 10 (2019). informed also by details found in on Zong et.
al. An Open Source Algorithm to Detect Onset of Arterial Blood Pressure Pulses.


# NOTES
# ----------------------------------------------------------------------------|


"""


# # Imports
# -----------------------------------------------------|
from scipy.signal import butter, filtfilt
import mdf

# # Primary Beat Detector Class
# -----------------------------------------------------|
class DetectBeats(object):
    """This is the main object for beats processing."""
    def __init__(self, params):
        """Initialize

        Args:
            params (dictionary): Dictionary of experiment params as keywords.
        """
        self.params = params
        self.sample_rate = params['sample_rate']

    def _pre_filter(self, filter_type, signal):
        """Define the pre-filtering type.

        Args:
            filter_type (string): One of filtfilt, fft, butterworth or fir,
                corresponding to scipy signal filters.

        Returns:
            Filtered signal.
        """
        if filter_type.lower() == "filtfilt":
            return self._filtfilt(signal)
        else:
            error_str = f'Bad pre_filter_type: {filter_type}'
            raise DetectBeatsException(error_str)

    def _filtfilt(self, signal):
        """Scipy's filtfilt signal processing.  This has no phase shift.
        The order and lowpass cutoff are defined in the params dictionary.

        Args:
            signal (1d-array): Signal to be filtered.

        Returns:
            filtfilt filtered signal
        """
        nyquist = self.sample_rate / 2.0
        order = self.params["filtfilt_params"]["order"]
        # cutoff value is in fractions of nyquist
        cutoff = self.params["filtfilt_params"]["lowpass"] / nyquist
        numcoeff, denumcoeff = butter(order, cutoff, analog=False)
        signal_filt = filtfilt(numcoeff, denumcoeff, signal)
        if "highpass" in self.params["filtfilt_params"]:
            cutoff = self.params["filtfilt_params"]["highpass"] / nyquist
            numcoeff, denumcoeff = \
                butter(order, cutoff, btype="high", analog=False)
            signal_filt = filtfilt(numcoeff, denumcoeff, signal_filt)
        return signal_filt

    def _get_starts_mdf(self, data, **kwargs):
        """Calls the mdf algorithm to calculate beat start indices.
        """
        starts = mdf.mdf(data, self.sample_rate, **kwargs)
        return starts

    def _get_possible_starts(self, data):
        """Gets possible beat valleys.

        Args:
            data (array): Timeseries data.
        Returns:
            starts (list): The possible beat start indeces within ``data``.
        """
        beat_algos = {"mdf": self._get_starts_mdf}
        # Determine which beat detection algorithm to use
        beat_algo = self.params.get("beat_algo", "mdf")
        kwargs = self.params.get("beat_algo_kwargs", {})
        # Run the beat detection algorithm
        if beat_algo not in beat_algos.keys():
            error_str = f'Bad beat detection algorithm, beat_algo : {beat_algo}'
            raise DetectBeatsException(error_str)
        starts = beat_algos[beat_algo](data, **kwargs)
        return starts

    def get_filtered_signal(self, raw_tcdy):
        return self._pre_filter(self.params['pre_filter_type'], raw_tcdy)

    def get_beat_starts(self, raw_tcdy):
        """External function used to get beat start indices from a 1-D time
        series.
        """
        if self.params.get('pre_filter_type'):
            tcdy = self._pre_filter(self.params['pre_filter_type'], raw_tcdy)
        else:
            error_str = 'pre_filter_type required'
            raise DetectBeatsException(error_str)
        starts = self._get_possible_starts(tcdy)
        return starts


# # Custom Exception Class
# -----------------------------------------------------|
DetectBeatsException = type('DetectBeatsException', (Exception,), {})
