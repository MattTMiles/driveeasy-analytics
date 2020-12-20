from scipy import signal
import numpy as np

def get_top_peaks(peaks, prominence, topK=2):
    if len(prominence) >= topK:
        sorted_prominence = sorted([(p, i) for i, p in enumerate(prominence)], reverse=True)

        top_peak_index = [sorted_prominence[i][1] for i in range(topK)]
        top_peaks = [peaks[p] for p in top_peak_index]
        return top_peaks
    else:
        print(f'extracted peaks less than {topK}')
        return list(range(topK))

def find_topk_peaks(arr, topK=2, height=0.1, distance=10):
    peaks, peak_props = signal.find_peaks(arr, height=height, distance=distance)
    if len(peaks) <= topK:
        return peaks

    peak_heights = peak_props['peak_heights']
    heights2idx = [(ph, idx) for idx, ph in enumerate(peak_heights)]
    sorted_heights = sorted(heights2idx, reverse=True)
    topK_heights = sorted_heights[:topK]
    topK_idx = sorted([idx for height, idx in topK_heights])
    topk_peaks = np.array([peaks[idx] for idx in topK_idx])
    # TODO: change datatype to int32
    return topk_peaks