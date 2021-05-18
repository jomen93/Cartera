#  ===========================================================================
#  @file:   low_variance.py
#  @brief:  Eliminate the low variance columns given a threshold 
#  @author: Johan Mendez
#  @date:   14/05/2021
#  @email:  johan.mendez@databiz.co
#  @status: Debug
#  @detail: version 1.0
#  ===========================================================================

from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


