# *****************************************************************************
#  @file low_variance.py
#  
#  @date:   14/05/2021
#  @author: Johan Mendez
#  @email:  johan.mendez@dtabiz.co
#  @status: Debug
# 
#  @brief
#  
# 
# 
#  @detail
# 
#
# 
#  *****************************************************************************

from sklearn.feature_selection import VarianceThreshold

def variance_threshold_selector(data, threshold=0.5):
    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


