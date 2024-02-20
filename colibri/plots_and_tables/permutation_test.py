from scipy.stats import permutation_test
import numpy as np
import matplotlib.pyplot as ptl

from reportengine.figure import figure


@figure
def plot_perm_test(x,y,**kwargs):
    """
    Plot the permutation test for two independent samples.
    
    Parameters
    ----------
    x: np.array
        First sample of vectors.
    
    y: np.array
        Second sample of vectors.
    
    kwargs: dict
        Keyword arguments for the permutation_test function from scipy.stats.
    """
    res = permutation_test((x,y),**kwargs)
    
    null_distribution = res.null_distribution
    
    p_value = res.pvalue.mean()
    ptl.plot(null_distributions, bins=20, alpha=0.5, label='x')
    ptl.hist(y, bins=20, alpha=0.5, label='y')
    ptl.legend()
    ptl.title(f'Permutation test p-value: {p_value}')
    ptl.show()

    