import pytest
import numpy as np
import re
from bayes import computeBayesTest,computeBayesTestSampleSize

def empiricalRateSuccess(n,p1,p2,n_mc):
    rate_list = np.random.binomial(n/2.,p2,n_mc)/(n/2.)
    pval_list = [np.nan for i in rate_list]
    for count,rate in enumerate(rate_list):
        pval_list[count] = computeBayesTest([p1,rate],n)[1].loc["B","A"]
    return(np.mean(np.array(pval_list)>0.8))

def test_size_1():
    p1,p2 = 0.1,0.12
    n_mc = 100
    n = computeBayesTestSampleSize([p1,p2],accuracy=1)
    rate_success = empiricalRateSuccess(n,p1,p2,n_mc)
    assert ((rate_success<=0.83)and(rate_success>=0.77)),f"Rate success is {rate_success} instead of\
    0.8, with sample size of {n}"

def test_size_2():
    p1,p2 = 0.5,0.8
    n_mc = 100
    n = computeBayesTestSampleSize([p1,p2],accuracy=1)
    rate_success = empiricalRateSuccess(n,p1,p2,n_mc)
    assert ((rate_success<=0.83)and(rate_success>=0.77)),f"""Rate success is {rate_success} instead of """ + \
    f"""0.8, with sample size of {n}"""

def test_error_large_diff(capfd):
    p1,p2 = 0.1,0.9
    computeBayesTestSampleSize([p1,p2],accuracy=1)
    out, _ = capfd.readouterr()
    assert bool(re.match("""The values in the explored size range are all large enough to """ + \
        """confirm that the positive rates""",out))

def test_error_small_diff(capfd):
    p1,p2 = 0.1,0.10001
    computeBayesTestSampleSize([p1,p2],accuracy=1)
    out, _ = capfd.readouterr()
    assert bool(re.match("""The values in the explored size range are too small to confirm """,out))
