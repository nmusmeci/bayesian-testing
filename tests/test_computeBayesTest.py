import pytest
from bayes import computeBayesTest

def test_monotonicity_rates_v1():
    _,P = computeBayesTest([0.2,0.1],1000)
    assert P.loc["A","B"] > 0.5

def test_monotonicity_rates_v2():
    _,P1 = computeBayesTest([0.9,0.1],1000)
    _,P2 = computeBayesTest([0.2,0.1],1000)
    assert P1.loc["A","B"] >= P2.loc["A","B"]

def test_monotonicity_volumes():
    _,P1 = computeBayesTest([0.2,0.1],1000)
    _,P2 = computeBayesTest([0.2,0.1],100)
    assert P1.loc["A","B"] >= P2.loc["A","B"]

def test_monotonicity_confidence():
    C1,_ = computeBayesTest([0.2,0.1],1000,credible_interval_prob=80.)
    C2,_ = computeBayesTest([0.2,0.1],1000,credible_interval_prob=90.)
    assert C2.loc["A","B"][1]-C2.loc["A","B"][0] >= C1.loc["A","B"][1]-C1.loc["A","B"][0]