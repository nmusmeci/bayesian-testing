# Bayesian A/B and Multivariate Testing
Python library containing a couple of functions for Bayesian A/B and Multivariate testing.
It can be used to:

1. Run a Bayesian analysis on an AB/multivariate test that you have run, returning also a Bayesian credible intervals for the relative uplift;

2. Calculate the minimum sample size required for a Bayesian AB/multivariate test, useful for designing future tests

The library is especially useful when doing repeated tests (as it avoids the multiple testing bias) and when you want to incorporate knowledge/results coming from previous tests.