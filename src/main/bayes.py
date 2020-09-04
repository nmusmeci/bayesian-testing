import string
import pandas as pd
import numpy as np
from scipy import optimize
from scipy.stats import binom

def computeBayesTest(positive_rates,sample_size,prior_settings={"positive_rate_prior":None,"sample_size_prior":0},
                    credible_interval_prob=80,n_mc=10000,rand_seed=123):
    
    """Calculate probability matrix and credible interval matrix given positive rates.
    
    Given N variants in a randomized controlled experiment and their empirical positive rates 
    (i.e. proportion of positive actions, e.g. email open rates), the probability matrix P is
    a NxN matrix whose entry (i,j) is the Bayesian (posterior) probability that variant "i" has 
    higher positive rate than "j". The credible interval matrix CI is a NxN matrix whose entry
    (i,j) is the Bayesian credible interval for the ratio between the true positive rate of 
    variant "i" and the true positive rate of variant "j".
    
    Parameters
    ----------
    positive_rates : list of float
        positive_rates[i] represents a meaningful metric (e.g. email open rate or 
        cross-sell rate in a marketing campaign) measured in the ith variant of the 
        experiment. Each element of the list must be in [0,1]. 
        Must be at least of length 2.
    sample_size : int or list of int
        Number of observations in the experiment. If an integer is provided, it is interpreted as the total number of 
        observations across variants and an even split of volume among variants is assumed. If a list is provided, the i-th
        element of the list is interpreted as the volume in the i-th variant.
    prior_settings : dict of {str : float or list of float or None, str : int or list of int}, optional (default={str:None,str:0}) 
        Parameters defining the prior probability distribution.
        prior_settings["positive_rate_prior"] is a prior estimate of the 
        true rate for each variant. If a float is provided, it is assumed that the 
        prior estimate is the same for each variant.
        prior_settings["sample_size_prior"] is the sample size used to estimate
        the prior in a previous experiment. If the prior was not estimated through
        an experiment, this parameter can be used a proxy for how confident we
        are about the prior (with higher values indicating higher confidence).
    credible_interval_prob : float, optional (default=80)
        Confidence level to use in the calculation of the credible intervals.
        It must be [0,100].
    n_mc : int, optional (default=10000)
        Number of iterations to use in the Monte Carlo simulation to calculate 
        posterior probabilities and credible intervals. It must be > 0.
    rand_seed : int, optional (default=123)
        Initial seed for the random number generation in the Monte Carlo simulation.
            
    Returns
    -------
    CI : DataFrame, shape (len(positive_rates), len(positive_rates))
        CI.iloc[i,j] represents the credible interval for the ratio between 
        the positive rate in variant "i" and the positive rate in variant "j".
    P : DataFrame, shape (len(positive_rates), len(positive_rates))
        P.iloc[i,j] represents the posterior probability that the true positive rate
        in variation "i" is higher than the true positive rate in variation "j".
        
    """
    
    if not isinstance(sample_size,list):
        # Assume even volume split if sample_size is a single number 
        volume_list = [sample_size/len(positive_rates) for i in positive_rates]
    else:
        assert len(sample_size) == len(positive_rates),"List sample_size must be either an integer or a list of same length as positive_rates"
        volume_list = sample_size
        
    np.random.seed(rand_seed)
    
    # Calculate prior parameters alpha and beta given prior_settings.
    # If prior_settings has been left to default, set alpha = beta = 0 (uninformative prior)
    alpha_list = np.array([1. for i in positive_rates])
    beta_list = np.array([1. for i in positive_rates])
    if prior_settings["positive_rate_prior"] != None :
        alpha_list = alpha_list + \
                     np.array(prior_settings["positive_rate_prior"])*np.array(prior_settings["sample_size_prior"])
        beta_list = beta_list + \
                     (1. - np.array(prior_settings["positive_rate_prior"]))*np.array(prior_settings["sample_size_prior"])    
          
    # Generate n_mc Monte Carlo data points for each variant given the prior and the 
    # empirical positive rates
    samples_list = []
    for theta,alpha,beta,volume in zip(positive_rates,alpha_list,beta_list,volume_list):
        samples_list.append(np.random.beta(theta*volume+alpha,(1-theta)*volume+beta, size=n_mc))
    
    CI = pd.DataFrame([],columns=[i for i in range(len(positive_rates))],
                                        index=[i for i in range(len(positive_rates))])
    P = pd.DataFrame(np.nan,columns=[i for i in range(len(positive_rates))],
                                        index=[i for i in range(len(positive_rates))])
    
    # Calculate credible intervals and posterior probability for each pair of variants
    for count_i,i in enumerate(samples_list[:-1]):
        for count_j,j in enumerate(samples_list[1:]):
            
            if count_i == count_j + 1:
                continue
            
            ratio_array = i/j
            
            CI.loc[count_i,count_j+1] = [round(np.percentile(ratio_array,100-credible_interval_prob),2),
                                                           round(np.percentile(ratio_array,credible_interval_prob),2)]
            with np.errstate(divide='ignore'):
                CI.loc[count_j+1,count_i] = [round(1./CI.loc[count_i,count_j+1][1],2),
                                             round(1./CI.loc[count_i,count_j+1][0],2)] 
            
            P.loc[count_i,count_j+1] = round(sum(i > j)/n_mc,2)
            P.loc[count_j+1,count_i] = round(1 - P.loc[count_i,count_j+1],2)
    
    # Rename columns and index in the format 'ABCD...'
    CI.columns = [i for i in string.ascii_uppercase[:CI.shape[1]]]
    CI.index = [i for i in string.ascii_uppercase[:CI.index.shape[0]]]
    P.columns = [i for i in string.ascii_uppercase[:P.shape[1]]]
    P.index = [i for i in string.ascii_uppercase[:P.index.shape[0]]]
    
    return([CI,P])


def computeBayesTestSampleSize(true_positive_rates,min_confidence=80.,accuracy=10,
                                prior_settings={"positive_rate_prior":None,"sample_size_prior":0}):
    
    """Calculate minimum sample size required to detect a given difference between positive rates.
    
    When designing an AB or Multivariate test, you want to make sure that the sample size you will 
    collect is large enough to confidently conclude that the variations are different when their true 
    positive rates are sufficiently dissimilar.
    
    For an AB test, a way to do this within a Bayesian framework is measuring the required minimum 
    sample size that detects a difference between the two variants with at least a x% Bayesian confidence
    at least p% of the times, given the two expected positive rates (true_positive_rates).
    In this function, we chose p = 80% (fixed) whereas "x" can be selected by the user (min_confidence,
    default value is 80%).
    
    For a multivariate test, the function looks at the largest and smallest positive rates in the input
    list and then works out the sample size as in the AB test case.
    
    The function will search for a solution by using the bisection method. The initial search range
    is size_range (default (100,10000)), but if the solution is not in this range the function will
    change it accordingly up to 3 times. If no solution is found after 3 times, the function will
    return a message explaining why.
    
    Parameters
    ----------
    true_positive_rates : list of float
        List of positive rates that are expected for each variant in the experiment.
    min_confidence : float, optional (default=80.)
        Minimum Bayesian posterior probability value required for a test to be considered "passed".
    accuracy : int, optional (default=10)
        Required accuracy for the min_sample_size. Smaller values correspond to higher 
        accuracy and longer computation times.
    
    Returns
    -------
    min_sample_size : float
        The smallest sample size for which a Bayesian test would detect a difference with at
        least min_confidence Bayesian confidence at least 80% of the times, assuming that 
        true_positive_rates are the true positive rates. 
        
    """
    
    # Calculate benchmark rate as the lowest true positive rate in the list
    benchmark_rate = min(true_positive_rates)
    largest_rate = max(true_positive_rates)
    assert benchmark_rate != largest_rate,"Error: you should provide at least two different positive rates"
    
    # Define function to calculate, given a sample size, how many successes (positive events) need
    # to be observed for a Bayesian test to return a Bayesian confidence at least equal to min_confidence
    def MinNumSuccessesForOnePositiveTest(benchmark_rate,n,min_confidence=80.,prior_settings=prior_settings):
        func_confidence = lambda k: computeBayesTest(positive_rates=[benchmark_rate,2.*k/n],sample_size=n,n_mc=1000,
                                    prior_settings=prior_settings)[1].iloc[1,0]\
                                 - (min_confidence/100.)    
        a,b = 1.,np.floor(n/2.)-1.
        f_a,_ = func_confidence(a),func_confidence(b)
        
        if f_a > 0:
            return(1)
        else:
            return(optimize.bisect(func_confidence,a=a,b=b,xtol=1))  
    
    # Define function to calculate, given a sample size, whether the probability to pass the Bayesian test
    # is less (return negative value) or greater than (return positive value) 80%
    MinSampleSizeForManyPositiveTests = lambda n : \
                                        1 - binom.cdf(int(MinNumSuccessesForOnePositiveTest(benchmark_rate,n,
                                        min_confidence=min_confidence,prior_settings=prior_settings)),int(n/2.),largest_rate) - 0.8
    
    # Find zero of the function defined above by using bisection method. Recalibrate a,b if the solution is
    # not in the initial interval [100,10000]
    a,b = [100,10000]
    f_a,f_b = [MinSampleSizeForManyPositiveTests(a),MinSampleSizeForManyPositiveTests(b)]
    counter = 0
    while((np.sign(f_a*f_b) == 1)and(counter<4)and(a > 10)):
        if f_b < 0:
            a = b
            b *= 10.
        elif f_a > 0:
            b = a
            a /= 10.
       
        f_a,f_b = [MinSampleSizeForManyPositiveTests(a),MinSampleSizeForManyPositiveTests(b)]
        counter += 1
    
    # If solution is not found even after attempt of recalibration, then print message explaining reason
    # for no convergence
    if np.sign(f_a*f_b) == 1:
        
        if f_a < 0:
            print(
            'The values in the explored size range are too small to confirm that the positive rates are truly different '
            'with a {}% Bayesian confidence. The largest value in the range ({}) corresponds to a Bayesian confidence of '
            '{}%. Try smaller Bayesian confidence'.format(round(min_confidence,2),b,round(f_b*100.+80.,2)))
            return(None)
        
        if f_a > 0:            
            print(
            'The values in the explored size range are all large enough to confirm that the positive rates '
            'are truly different with a {}% Bayesian confidence. '
            'The smallest sample size ({}) corresponds to a Bayesian confidence of '
            '{}%.'.format(round(min_confidence,1),a,round(f_a*100.+80.,1)))
            
            return(None)    
    
    min_sample_size = optimize.bisect(MinSampleSizeForManyPositiveTests,a=a,b=b,xtol=accuracy)
    
    return(round(min_sample_size))
