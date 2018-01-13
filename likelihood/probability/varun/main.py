# importing libraries
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn

'''
This method gives the loglikelihood per row using the computed values of Beta0 
and Beta1
'''
def compute_bayesean_likelihood(features, mean_vector, var_vector, beta, index):
    # null check
    if (features is None or mean_vector is None or var_vector is None):
        raise ValueError('Input data is None')

    # data validation
    if (len(features) != len(mean_vector) or len(features) != len(var_vector)):
        raise ValueError('Input data is not of valid size')

    pdf_list = np.empty([1, len(features)])
    log_conditional_prob = 0
    # Iterate each value and compute PDF
    for i in range(0, len(features)):
        # compute probability mass function
        if (i == index):
            temp1 = (-0.5 * math.log(2 * math.pi * var_vector[i]))
            temp2 = (-0.5) * (1 / var_vector[i]) * ((beta[0] + beta[1] *
                                         features[0] - features[i]) ** 2)
            log_conditional_prob = temp1 + temp2
            pdf_list[0][i] = 1
            continue

        temp = (features[i] - mean_vector[i]) ** 2
        temp = temp / var_matrix[i]
        multiplier = 1 / (math.sqrt(2 * math.pi * var_vector[i]))
        exponent = math.exp((-0.5) * temp)
        pdf_list[0][i] = multiplier * exponent

    # compute log likelihood
    loglikelihood = math.log(np.prod(pdf_list)) + log_conditional_prob

    return loglikelihood


'''
The method assumes that each feature is normally distributed and all features are 
independent of each other.

The log likelihood is computed by taking a logarithm over the
product of the probability mass function of each value in the vector

@:param features - vector of size 1 x 4,
                   where each column value represents the value of one feature 
                   !null
@:param mean_vector - mean vector of size 1 x 4
                      where each column value represents the value of one feature
                      !null
@:param std_vector - standard deviation of size 1 x 4
                    where each column value represents the value of one feature
                    !null
'''
def compute_independent_loglikelihood(features, mean_vector, var_vector):
    # null check
    if (features is None or mean_vector is None or var_vector is None):
        raise ValueError('Input data is None')

    # data validation
    if (len(features) != len(mean_vector) or len(features) != len(var_vector)):
        raise ValueError('Input data is not of valid size')

    pdf_list = np.empty([1, len(features)])
    # Iterate each value and compute PDF
    for i in range(0, len(features)):
        # compute probability mass function
        temp = (features[i] - mean_vector[i]) ** 2
        temp = temp / var_matrix[i]
        multiplier = 1 / (math.sqrt(2 * math.pi * var_vector[i]))
        exponent = math.exp((-0.5) * temp)
        pdf_list[0][i] = multiplier * exponent

    # compute log likelihood
    return math.log(np.prod(pdf_list))


'''
The method assumes that each feature is normally distributed and features have 
dependency over one another.

The log likelihood is computed by taking a logarithm over the
probability density function of the feature vector.

@:param feature_vector - vector of size 1 x 4,
                         where each column value represents the value of one 
                         feature 
                         !null
@:param mean_vector - mean vector of size 1 x 4 where each column value represents
                      the value of one feature
                      !null
@:param covariance_matrix - Covariance matrix of 4 X 4 size.
                            !null
'''
def compute_multi_loglikelihood(feature_vector, mean_vector, covariance_matrix):
    # null check
    if (
                feature_vector is None or mean_vector is None or covariance_matrix is None):
        raise ValueError('Input data is None')

    # data validation
    if (len(feature_vector) != len(mean_vector) or len(feature_vector) != len(
            covariance_matrix)):
        raise ValueError('Input data is not of valid size')

    # compute determinant of covariance matrix
    covariance_determinant = np.linalg.det(covariance_matrix)

    # calculate the prefix before exponent
    prefix = (2 * math.pi) ** (len(feature_vector) / 2)
    prefix *= math.sqrt(covariance_determinant)
    prefix = (1 / prefix)

    # calculate exponent
    diff_mean = np.subtract(feature_vector, mean_vector)
    exponent = np.dot(diff_mean,
                      np.linalg.inv(covariance_matrix))
    exponent = np.dot(exponent, diff_mean.reshape((-1, 1)))
    exponent = np.multiply(exponent, (-1 / 2))
    exponent = math.exp(exponent)

    # compute log likelihood
    return math.log(prefix * exponent)


'''
This project is to understand the basic probability concepts
used for machine learning. To understand the concept of Gaussian
distribution, log likelihood and Baysean networks

@author: Varun Jain
@UBIT Number: 50247176 
'''

print('UBitName = varunjai')
print('personNumber = 50247176')
print('UBitName = dilipred')
print('personNumber = 50248867')

df = pd.read_excel(
    'C://dump//my-space//TestPython//resources//university data.xlsx')
df.drop(['rank', 'name', 'Grad Student No.',
         'TT Faculty', 'Lecturers', 'G-TT Ratio', 'G-TTL Ratio'], axis=1,
        inplace=True)
df.dropna(inplace=True)
dataset = df.as_matrix()

# compute mean, variance and standard deviation
# Axis zero signified taking values across a column in the dataset
mean_matrix = np.around(np.mean(dataset, axis=0), 3)
var_matrix = np.around(np.var(dataset, axis=0), 3)
sigma_matrix = np.around(np.std(dataset, axis=0), 3)

# printing computed values
mu1 = mean_matrix[0]
mu2 = mean_matrix[1]
mu3 = mean_matrix[2]
mu4 = mean_matrix[3]

var1 = var_matrix[0]
var2 = var_matrix[1]
var3 = var_matrix[2]
var4 = var_matrix[3]

sigma1 = sigma_matrix[0]
sigma2 = sigma_matrix[1]
sigma3 = sigma_matrix[2]
sigma4 = sigma_matrix[3]

print('mu1', mu1)
print('mu2', mu2)
print('mu3', mu3)
print('mu4', mu4)
print('var1', var1)
print('var2', var2)
print('var3', var3)
print('var4', var4)
print('sigma1', sigma1)
print('sigma2', sigma2)
print('sigma3', sigma3)
print('sigma4', sigma4)

# compute covariance and correlation
# we will transpose the matrix to enable correlation across the four features
covarianceMat = np.around(np.cov(np.transpose(dataset)), 3)
correlationMat = np.around(np.corrcoef(np.transpose(dataset)), 3)

print('covarianceMat =')
print(covarianceMat)
print('correlationMat = ')
print(correlationMat)

# pairwise plotting of each variable
# using seaborn library
graph = seaborn.pairplot(df, kind='reg')
plt.show(graph)

# compute log likelihood
'''
Here each university is a hypothesis and we will calculate likelihood of
occurrence of each university based on the value of the 4 features passed.

We then compute the single log likelihood for the entire dataset which can
then be used in future predictions.
'''

# ASSUMING INDEPENDENT VARIABLES and each feature is normally distributed
# p(x) = (1 / sqrt(2 * pi) * std) exp[-1/2 * (x - mu)/std]^2

log_likelihoods_iid = []
for i in range(0, len(dataset)):
    log_likelihoods_iid.append(compute_independent_loglikelihood(dataset[i],
                                                                 mean_matrix,
                                                                 var_matrix))

logLikelihood = round(sum(log_likelihoods_iid), 3)
print('Log Likelihood assuming independent variables', logLikelihood)

# Assuming multivariate form of distribution
# computing the normal distribution for a vector of 4 features
log_likelihoods_mvar = []
for i in range(0, len(dataset)):
    log_likelihoods_mvar.append(compute_multi_loglikelihood(dataset[i],
                                                            mean_matrix,
                                                            covarianceMat))

logLikelihood = round(sum(log_likelihoods_mvar), 3)
print('Log Likelihood for multi-variate form of the distribution',
      logLikelihood)

'''
 Baysean Network
 BN: x1, x1->x2, x3, x4
 From the correlation matrix we observe that the variables X1 and X2 have high
 positive correlation and we are assuming that variable X1 is a parent of X2. 
 We also assume that variables X1, X3 and X4 are independent variables.
 Resulting in the below BNGraph
 0 0 0 0
 1 0 0 0 
 0 0 0 0
 0 0 0 0
'''
BNGraph = np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
print('BNGraph = ')
print(BNGraph)


# Computing values of Beta parameters, using the matrix form beta = inv(A).Y
a4= (df['CS Score (USNews)'] * df['CS Score (USNews)']).sum()
A = np.array([[49, np.sum(dataset, axis=0)[0]], [np.sum(dataset, axis=0)[0],
                                                  a4]])

Y = np.array([[np.sum(dataset, axis=0)[1]], [(df['CS Score (USNews)'] * df[
    'Research Overhead %']).sum()]])
beta = np.dot(np.linalg.inv(A), Y)
print('Value of Beta computed = ')
print(beta)

beta = beta.ravel()
log_likelihoods_bayesian = []
for i in range(0, len(dataset)):
    log_likelihoods_bayesian.append(compute_bayesean_likelihood(dataset[i],
                                                                 mean_matrix,
                                                                 var_matrix,
                                                                     beta, 1))
# Log likelihood for Bayesian
BNlogLikelihood = round(sum(log_likelihoods_bayesian), 3)
print('Log Likelihood for Bayesian Network',
      BNlogLikelihood)

