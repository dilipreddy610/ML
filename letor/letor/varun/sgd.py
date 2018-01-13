import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
 
'''
Container class to store the parameters related to defined K i.e. number of
cluster. This is a helper class.
'''  
class OptimumValues:
    
    def __init__(self, k, minimum_train_error, minimum_val_error, centers, spreads, weight_vector, L2_lambda):  
        self.k = k
        self.minimum_train_error = minimum_train_error
        self.minimum_val_error = minimum_val_error
        self.centers = centers
        self.spreads = spreads
        self.weight_vector = weight_vector  
        self.L2_lambda = L2_lambda
    
    def get_k_clusters(self):
        return self.k
    
    def get_minimum_val_error(self):
        return self.minimum_val_error
    
    def get_minimum_train_error(self):
        return self.minimum_train_error
    
    def get_centers(self):
        return self.centers
    
    def get_spreads(self):
        return self.spreads
    
    def get_weight_vector(self):
        return self.weight_vector
    
    def get_L2_lambda(self):
        return self.L2_lambda
    
'''
Compute closed form solution of the input data
'''
def closed_form_sol(L2_lambda, design_matrix_training, output_data):
    return np.linalg.solve(
    L2_lambda * np.identity(design_matrix_training.shape[1]) + 
    np.matmul(design_matrix_training.T, design_matrix_training),
    np.matmul(design_matrix_training.T, output_data)).flatten()

'''
This method is used to compute the Kmeans cluster fitting for input data
 and returns the centers, spreads for the clusters computed.
'''
def compute_cluster_kmeans(X, num_cluster, dimensions):
    # fit data using K-Means
    kmeans = KMeans(n_clusters=num_cluster).fit(X) 
    centers = kmeans.cluster_centers_
    spreads = np.ndarray(shape=(num_cluster, dimensions, dimensions))

    # compute covariance matrix from K-means label array
    for i in range(0, num_cluster):
        indices = np.where(kmeans.labels_ == i)[0]
        cov_matrix = np.cov(np.transpose(np.take(X, indices, axis=0)))
        spreads[i] = np.linalg.pinv(cov_matrix)
 
 
 
    # compute the design matrix  
    centers = centers[:, np.newaxis, :]
    return centers, spreads

'''
Compute the design matrix
'''    
def compute_design_matrix(X, centers, spreads):
    # use broadcast
    basis_func_outputs = np.exp(
    np.sum(
    np.matmul(X - centers, spreads) * (X - centers),
    axis=2) / (-2)).T
    # insert ones to the 1st col
    return np.insert(basis_func_outputs, 0, 1, axis=1)

'''
Compute the root mean square error using the predicted value and the actual output
against 
'''
def err_func(L2_lamda, predict_val, output, weight_vector):
    err_data = (0.5) * np.sum(np.power(predict_val - output, 2), axis=0) + L2_lamda * np.dot(np.transpose(weight_vector), weight_vector)
    return math.sqrt(2 * err_data / len(output))
 
'''
Compute the stochastic gradient descent based weight vector.
''' 
def SGD_sol(learning_rate, minibatch_size, num_epochs, L2_lambda, design_matrix_training,
            design_matrix_validation, output_data, output_validation, patience, validation_steps):
    N, _ = design_matrix_training.shape
    # You can try different mini-batch size size
    # Using minibatch_size = N is equivalent to standard gradient descent
    # Using minibatch_size = 1 is equivalent to stochastic gradient descent
    # In this case, minibatch_size = N is better
    weights = np.zeros([1, design_matrix_training.shape[1]])
    # The more epochs the higher training accuracy. When set to 1000000,
    # weights will be very close to closed_form_weights. But this is unnecessary
    p = patience
    v = sys.maxsize
    weights_optimal = np.zeros([1, design_matrix_training.shape[1]])
    best_train_steps = 0

    '''
    We are using early stopping as a regularization technique
    '''
    for epoch in range(1, num_epochs + 1):
      
        for i in range(int(N / minibatch_size)):
            lower_bound = i * minibatch_size
            upper_bound = min((i + 1) * minibatch_size, N)
            Phi = design_matrix_training[lower_bound : upper_bound, :]
            t = output_data[lower_bound : upper_bound, :]
            E_D = np.matmul(
                 (np.matmul(Phi, weights.T) - t).T,
                 Phi)
            E = (E_D + L2_lambda * weights) / minibatch_size
            weights = weights - learning_rate * E
        np.linalg.norm(E)
        
        # early stopping check        
        if(epoch % validation_steps == 0):
            if(p < 1):
                break   
            
            letor_predict_val_sgd = np.transpose(np.mat(np.matmul(design_matrix_validation, weights.flatten())))
            v2 = err_func(0.1, letor_predict_val_sgd, output_validation, weights.flatten())
            
            if(v2 < v):
                p = patience
                weights_optimal = weights
                v = v2
                best_train_steps = epoch
                 
            else:
                p -= 1
            
              
                                  
    return weights_optimal.flatten(), best_train_steps
 

'''
This method computes the weight vector and the error against the validation data 
set for closed form solution 
'''
def compute_closed_error(L2_lamda, input_training, input_validation, output_training, output_validation, centers, spreads):
  
    design_matrix_training = compute_design_matrix(input_training, centers, spreads)
    design_matrix_validation = compute_design_matrix(input_validation, centers, spreads)
    weight_vector_closed = closed_form_sol(L2_lamda, design_matrix_training, output_training)

    # training data error
    predict_training_closed = np.transpose(np.mat(np.matmul(design_matrix_training, weight_vector_closed)))
    err_training_closed = err_func(L2_lamda, predict_training_closed, output_training, weight_vector_closed)
    
    # validation data error
    predict_val_closed = np.transpose(np.mat(np.matmul(design_matrix_validation, weight_vector_closed)))
    err_validation_closed = err_func(L2_lamda, predict_val_closed, output_validation, weight_vector_closed)
    
    return weight_vector_closed, err_training_closed, err_validation_closed

'''
This method computes the weight vector and the error against the validation data 
set for stochastic gradient descent form solution 
'''
def compute_sgd_error(L2_lamda, input_training, input_validation, output_training, 
                      output_validation, centers, spreads, patience, validation_steps,
                      epoch, minibatch_size):
    
    design_matrix_training = compute_design_matrix(input_training, centers, spreads)
    design_matrix_validation = compute_design_matrix(input_validation, centers, spreads)
    
    weight_vector_sgd, best_train_steps = SGD_sol(1, minibatch_size, epoch, L2_lamda, 
                                                  design_matrix_training, design_matrix_validation, 
                                                  output_training, output_validation, 
                                                  patience, validation_steps)
    
    # training data error
    predict_train_sgd = np.transpose(np.mat(np.matmul(design_matrix_training, weight_vector_sgd)))
    syn_train_err_sgd = err_func(L2_lamda, predict_train_sgd, output_training, weight_vector_sgd)
    
    # validation data error
    predict_val_sgd = np.transpose(np.mat(np.matmul(design_matrix_validation, weight_vector_sgd)))
    syn_val_err_sgd = err_func(L2_lamda, predict_val_sgd, output_validation, weight_vector_sgd)
    return weight_vector_sgd, syn_train_err_sgd, syn_val_err_sgd, best_train_steps
    
   
    
# starter
# syn_input_data = np.genfromtxt('..//resources//input.csv', delimiter=',')
df = pd.read_csv('..//resources//input.csv')
syn_input_data = df.as_matrix()

# syn_output_data = np.genfromtxt(
# '..//resources//output.csv', delimiter=',').reshape([-1, 1])
df = pd.read_csv('..//resources//output.csv')
syn_output_data = df.as_matrix()

letor_input_data = np.genfromtxt(
'..//resources//Querylevelnorm_X.csv', delimiter=',')
letor_output_data = np.genfromtxt(
'..//resources//Querylevelnorm_t.csv', delimiter=',').reshape([-1, 1])

letor_input_data = np.append(letor_input_data, letor_output_data, axis=1)

np.random.shuffle(letor_input_data)

letor_output_data = letor_input_data[:, letor_input_data.shape[1] - 1:]
letor_input_data = letor_input_data[:, 0: (letor_input_data.shape[1] - 1)]

syn_input_data = np.append(syn_input_data, syn_output_data, axis=1)

np.random.shuffle(syn_input_data)
syn_output_data = syn_input_data[:, syn_input_data.shape[1] - 1:]
syn_input_data = syn_input_data[:, 0: (syn_input_data.shape[1] - 1)]

letor_input_training, letor_input_validation, letor_input_test = np.split(letor_input_data,
                                                                          [int(0.8 * len(letor_input_data)), int(0.9 * len(letor_input_data))])

letor_output_training, letor_output_validation, letor_output_test = np.split(letor_output_data,
                                                                          [int(0.8 * len(letor_output_data)), int(0.9 * len(letor_output_data))])

syn_input_training, syn_input_validation, syn_input_test = np.split(syn_input_data,
                                                                          [int(0.8 * len(syn_input_data)), int(0.9 * len(syn_input_data))])

syn_output_training, syn_output_validation, syn_output_test = np.split(syn_output_data,
                                                                          [int(0.8 * len(syn_output_data)), int(0.9 * len(syn_output_data))])

'''
To find the optimum number of clusters for each data set and each form of solution,
we iterate between a range of number of cluster. We compute weights for each unique 
number of clusters we compute the weights and then find the cross-validation error
against it. After iterating through the range of number of clusters we identify
the cluster with the minimum cross validation error as the optimal number of clusters.

You will notice that the optimal number of clusters will vary on each run. This
is because of the initial cluster centers chosen by the KMEANS library used and
is hence expected. 
'''
minimum_number_clusters = 5
maximum_number_clusters = 15
clusters_step_size = 2
patience = 10
validation_steps = 5
L2_lambdas = [0.1, 0.01, 0.03]
epoch = 10000

letor_closed_min_err = sys.maxsize
letor_sgd_min_err = sys.maxsize
syn_closed_min_err = sys.maxsize
syn_sgd_min_err = sys.maxsize

# containers to store the optimal number of clusters and its properties like centers
# and spreads
letor_closed_optimum_values = OptimumValues(0, 0, sys.maxsize, 0, 0, 0, 0)
letor_sgd_optimum_values = OptimumValues(0, 0, sys.maxsize, 0, 0, 0, 0)
syn_closed_optimum_values = OptimumValues(0, 0, sys.maxsize, 0, 0, 0, 0)
syn_sgd_optimum_values = OptimumValues(0, 0, sys.maxsize, 0, 0, 0, 0)


# iterate to find optimum no. of clusters 
letor_closed_num_clusters_to_err = {}
letor_sgd_num_clusters_to_err = {}
syn_closed_num_clusters_to_err = {}
syn_sgd_num_clusters_to_err = {}

# iterate to compute values in the range of clusters
for num_cluster in range(minimum_number_clusters, maximum_number_clusters, clusters_step_size):
    
    # compute centers and spreads
    centers, spreads = compute_cluster_kmeans(letor_input_training, num_cluster, letor_input_training.shape[1])
    centers_synthetic, spreads_synthetic = compute_cluster_kmeans(syn_input_training, num_cluster, syn_input_training.shape[1])
    
    # creating a copy of the data for closed form solution with added axis    
    letor_input_training_copy = letor_input_training[np.newaxis, :, :]
    syn_input_training_copy = syn_input_training[np.newaxis, :, :]
    
    for lmda in L2_lambdas:
        # compute the closed form and SGD for letor data for each num_cluster
        letor_closed_weight_vector, letor_closed_train_err, letor_closed_val_err = compute_closed_error(lmda, letor_input_training_copy, 
                                                       letor_input_validation, letor_output_training, 
                                                       letor_output_validation, centers, spreads)
    
        letor_sgd_weight_vector, letor_sgd_train_err, letor_sgd_val_err, best_train_steps = compute_sgd_error(lmda, 
                        letor_input_training, letor_input_validation, letor_output_training, 
                        letor_output_validation, centers, spreads, patience, validation_steps, epoch, int(len(letor_input_training) / 100))

        # compute the closed form and SGD for synthetic data for each num_cluster 
        syn_weight_vector_closed, syn_closed_train_err, syn_closed_train_err = compute_closed_error(lmda, 
                                                   syn_input_training, syn_input_validation, 
                                                   syn_output_training, syn_output_validation, 
                                                   centers_synthetic, spreads_synthetic)
    
        syn_sgd_weight_vector, syn_sgd_train_err, syn_sgd_val_err, best_train_steps = compute_sgd_error(lmda, 
                               syn_input_training, syn_input_validation, syn_output_training, 
                               syn_output_validation, centers_synthetic, spreads_synthetic, patience, validation_steps,
                               epoch, int(len(syn_input_training) / 100))
    
        # find value of cluster with minimum error for Closed form solution for LETOR data set
        if(letor_closed_min_err > letor_closed_val_err):
            letor_closed_min_err = letor_closed_val_err
            letor_closed_optimum_values = OptimumValues(num_cluster, letor_closed_train_err, letor_closed_val_err, 
                                                       centers, spreads, letor_closed_weight_vector, lmda)
    
        # find value of cluster with minimum error for SGD form solution for LETOR data set
        if(letor_sgd_min_err > letor_sgd_val_err):
            letor_sgd_min_err = letor_sgd_val_err
            letor_sgd_optimum_values = OptimumValues(num_cluster, letor_sgd_train_err, letor_sgd_val_err, 
                                                     centers, spreads, letor_sgd_weight_vector, lmda)
        
        # find value of cluster with minimum error for Closed form solution for synthetic data set    
        if(syn_closed_min_err > syn_closed_train_err):
            syn_closed_min_err = syn_closed_train_err
            syn_closed_optimum_values = OptimumValues(num_cluster, syn_closed_train_err, syn_closed_train_err, 
                                                    centers_synthetic, spreads_synthetic, syn_weight_vector_closed, lmda)
        
        # find value of cluster with minimum error for SGD form solution for synthetic data set    
        if(syn_sgd_min_err > syn_sgd_val_err):
            syn_sgd_min_err = syn_sgd_val_err
            syn_sgd_optimum_values = OptimumValues(num_cluster, syn_sgd_train_err, syn_sgd_val_err, 
                                                   centers_synthetic, spreads_synthetic, syn_sgd_weight_vector, lmda)



print('UBitName = varunjai')
print('personNumber = 50247176')
print('UBitName = dilipred')
print('personNumber = 50248867')
            
# print the characteristics observed for the optimal number of clusters chosen        
print('CLOSED FORM SOLUTION')            
print('Optimum number of clusters: ', letor_closed_optimum_values.get_k_clusters())
print('Minimum error on TRAINING data', letor_closed_optimum_values.get_minimum_train_error()) 
print('Minimum error on VALIDATION data', letor_closed_optimum_values.get_minimum_val_error())
print('Optimum value of lambda: ', letor_closed_optimum_values.get_L2_lambda())

# get the error on test data
letor_design_matrix_test_closed = compute_design_matrix(letor_input_test, letor_closed_optimum_values.get_centers(), letor_closed_optimum_values.get_spreads())
letor_predict_val_test_closed = np.transpose(np.mat(np.matmul(letor_design_matrix_test_closed, letor_closed_optimum_values.get_weight_vector())))

letor_test_err_closed = err_func(letor_closed_optimum_values.get_L2_lambda(), letor_predict_val_test_closed, letor_output_test, letor_closed_optimum_values.get_weight_vector())


print("Root Mean Square Error on TEST data:  ", letor_test_err_closed)
print()

print('STOCHASTIC GRADIENT DESCENT SOLUTION')
print('Optimum number of clusters: ', letor_sgd_optimum_values.get_k_clusters())
print('Optimum value of lambda: ', letor_sgd_optimum_values.get_L2_lambda())
print('Root Mean Square Error on TRAINING data:  ', letor_sgd_optimum_values.get_minimum_train_error())
print('Root Mean Square Error on VALIDATION data:  ', letor_sgd_optimum_values.get_minimum_val_error())
print('Best training steps: ', best_train_steps)

letor_design_matrix_test_sgd = compute_design_matrix(letor_input_test, letor_sgd_optimum_values.get_centers(), letor_sgd_optimum_values.get_spreads())
letor_predict_val_test_sgd = np.transpose(np.mat(np.matmul(letor_design_matrix_test_sgd, letor_sgd_optimum_values.get_weight_vector())))

test_err_sgd = err_func(letor_sgd_optimum_values.get_L2_lambda(), letor_predict_val_test_sgd, letor_output_test, letor_sgd_optimum_values.get_weight_vector())
print("Root Mean Square Error on TEST data:   ", test_err_sgd)
print()
print('=========================================================================')


print()
print('SYNTHETIC CLOSED FORM SOLUTION')            
print('Optimum number of clusters: ', syn_closed_optimum_values.get_k_clusters())
print('Optimum value of lambda: ', syn_closed_optimum_values.get_L2_lambda())
print('Minimum error on TRAINING data', syn_closed_optimum_values.get_minimum_train_error())
print('Minimum error on VALIDATION data', syn_closed_optimum_values.get_minimum_val_error())


# get the error on test data
syn_design_matrix_test_closed = compute_design_matrix(syn_input_test, syn_closed_optimum_values.get_centers(), syn_closed_optimum_values.get_spreads())
syn_predict_val_test_closed = np.transpose(np.mat(np.matmul(syn_design_matrix_test_closed, syn_closed_optimum_values.get_weight_vector())))

syn_test_err_closed = err_func(syn_closed_optimum_values.get_L2_lambda(), syn_predict_val_test_closed, syn_output_test, syn_closed_optimum_values.get_weight_vector())
print("Root Mean Square error value on TEST data:   ", syn_test_err_closed)
print()

print('SYNTHETIC SGD FORM SOLUTION')
print('Optimum number of clusters:   ', syn_sgd_optimum_values.get_k_clusters())
print('Optimum value of lambda: ', syn_sgd_optimum_values.get_L2_lambda())
print('Root Mean Square Error on TRAINING data:  ', syn_sgd_optimum_values.get_minimum_train_error())
print('Root Mean Square Error on VALIDATION data:  ', syn_sgd_optimum_values.get_minimum_val_error())
print('Best training steps: ', best_train_steps)

syn_design_matrix_test_sgd = compute_design_matrix(syn_input_test, syn_sgd_optimum_values.get_centers(), syn_sgd_optimum_values.get_spreads())
syn_predict_val_test_sgd = np.transpose(np.mat(np.matmul(syn_design_matrix_test_sgd, syn_sgd_optimum_values.get_weight_vector())))

test_err_sgd = err_func(syn_sgd_optimum_values.get_L2_lambda(), syn_predict_val_test_sgd, syn_output_test, syn_sgd_optimum_values.get_weight_vector())
print("Root Mean Square Error on TEST data:   ", test_err_sgd)








