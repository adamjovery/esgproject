import numpy as np
#Function to generate the correlated standard normal shocks for multiple factors
def correlated_shocks(corr_matrix, steps, paths):
    #Initialising shock matrix
    k = corr_matrix.shape[0]

    #Cholesky decomposition to transform the independent standard-normal shocks into correlated standard-normal shocks
    L = np.linalg.cholesky(corr_matrix)

    #Random generation from normal distribution
    independent_z = np.random.standard_normal((k, steps, paths))

    #Multiplying the shock dimension of the generated shocks by the Cholesky decomposed matrix L to gain correlated shocks
    correlated_z = np.einsum('ij,jkl->ikl', L, independent_z)
   
    #Returning the correlated shocks
    return correlated_z