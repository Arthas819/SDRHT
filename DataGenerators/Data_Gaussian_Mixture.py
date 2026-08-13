"""
    Generator for the Gaussian Mixture Dataset.
"""
import numpy as np
import matplotlib.pyplot as plt  

def generate_Gaussian_mixture_data(N, d):

    # H0: 0.5N(0.4e, Id) + 0.5N(-0.4e, Id)
    # H1: 0.5N(0.4f, Id) + 0.5N(-0.4f, Id)

    mean = 0.4

    # H0
    e = np.ones((d, 1))
    distribution_h0 = np.random.multivariate_normal(mean=mean*e.flatten(), cov=np.eye(d), size=N//2)
    distribution_h0 = np.concatenate((distribution_h0, np.random.multivariate_normal(mean=-mean*e.flatten(), cov=np.eye(d), size=N//2)), axis=0)

    # H1
    f = np.concatenate((np.ones((d//2, 1)), - np.ones((d//2, 1))), axis=0)
    distribution_h1 = np.random.multivariate_normal(mean=mean*f.flatten(), cov=np.eye(d), size=N//2)
    distribution_h1 = np.concatenate((distribution_h1, np.random.multivariate_normal(mean=-mean*f.flatten(), cov=np.eye(d), size=N//2)), axis=0)

    np.random.shuffle(distribution_h0)
    np.random.shuffle(distribution_h1)

    # Return two training sets, X_1 belongs to P_1 and X_2 belongs to P_2.
    return distribution_h0, distribution_h1


if __name__ == "__main__":
    # Visualize the generated Gaussian mixture when d=2. 
    N = 100
    d = 2
    X_0, X_1 = generate_Gaussian_mixture_data(N, d)
    plt.scatter(X_0[:, 0], X_0[:, 1], color='blue', alpha=0.5, label='H0')
    plt.scatter(X_1[:, 0], X_1[:, 1], color='red', alpha=0.5, label='H1')
    plt.title('Gaussian Mixture Data (d=2)')    
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid()
    plt.show()  