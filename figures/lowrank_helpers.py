import numpy as np
import scipy
import matplotlib.pylab as plt


def phi (mu, delta0):
    integrand = np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def phi_prime (mu, delta0):
    integrand = 1 - (np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def calc_effective_conn(kappa, kappa_I,I,n,overlaps):


  sigma_II = np.dot(I,I)/len(I)
  sigma_nI = np.dot(n,I)/len(I)

  sigma_mm = overlaps[0,0]
  sigma_mn = overlaps[0,1]

  delta =sigma_mm * kappa**2 + sigma_II * kappa_I**2

  s_sigma_mn=sigma_mn * phi_prime(0,delta)
  s_sigma_nI=sigma_nI * phi_prime(0,delta)

  return s_sigma_mn, s_sigma_nI

def gram_schmidt(vecs):
    vecs_orth = []
    vecs_orth.append(vecs[0] / np.linalg.norm(vecs[0]))
    for i in range(1, len(vecs)):
        v = vecs[i]
        for j in range(i):
            v = v - (v @ vecs_orth[j]) * vecs_orth[j]
        v = v / np.linalg.norm(v)
        vecs_orth.append(v)
    return np.array(vecs_orth)

def aprox_PSD(G):
    w, v = np.linalg.eigh(G)
    if any(w<0):
        print('not PSD, truncated negative evals to 0, %f' % min(w))
        w[w<0] = 0
    x = v * np.sqrt(w)
    return x

def generate_low_rank_network(overlaps, N):
    """ overlaps: a list of vector covariances
        N: number of neurons
    """

    n_pop = len(overlaps)
    As = [aprox_PSD(S) for S in overlaps]

    pops = []

    for ni in range(n_pop):
        X = np.random.randn(len(overlaps[0]),N//n_pop)
        X = gram_schmidt(X)
        X = (X.T / np.std(X,1)).T
        pop = As[ni] @ X
        pops.append(pop)

    return np.concatenate(pops,1)

N = 3000

overlaps = np.zeros((4,4))

overlaps[0,0] = 4
overlaps[1,1] = 4
overlaps[2,2] = 4
overlaps[3,3] = 4

overlaps[1,2] = 4
overlaps[2,1] = 4

m,n,IA,IB = generate_low_rank_network([overlaps], N)


gaussian_norm = (1/np.sqrt(np.pi))
gauss_points, gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)