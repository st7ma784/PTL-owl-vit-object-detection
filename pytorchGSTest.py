import torch 
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix

def log_sinkhorn(log_alpha, n_iter):
    for _ in range(n_iter):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
    return log_alpha.exp()

def sample_gumbel(shape, device='cpu', eps=1e-20):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def gumbel_sinkhorn(log_alpha, tau=0.1, n_iter=15 , noise_factor=1.0):
    gumbel_noise = sample_gumbel(log_alpha.shape, device=log_alpha.device) * noise_factor
    # Apply the Sinkhorn operator!
    log_alpha = log_alpha + gumbel_noise
    return log_sinkhorn(log_alpha /tau, n_iter)



def matching(alpha):
    # Negate the probability matrix to serve as cost matrix. This function
    # yields two lists, the row and colum indices for all entries in the
    # permutation matrix we should set to 1.
    row, col = linear_sum_assignment(-alpha)

    row2, col2 = linear_sum_assignment(alpha, maximize=True)
    assert np.all(row == row2) and np.all(col == col2)

    # Create the permutation matrix.
    permutation_matrix = coo_matrix((np.ones_like(row), (row, col))).toarray()
    return torch.from_numpy(permutation_matrix)

def test_best_convergence():
    num_steps = 100
    res=0
    CEloss=torch.nn.CrossEntropyLoss()
    L1=torch.nn.L1Loss()
    BinaryCEloss=torch.nn.BCELoss()

    # Test Params
    LRs=[0.1, 0.2, 0.5, 0.7, ]
    noises=[i/5 for i in range(5)]
    taus=[0.01, 0.03, 0.05, 0.1, 0.5]
    n_iters=[1,10,20,  50, 100, 500]
    scale=[1,10,1000,10000]
    Loss_methods=["CEloss", "L1", ]
    results=torch.zeros(len(LRs), len(noises), len(taus), len(n_iters),len(scale), len(Loss_methods))
    labels=torch.nn.functional.one_hot(torch.arange(26)).float()
    for LRi in range(len(LRs)):
        lr=LRs[LRi]
        for noise in range(len(noises)):
            noise_factor=noises[noise]
            for tauI in range(len(taus)):
                tau=taus[tauI]
                for n_iterI in range(len(n_iters)):
                    n_iter=n_iters[n_iterI]
                    for scaleI in range(len(scale)):
                        scaleFactor = scale[scaleI]
                        for loss_methodI in range(len(Loss_methods)):
                            lossMethod=Loss_methods[loss_methodI]
                            if lossMethod=="CEloss":
                                lossCall=CEloss
                            elif lossMethod=="L1":
                                lossCall=L1
                            else:
                                lossCall=BinaryCEloss
                            for i in range(3):
                                losses=[]
                                log_alpha = torch.randn(26, 26)
                                log_alpha.requires_grad = True
                                loss=lossCall(log_alpha, labels)
                                for test in tqdm(range(num_steps)):
                                    WithGradtensor = gumbel_sinkhorn(scaleFactor*log_alpha, tau=tau, n_iter=n_iter, noise_factor=noise_factor)
                                    Loss=lossCall(WithGradtensor, labels)
                                    Loss.backward()
                                    #apply gradient descent
                                    log_alpha.data -= lr * log_alpha.grad
                                    log_alpha.grad.zero_()

                                print("start Loss : {} -> end Loss : {}".format(loss, Loss))
                                losses.append(Loss)

                    results[LRi, noise, tauI, n_iterI,scaleI,loss_methodI]=sum(losses)/len(losses)
        #find the smallest loss
        minLoss=torch.min(results)
        print(minLoss)
        #find the x,y,z,w,v,u values of the minloss location in the array
        minLossLocation=torch.argmin(results.flatten(0,-1))

        coords=torch.unravel_index(minLossLocation, results.shape)
        best_settings=[LRs[coords[0]], noises[coords[1]], taus[coords[2]], n_iters[coords[3]], scale[coords[4]], Loss_methods[coords[5]]]
        print(best_settings)

def test_best_LSA():
    taus=[0.0001,0.001,0.01, 0.03, 0.05, 0.1, 0.5]
    n_iters=[1,2,3,5,10,15,20,  50,]
    scale=[0.00001, 0.001, 0.01, 0.1, 0.2, 0.4, 0.7, 1,2,5,10]
    Factor=[1,2,10,20,100,1000]
    results=torch.zeros(len(taus), len(n_iters),len(scale),len(Factor))
    for t_i,t in enumerate(taus):
        for n_i,n in enumerate(n_iters):
            for s_i,s in enumerate(scale):
                for f_i,f in enumerate(Factor):
                    result=0
                    for i in range(100):
                        randMatrix=torch.rand((26,26))
                        GS=gumbel_sinkhorn(f*randMatrix,t,n,s)
                        LSA=linear_sum_assignment(randMatrix.cpu().detach().numpy(),maximize=True)
                        #make LSA tuple into matrix with coo
                        LSA_matrix=coo_matrix((np.ones_like(LSA[0]), (LSA[0], LSA[1])))
                        #convert to pytorch Tensor
                        LSA_matrix=torch.tensor(LSA_matrix.toarray())
                        score= (GS * LSA_matrix).sum().div(26)
                        result+=score.item()
                    results[t_i,n_i,s_i,f_i]=result/100
    
    #return highest score, 
    best_Scale=torch.argmax(torch.max(torch.max(torch.max(results,dim=0).values,dim=0).values,dim=-1).values)
    best_Tau=torch.argmax(torch.max(torch.max(torch.max(results,dim=1).values,dim=1).values,dim=1).values)
    best_n_iter=torch.argmax(torch.max(torch.max(torch.max(results,dim=0).values,dim=-1).values,dim=-1).values)
    best_factor=torch.argmax(torch.max(torch.max(torch.max(results,dim=0).values,dim=0).values,dim=0).values)
    best_result=torch.max(results).item()
    print("Best Scale : {} , Best Tau : {} , Best n_iter : {},Best Factor: {},  with result: {}".format(scale[best_Scale], taus[best_Tau], n_iters[best_n_iter],Factor[best_factor],best_result))
    print(results)

if __name__ == "__main__":
    test_best_LSA()