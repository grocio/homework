#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def normalise(v):
    return v/np.linalg.norm(v)

def dis_sim(v1, v2):
    dis = sum((normalise(v1)-normalise(v2))**2)
    return np.exp(-1*dis)

"""
def dis_sim(v1, v2):
    dis = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))
    return np.exp(dis+1)
"""

def hebbian_encoding(target_vec, cue_vec, alpha=1.0):
    return alpha*np.outer(target_vec, cue_vec)

def hebbian_retrieval(W, cue_vec):
    return np.dot(W, cue_vec)
# It gives a random vector with the elements being 1 or -1
# e.g., [1,1,-1,1,1,-1,-1,1]
def random_posi_neg_vec(unit_num):
    zero_one_vec = np.random.randint(0,2, unit_num)
    return 1 - 2*zero_one_vec

# It stochastically changes the elements of a vector
# e.g., [1,1,1,-1,-1,-1] -> [-1,1,1,-1,1,-1]
def random_flip(vec, drift_rate):
    target_vec = np.copy(vec)
    unit_num = len(vec)
    change_num = round(unit_num*drift_rate)
    change_index = np.random.choice(unit_num, change_num)
    
    target_vec[change_index] = -1 * target_vec[change_index]

    return target_vec

# It gives the vectors that represent the drifting context
def context_vecs(position_num, drift_rate, unit_num):
    context_vecs = [0 for i in range(position_num)]
    for i in range(position_num):
        if i == 0:
            context_vecs[i] = random_posi_neg_vec(unit_num)
            # context_vecs[i] = random_flip(random_posi_neg_vec(unit_num), drift_rate)
        else:
            context_vecs[i] = random_flip(context_vecs[i-1], drift_rate)
    return context_vecs

def serial_recall_RS(position_num=5, unit_num=2**4, run_num=50000, RS_flag=True):
    results = [0.0 for i in range(position_num)]

    for run in range(run_num):
        cue_vecs = context_vecs(position_num, 0.2, unit_num)
        target_vecs = [random_posi_neg_vec(unit_num) for i in range(position_num)]

        # Learning
        W = np.zeros((unit_num, unit_num))
        for i in range(position_num):
            W += hebbian_encoding(target_vecs[i], cue_vecs[i])

        # Retrieval with or withour RS
        for i in range(position_num):
            recovery = hebbian_retrieval(W, cue_vecs[i])
            dis_sim_vecs = []
            for j in range(position_num):
                dis_sim_vecs.append(dis_sim(recovery, target_vecs[j]))
            prob = np.array(dis_sim_vecs)
            prob /= sum(prob)
            out = target_vecs[np.random.choice(position_num, p=prob)]
            results[i] += np.allclose(target_vecs[i], out) 

            # RS
            if RS_flag == True:
                W -= hebbian_encoding(out, cue_vecs[i])
            
    
    results = np.array(results)
    results /= run_num
    return results

if __name__ == '__main__':
    position_num = 5
    run_num = 500000
    x = np.array([i for i in range(position_num)])

    RS_on_results = serial_recall_RS(position_num=position_num, run_num=run_num, RS_flag=True)
    RS_off_results = serial_recall_RS(position_num=position_num, run_num=run_num, RS_flag=False)
    RS_diff_results = RS_on_results - RS_off_results

    fig, ax = plt.subplots(1,2)
    ax[0].plot(x, RS_on_results, color='red', label='RS_on', marker='o') 
    ax[0].plot(x, RS_off_results, color='blue', label='RS_off', marker='o') 
    ax[0].set_xticks(x)
    ax[0].set_xlim(-0.5, position_num-1+0.5)
    #ax[0].set_ylim(0.0, 1.0)
    ax[0].legend(loc = 'upper left')
    ax[0].set_title('Average Performance by Positions')
    ax[0].set_xlabel('Position')
    ax[0].set_ylabel('Probability of Correct Recall')
    ax[1].plot(x, RS_diff_results, color='green', marker='o')
    ax[1].set_xticks(x)
    ax[1].set_xlim(-0.5, position_num-1+0.5)
    #ax[1].set_ylim(0.0, 1.0)
    ax[1].set_title('Differences of RS_on vs. RS_off by Positions')
    ax[1].set_xlabel('Position')
    ax[1].set_ylabel('Difference')
    print('diff', RS_on_results - RS_off_results)
    print('RS_on_results', RS_on_results)
    print('RS_off_results', RS_off_results)
    plt.show()
