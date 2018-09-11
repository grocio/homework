#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

def normalise(v):
    return v/np.linalg.norm(v)

def dis_sim(v1, v2):
    dis = sum((normalise(v1)-normalise(v2))**2)
    return np.exp(-1*dis)

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

def complex_recall_RS(position_num=5, unit_num=2**4, run_num=50000, RS_flag=True, distractor_repeat=False):
    results = [0.0 for i in range(position_num)]

    for run in range(run_num):
        cue_vecs = context_vecs(position_num, 0.2, unit_num)
        target_vecs = [random_posi_neg_vec(unit_num) for i in range(position_num)]
        distractor_vecs = [random_posi_neg_vec(unit_num) for i in range(position_num)]

        # Learning
        W = np.zeros((unit_num, unit_num))
        for i in range(position_num):
            W += hebbian_encoding(target_vecs[i], cue_vecs[i])
            # This model does not include the removal mechanism.
            # I reduced the alpha (the encoding strength) for encoding of distractors
            if distractor_repeat == False:
                W += hebbian_encoding(distractor_vecs[i], cue_vecs[i], alpha=0.3)
            # If distractor_repeat == True, the same distractor is repeated
            else:
                W += hebbian_encoding(distractor_vecs[0], cue_vecs[i], alpha=0.3)


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
    run_num = 50000

    RS_off_repeat = complex_recall_RS(position_num=position_num, run_num=run_num, RS_flag=False, distractor_repeat=True)
    RS_off_change = complex_recall_RS(position_num=position_num, run_num=run_num, RS_flag=False, distractor_repeat=False)

    RS_on_repeat = complex_recall_RS(position_num=position_num, run_num=run_num, RS_flag=True, distractor_repeat=True)
    RS_on_change = complex_recall_RS(position_num=position_num, run_num=run_num, RS_flag=True, distractor_repeat=False)

    
    print('RS_off_repeat', RS_off_repeat)
    print('RS_off_change', RS_off_change)

    print('RS_on_repeat', RS_on_repeat)
    print('RS_on_change', RS_on_change)

    # Plot
    x = np.array([i for i in range(position_num)])

    plt.plot(x, RS_off_repeat, color='blue', label='RS_off_repeat', marker='o') 
    plt.plot(x, RS_off_change, color='blue', label='RS_off_change', marker='x') 
    plt.plot(x, RS_on_repeat, color='red', label='RS_on_repeat', marker='o') 
    plt.plot(x, RS_on_change, color='red', label='RS_on_change', marker='x') 
    plt.legend(loc = 'upper center')
    plt.xticks(x)
    plt.xlabel('Position')
    plt.ylabel('Probability of Correct Recall')
    plt.show()
