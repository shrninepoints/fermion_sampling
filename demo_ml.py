import fermion_sampling as fs
import numpy as np

def sampleL(e,v,method): # sample with input L
    size = len(e)
    v = v.T
    for i in range(size):
        if np.random.rand() > (e[i]/(e[i]+1)):
            np.delete(v,i)
    if method == 'dpp':    
        result = fs.dpp(v,len(v),size)
    elif method == 'sampling':
        result = fs.sampling(v,len(v),size)
    return result

def cosine(a,b): # cosine similarity between two vectors
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def readData(path): # return list of strings in form of text-summary pair
    with open(path) as t:
        lines = t.readlines()
        lines = [i for i in lines if not i == '\n']
        sep = lines.index('@highlight\n')
        text = lines[0:sep]
        summary = [lines[sep+2*i+1] for i in range((len(lines) - sep) // 2)]
    return text,summary

def matrixS(text_vec): # return S matrix for a text TODO: add hyperparameter to increase replusion
    l = len(text_vec)
    S = np.zeros((l,l))
    for i in range(l):
        for j in range(l):
            S[i,j] = cosine(text_vec[i],text_vec[j])

def featureFunction(text): # return a d-dim vector for each sentence, in size*d matrix
    # TODO: add more features
    length = len(text)
    const = np.array([1 for i in range(length)])# propornational to sample size
    len_list = np.array([length for i in text]) # longer sentence has higher weight
    position_mark = np.array([abs(length/2-i) // 5 for i in range(length)]) # sentence at front/end has higher weight
    pronouns_num = np.array([i.count("I ") + i.count("he ") + i.count("she ") + i.count("He ") \ 
                + i.count("She ") + i.count("You ") + i.count("you ") for i in text])
    return np.array([const,len_list,len_list ** 2,len_list ** 3,position_mark,position_mark ** 2,position_mark ** 3,pronouns_num,pronouns_num ** 2,pronouns_num ** 3]).T

def quality(feature,theta): # return quality vector in 1*size
    q = [np.exp(np.dot(feature[i],theta)) for i in len(feature)]
    return q

def generateL(text,theta):
    text_vec = vector(text)
    S = matrixS(text_vec)
    q = quality(featureFunction(text),theta)
    L = np.dot(q.T,S).dot(q)
    return L

if __name__ == "__main__":
    print(readData("/Users/macbookpro/projects/FermionSampling/cnn_stories_tokenized/0a0a4c90d59df9e36ffec4ba306b4f20f3ba4acb.story"))