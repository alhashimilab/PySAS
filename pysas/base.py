import pysasext as ext
import numpy as np

# ******************************************************************************
def sas(pool, meas, etype, esizes, ws=None, rep=10,
        t_init=2.0, t_scale=0.9, nt=100, nstep=20000):
    '''
    pool: ***********************************************
    meas: ***********************************************
    esizes: *********************************************
    '''

    # check input
    if not isinstance(pool, list) or not isinstance(meas, list) \
            or not isinstance(etype, list):
        print('Please convert pool, meas and etype into list. Aborted!')
        exit(1)
    if len(pool)!=len(meas) or len(pool)!=len(etype):
        print('pool/meas/etype have difference # of sub-datasets. Aborted!')
        exit(1)
    nds = len(meas)  # number of datasets
    dic = {'RDC':0, 'CS':1}
    types = np.zeros((3,nds), dtype=int)
    if ws is None:
        ws = [1.0]*nds
    if len(ws) != nds:
        print('# of weights does not match # of datasets. Aborted!')
        exit(1)
    ibeg = 0
    for i in range(nds):
        types[0,i] = dic[etype[i].upper()]
        types[1,i] = ibeg
        ibeg += len(meas[i])
        types[2,i] = ibeg
    pool = [p.astype('float32') for p in pool]
    pool = np.hstack(pool)
    meas = [m.astype('float32') for m in meas]
    meas = np.hstack(meas)
    if pool.shape[1]!=meas.shape[0]:
        print('pool and meas do not match each other!')
        exit(1)
    nes = len(esizes)  # number of ensembles to reconstruct
    enses = [[] for i in range(nes*rep)]
    ext._sas(pool, meas, types, esizes*rep, ws,
             t_init, t_scale, nt, nstep, enses)

    result = []
    for ies in range(nes):
        res = []
        for j in range(rep):
            idx = enses[j*nes+ies]
            chi2 = 0.0
            for i in range(types.shape[1]):
                s = np.mean(pool[idx,types[1,i]:types[2,i]], axis=0)
                m = meas[types[1,i]:types[2,i]]
                if types[0,i] == 0:  # RDC
                    k = np.dot(m,s) / np.dot(s,s)
                    chi2 += (sum((k*s-m)**2)) * ws[i]
                elif types[0,i] == 1:  # CS
                    chi2 += sum(((s-m)-(np.mean(s)-np.mean(m)))**2) * ws[i]
            rmsd = np.sqrt(chi2/pool.shape[1])
            res.append([rmsd, idx])
        res.sort()
        result.append([esizes[ies]]+res[0])

    return result
