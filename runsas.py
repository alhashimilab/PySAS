#!/usr/bin/python

from pysas import *
from numpy import *

# ===========================  User Input  =====================================
# RDC input data

pfile = 'subFARFAR2-mt_pRDCs.txt'
mfile = 'subFARFAR2-mt_mRDCs.txt'

oupfile = 'sas_subfarfar2_1_50.tab'

pool = loadtxt("./Input/"+pfile, dtype='float32').T
meas = loadtxt("./Input/"+mfile, dtype='float32')
idx = [0,35,74,108]
etype = ['rdc','rdc','rdc','rdc']
ws = [1.,1.,1.,1.]

#ws = [7.05,4.73,4.02,4.63]

# SAS
t_init = 100.
rep = 1
nstep = 500000
esizes = range(1,51)
#esizes = [20]
#esizes = [200]
#esizes = [20] * 100
# ==============================================================================

ncol = pool.shape[1]
pool = [pool[:,i:j] for i,j in zip(idx,idx[1:]+[ncol])]
meas = [meas[i:j] for i,j in zip(idx,idx[1:]+[ncol])]

result =sas(pool, meas, etype, esizes, ws, rep=rep, t_init=t_init, nstep=nstep)

RMSD = []
f = open('./Output/%s'%oupfile, 'wt')
for res in result:
    es,rmsd,idx = res
    f.write('%2d  %.3f'%(es,rmsd))
    f.write((' %d'*es+'\n')%tuple(idx))
f.close()
