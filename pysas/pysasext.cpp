// This SAS code was written by Yi following flowchart of AAron/Shan's code

#include <boost/python.hpp>
#include <numpy/ndarrayobject.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

using namespace boost::python;
using namespace std;
using namespace Eigen;

void _sas(object pool, object meas, object types, list esizes, list ws,
          float t_init, float t_scale, int nt, int nstep, list enses)
{
    Matrix<double,Dynamic,Dynamic,RowMajor> P;
    Matrix<double,1,Dynamic,RowMajor> M, W, S, S_new;
    Matrix<int,1,Dynamic,RowMajor> Idx;
    Matrix<int,Dynamic,Dynamic,RowMajor> TP;
    double *p, *m, *w, *s, *s_new;
    int *idx, *tp;
    double c2, chi2, chi2_new, k, t;
    double sumSM, sumSS, tmp;
    int esize, i, j, ids;
    list ens;

    // pool
    npy_intp *dim_pool = PyArray_DIMS(pool.ptr());
    npy_intp *std_pool = PyArray_STRIDES(pool.ptr());
    char *data_pool = (char *)PyArray_DATA(pool.ptr());
    // meas
    npy_intp *dim_meas = PyArray_DIMS(meas.ptr());
    npy_intp *std_meas = PyArray_STRIDES(meas.ptr());
    char *data_meas = (char *)PyArray_DATA(meas.ptr());
    // types
    npy_intp *dim_types = PyArray_DIMS(types.ptr());
    npy_intp *std_types = PyArray_STRIDES(types.ptr());
    char *data_types = (char *)PyArray_DATA(types.ptr());
    // fetch nframe, ndpt, nds
    // ndpt: # of data points (RDC+CS); nds: # of datasets
    int nframe = dim_pool[0];
    int ndpt = dim_pool[1];
    int nds = dim_types[1];
    ssize_t nens = len(esizes);
    int ndp;  // # of data points for each dataset

    // build P
    P.resize(nframe, ndpt);
    for (i=0; i<nframe; i++){
        for (j=0; j<ndpt; j++)
            P(i,j) = *((float *)(data_pool + i*std_pool[0] + j*std_pool[1]));
    }
    p = P.data();
    // build M
    M.resize(ndpt);
    for (i=0; i<ndpt; i++){
        M(i) = *((float *)(data_meas + i*std_meas[0]));
    }
    m = M.data();
    // build TP
    TP.resize(3, nds);
    for (i=0; i<3; i++){
        for (j=0; j<nds; j++)
            TP(i,j) = *((int *)(data_types + i*std_types[0] + j*std_types[1]));
    }
    tp = TP.data();
    // build weights
    W.resize(nds);
    for (i=0; i<nds; i++){
        W(i) = extract<double>(ws[i]);
    }
    w = W.data();
    // initialize S and S_new
    S.resize(ndpt);
    s = S.data();
    S_new.resize(ndpt);
    s_new = S_new.data();

    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<int> dis1(0, nframe-1);
    uniform_real_distribution<double> dis3(0., 1.);
    for (int iens=0; iens<nens; iens++){
        esize = extract<int>(esizes[iens]);
        cout << ">>> processing N = " << esize << " (" << iens+1
             << "/" << nens << ")...\n";
        ens = extract<list>(enses[iens]);
        // initialize Idx
        Idx.resize(esize);
        idx = Idx.data();
        // build intial ensemble
        for (i=0; i<ndpt; i++)
            s[i] = 0.0;
        for (i=0; i<esize; i++){
            idx[i] = dis1(gen);
            for (j=0; j<ndpt; j++)
                s[j] += p[idx[i]*ndpt + j];
        }
        // calculate chi2
        chi2 = 0.0;
        for (ids=0; ids<nds; ids++){
            sumSM = 0.0; sumSS = 0.0; c2 = 0.0;
            if (tp[ids]==0){  // RDC
                for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                    sumSM += s[i]*m[i];
                    sumSS += s[i]*s[i];
                }
                k = sumSM / sumSS;
                for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                    tmp = s[i]*k -m[i];
                    c2 += tmp*tmp;
                }
            }
            else if (tp[ids]==1){ // CS
                ndp = tp[nds*2+ids] - tp[nds+ids];
                for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                    sumSM += m[i]*esize;
                    sumSS += s[i];
                }
                for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                    tmp = (s[i]-m[i]*esize)-(sumSS-sumSM)/ndp;
                    c2 += tmp*tmp;
                }
            }
            chi2 += c2 * w[ids];
        }
        // start SA
        // pidx: pool index; eidx: ensemble index
        uniform_int_distribution<int> dis2(0, esize-1);
        t = t_init;
        for (int it=0; it<nt; it++){
            for (uint32_t istep=0; istep<nstep; istep++){
                int eidx = dis2(gen); 
                int pidx;
                int found = 0;
                while (!found){
                    pidx = dis1(gen);
                    found = 1;
                    for (i=0; i<esize; i++){
                        if (pidx==idx[i]){
                            found = 0;
                            break;
                        }
                    }
                }
                // calculate new chi2
                for (i=0; i<ndpt; i++)
                    s_new[i] = s[i] - p[idx[eidx]*ndpt+i] + p[pidx*ndpt+i];
                chi2_new = 0.0;
                for (ids=0; ids<nds; ids++){
                    sumSM = 0.0; sumSS = 0.0; c2 = 0.0;
                    if (tp[ids]==0){  // RDC
                        for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                            sumSM += s_new[i]*m[i];
                            sumSS += s_new[i]*s_new[i];
                        }
                        k = sumSM / sumSS;
                        for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                            tmp = s_new[i]*k -m[i];
                            c2 += tmp*tmp;
                        }
                    }
                    else if (tp[ids]==1){ // CS
                        ndp = tp[nds*2+ids] - tp[nds+ids];
                        for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                            sumSM += m[i]*esize;
                            sumSS += s_new[i];
                        }
                        for (i=tp[nds+ids]; i<tp[nds*2+ids]; i++){
                            tmp = (s_new[i]-m[i]*esize)-(sumSS-sumSM)/ndp;
                            c2 += tmp*tmp;
                        }
                        c2 /= (esize*esize);
                    }
                    chi2_new += c2 * w[ids];
                }
                // Metropolis criteria
                if (chi2_new<chi2 || dis3(gen)<exp((chi2-chi2_new)/ndpt/t)){
                    // accept it
                    idx[eidx] = pidx;
                    chi2 = chi2_new;
                    for (i=0; i<ndpt; i++)
                        s[i] = s_new[i];
                }
            }
            t *= t_scale;
        }
        for (i=0; i<esize; i++)
            ens.append(idx[i]);
    }
}


BOOST_PYTHON_MODULE(pysasext)
{
    def("_sas", _sas);
};
