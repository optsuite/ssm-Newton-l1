#include <math.h>
#include <string.h>
#include "mex.h"
#include "blas.h"

/* mex lbfgsUpdate.c -lmwblas */

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Variable Declarations */
    /* lbfgsUpdate(yy,dv,status,perm,rhomm,HDiag0,SK,YK) */
    
    double *yy, *dv, pstatus, *perm, *rhomm, *HDiag0, *SK, *YK;
    double *alp, betaa;
    ptrdiff_t n, mm, status;
    double temp;
    ptrdiff_t i,j,kk,inc = 1;
    
    /* Get Input Pointers */
    yy      = mxGetPr(prhs[0]);
    dv      = mxGetPr(prhs[1]);
    pstatus = mxGetScalar( prhs[2] ); status = (ptrdiff_t) pstatus;
    perm    = mxGetPr(prhs[3]);
    rhomm   = mxGetPr(prhs[4]);
    HDiag0  = mxGetPr(prhs[5]);
    SK      = mxGetPr(prhs[6]);
    YK      = mxGetPr(prhs[7]);
    
    n  = mxGetM( prhs[6] );
    mm = mxGetN( prhs[6] );
    
    alp = (double*)mxCalloc(status, sizeof(double));
    /*dcopy(&n, dv, &inc, yy, &inc);*/
    memcpy(mxGetPr(prhs[0]),mxGetPr(prhs[1]),n*sizeof(double));

    /*
    mexPrintf("n: %d, mm: %d, status: %d, HDiag0: %e \n", n,mm, status, *HDiag0);
    for(i=0;i<status;i++)
    {
        kk = (ptrdiff_t)perm[i];
        mexPrintf("perm: %d, rho: %e \n", kk, rhomm[i]);
    }

    kk = 5; 
    for(i=0;i<kk;i++)
    {
        mexPrintf("dv: %e, yy: %e\n", dv[i], yy[i]);
    }
    */

	/*  r(:,1) = q(:,1) */
	for(i=status-1;i>=0;i--)
	{
		kk = (ptrdiff_t)perm[i]-1;
        alp[i] = ddot(&n, yy, &inc, SK+kk*n, &inc)*rhomm[kk];
        temp = -alp[i];
        daxpy(&n, &temp, YK+kk*n, &inc, yy, &inc);
	}

    dscal(&n, HDiag0, yy, &inc);

	/* d = r(:,k+1) */
	for(i=0;i<status;i++)
	{
		kk = (ptrdiff_t)perm[i]-1;
        betaa = ddot(&n, yy, &inc, YK+kk*n, &inc)*rhomm[kk];
        temp = alp[i]-betaa;
        daxpy(&n, &temp, SK+kk*n, &inc, yy, &inc);        
	}

	/* Free Memory */
	mxFree(alp);
}
