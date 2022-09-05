/* 
 MEX function [s,ss]=mhgi(MAX,alpha,p,q,n,x).
 Computes the truncated hypergeometric function pFq ^alpha(p;q;x*I_n)
 The sum is only over partitions |kappa|<=MAX(1); and kappa(1)<=MAX(2)
 (thus all kappa(i)<=MAX(2))
 p and q are arrays, so hgi([30 10],a,[3 4],[5 6 7],3,0.5) is 
 2F3^a([3 4];[5 6 7];0.5*I_3) summed over all partitions |kappa|<=30
 with kappa(i)<=10
 x may be a vector; if so, then a vector is returned.
 Uses the formula 
 J_lambda(x I_n)=x^|lambda| \prod_{(i,j)\in \lambda (n-(i-1)+alpha(j-1))

 by Plamen Koev, February 2004, May 2004.

 Removed recursion, April 2005.

 November 2005, returning the coefficients now and using Horner's method 
 to evaluate, 
 December 2005, allowed for MAX to have 2 parts 
                
 now returns the vector ss with coefficients s.t. 
 s=ss[0]+ss[1]*x+...+ss[m+1]*x^m, meant componentwise when x is a vector
 in other words, s=polyval(ss(end:-1:1),x)
*/

#include <mex.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[]) {
  int i, MAX, n, nx, np, nq, *l, *lc, j, sl, K;
  double alpha, *x,*p, *q, *s, *ss;
  double *z, zn,c,d,e,f,g;

  if ((nrhs<6) || (nrhs>6))
    mexErrMsgTxt("mhgi takes six input arguments.");
 
  if (nlhs>2)
    mexErrMsgTxt("Too many output arguments.");
  
  if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3]) || !mxIsDouble(prhs[4]) || !mxIsDouble(prhs[5]))
    mexErrMsgTxt("All inputs must be double arrays.");
  if (nrhs==7) if (!mxIsDouble(prhs[6])) mexErrMsgTxt("All inputs must be double arrays.");

  if (mxGetM(prhs[0])!=1 || mxGetN(prhs[0])<1 || mxGetN(prhs[0])>2)
    mexErrMsgTxt("First input must be an array of size 1 or 2 = [max size of partition, max size of the largest part].");
  if (mxGetM(prhs[1])!=1 || mxGetN(prhs[1])!=1)
    mexErrMsgTxt("Second input must be a scalar = alpha.");
  if (mxGetM(prhs[2])>1 && mxGetN(prhs[2])>1)
    mexErrMsgTxt("Third input must be a vector = (a_1,...,a_p).");
  if (mxGetM(prhs[3])>1 && mxGetN(prhs[3])>1)
    mexErrMsgTxt("Forth input must be a vector = (b_1,...,b_q).");
  if (mxGetM(prhs[4])!=1 || mxGetN(prhs[4])!=1)
    mexErrMsgTxt("Fifth input must be a scalar = number of variables.");
  if (mxGetM(prhs[5])!=1 && mxGetN(prhs[5])!=1)
    mexErrMsgTxt("Sixth input must be a vector = x.");

  x=mxGetPr(prhs[5]);
  nx=mxGetNumberOfElements(prhs[5]);  /* size of x */
  p=mxGetPr(prhs[2]);
  np=mxGetNumberOfElements(prhs[2]);  /* size of p */
  q=mxGetPr(prhs[3]);
  nq=mxGetNumberOfElements(prhs[3]);  /* size of q */


  MAX=(int) *mxGetPr(prhs[0]);
  if (mxGetNumberOfElements(prhs[0])==2) K=(int) *(mxGetPr(prhs[0])+1);
     /* K is the second element of MAX, when MAX has 2 elements */
  else K=MAX;

  alpha=*mxGetPr(prhs[1]);
  n = (int) *mxGetPr(prhs[4]);

  plhs[0]=mxDuplicateArray(prhs[5]);   /* output is the same size as input */
  s=mxGetPr(plhs[0]);

  if (nlhs==2) {
     plhs[1]=mxCreateDoubleMatrix(1,MAX+1,mxREAL); 
     ss=mxGetPr(plhs[1]); 
  }
  else ss=(double*)mxMalloc(sizeof(double)*(MAX+1)); 

  if (K==0) return;

  l =(int*)mxMalloc(sizeof(int)*(MAX+2));
  memset(l,0,sizeof(int)*(MAX+2)); 
  lc =(int*)mxMalloc(sizeof(int)*(MAX+2));
  memset(lc,0,sizeof(int)*(MAX+2));
  z=(double*)mxMalloc(sizeof(double)*(MAX+1));
  z[1]=1;

  for (i=1;i<=MAX;i++) ss[i]=0;  /* set to zero, these are the coefficients of the polynomial */ 
  ss[0]=1; /* free term equals one */

  l[0]=MAX+1; /* purely for convenience */
  sl=0;
  i=1;

  while (i>0) {
     if ((sl<MAX) && (i<=n) && (l[i]<l[i-1])  && (l[i]<K) ) {
       l[i]++;
       lc[l[i]]++;        /* update conjugate partition */
       sl++;

       zn=1;
       c=-(i-1)/alpha+l[i]-1;
       for (j=0;j<np;j++)  zn*=p[j]+c;
       for (j=0;j<nq;j++)  zn/=q[j]+c;

       d=l[i]*alpha-i;             /* update j_lambda  */
       for (j=1;j<l[i];j++) {
          e=d-j*alpha+lc[j];
          g=e+1;
          zn*=(g-alpha)*e/(g*(e+alpha));
       }
       for (j=1;j<i;j++) {
          f=l[j]*alpha-j-d;
          g=f+alpha;
          e=f*g;
          zn*=(e-f)/(g+e);
       }
       zn*=(n+alpha*c);
       z[i]*=zn;
       ss[sl]+=z[i];

	    if ((sl<MAX)&&(i<n))
		   z[i+1]=z[i]; /* memcpy(z+(i+1)*nx,z+i*nx,nx * sizeof(double)); */
       i++;
     }
     else {
       sl-=l[i];
       for (j=1;j<l[i]+1;j++) lc[j]--;
       l[i]=0;
       i--;
     }
  } /* of while */

  for (j=0; j<nx; j++) {  
     s[j]=0; for(i=MAX;i>=0;i--) s[j]=ss[i]+s[j]*x[j];  
  }

  mxFree(z);
  mxFree(lc);
  mxFree(l);
  if (nlhs==1) mxFree(ss); 
}
