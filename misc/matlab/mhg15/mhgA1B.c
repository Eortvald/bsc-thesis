/* 
 MEX function [s,ss]=mhgA1B(MAX,p,q,x,y).
 Computes the truncated hypergeometric function of one or two matrix arguments
 pFq^1(p;q;x,y). This is only the alpha=1 case!
 The sum is only over partitions |kappa|<=MAX(1); and kappa(1)<=MAX(2)
 (thus all kappa(i)<=MAX(2))
 MAX(2) may be omitted.
 y may be omitted.
 
 p and q are arrays, so mhgA1B([30 10],[3 4],[5 6 7],[0.5 0.6]) is 
 _2F_3^1([3 4];[5 6 7];[0.5,0.6]) summed over all partitions |kappa|<=30
 with kappa(i)<=10

 by Plamen Koev, January 2007
 
 May 2007: Introduced Raymond Kan's faster Q_kappa update. Allowed
 two matrix arguments.
 
 Implements the second algorithm in the paper 
 "On Computing Schur Functions and Series ThereOf"
 by Chan, Drensky, Edelman, Kan, and Koev.

*/

#include <mex.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

void WorkB(double *b, int hmn, int m, int n, int L, int ii) {
int i,j,r,k;
double t,xx,e;

       /* erase row ii + L of B */
       for (j=1;j<=ii-1;j++) {
          xx=b[hmn + (ii+L)*n + j-1];
          k=ii-j+1;
          t=b[hmn + (ii+L)*n + j];
          b[hmn + (ii+L)*n + j]+=xx;
          r=j+1;
          while ((r+k+L<=m) & (r<=n)) {
             e=b[hmn + (k+r-1+L)*n + r-1]/b[hmn + (r+k-2+L)*n + r-1];
             xx*=e;
             b[hmn + (k+r-1+L)*n + r-1]=e*t;
             if (r+1<=n) {
                t=b[hmn + (r+k-1+L)*n + r];
                b[hmn + (r+k-1+L)*n + r]+=xx;
             }
             r++;
          }
          b[hmn + (ii+L)*n+j-1]*=b[hmn + (ii-1+L)*n+j-1];       
       }
       for (j=n+1;j>=ii+1;j--) {
          b[hmn + (j-1+L)*n + j-2]*=b[hmn + (j-2+L)*n + j-2];           
          if (j<=n) b[hmn + (j-1+L)*n + j-1]/=b[hmn + (j-1+L)*n + j-1-1];
       }
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[]) {
  int XY, i, MAX, n, np, nq, *l, *lc, j, sl, K, holes,m,hmn;
  double *x, *y, *p, *q, *s, *ss, *z, *bx, *by, zn, dn, d, e, f, g, xx, t, yy;

  if ((nrhs<4) || (nrhs>5)) mexErrMsgTxt("Expect 4 or 5 input arguments.");
 
  if (nlhs>2) mexErrMsgTxt("Too many output arguments.");
  
  if (!mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || 
      !mxIsDouble(prhs[2]) || !mxIsDouble(prhs[3]) || 
      ((nrhs==5) && !mxIsDouble(prhs[4])))
     mexErrMsgTxt("All inputs must be double arrays.");

  if (mxGetM(prhs[0])!=1 || mxGetN(prhs[0])<1 || mxGetN(prhs[0])>2)
    mexErrMsgTxt("First input must be an array of size 1 or 2 = [max size of partition, max size of the largest part].");
  if (mxGetM(prhs[1])>1 && mxGetN(prhs[1])>1)
    mexErrMsgTxt("Second input must be a vector = (a_1,...,a_p).");
  if (mxGetM(prhs[2])>1 && mxGetN(prhs[2])>1)
    mexErrMsgTxt("Third input must be a vector = (b_1,...,b_q).");
  if (mxGetM(prhs[3])!=1 && mxGetN(prhs[3])!=1)
    mexErrMsgTxt("Fourth input must be a vector = x.");

  x=mxGetPr(prhs[3]);
  n=mxGetNumberOfElements(prhs[3]);  /* size of x */

  XY=(nrhs==5);
  if (XY) y=mxGetPr(prhs[4]);
 
  p=mxGetPr(prhs[1]);
  np=mxGetNumberOfElements(prhs[1]);  /* size of p */
  q=mxGetPr(prhs[2]);
  nq=mxGetNumberOfElements(prhs[2]);  /* size of q */

  MAX=(int) *mxGetPr(prhs[0]);
  if (mxGetNumberOfElements(prhs[0])==2) K=(int) *(mxGetPr(prhs[0])+1);
     /* K is the second element of MAX, when MAX has 2 elements */
  else K=MAX;

/*  plhs[0]=mxDuplicateArray(prhs[3]); /*output is the same size as input*/

  plhs[0]=mxCreateDoubleScalar(1);   /* output is scalar */
  s=mxGetPr(plhs[0]); 

  if (nlhs==2) {
     plhs[1]=mxCreateDoubleMatrix(1,MAX+1,mxREAL); 
     ss=mxGetPr(plhs[1]); 
  }
  else ss=(double*)mxMalloc(sizeof(double)*(MAX+1)); 

  if (K==0) return;

  i=MAX+2; if (i<n+2) i=n+2;
  l =(int*)mxMalloc(i*sizeof(int));
  memset(l,0,i*sizeof(int));
  lc =(int*)mxMalloc(i*sizeof(int));
  memset(lc,0,i*sizeof(int));
  z=(double*)mxMalloc(sizeof(double)*(MAX+2));
  z[1]=1;

  for (i=1;i<=MAX;i++) ss[i]=0;  /* set to zero, these are the coefficients of the polynomial */ 
  ss[0]=1; /* free term equals one */

  l[0]=MAX+1; /* purely for convenience; but once so, l[0] must be >=MAX+1, not just MAX!*/
  sl=0;     /* this is sum(l); */

  m=n+MAX; /* B is m-by-n */

  bx=(double*)mxMalloc((n+1)*m*n*sizeof(double));
  memset(bx,0,(n+1)*m*n*sizeof(double));

  for (j=0;j<n;j++) {            /* stuffing b with initial data */
     bx[0 + j*n + j]=1;
     for (i=j+1;i<m;i++) bx[0 + i*n + j]=x[j];
  }
  
  if (XY){
      by=(double*)mxMalloc((n+1)*m*n*sizeof(double));
      memset(by,0,(n+1)*m*n*sizeof(double));
      for (j=0;j<n;j++) {            
          by[0 + j*n + j]=1;
          for (i=j+1;i<m;i++) by[0 + i*n + j]=y[j];
      }
  }

  holes=0;

  i=1;
  while (i>0) {
     if ((sl<MAX) && (l[i]<n) && (l[i]<l[i-1])  && (i<=K) ) {
       l[i]++;
       lc[l[i]]++;        /* update conjugate partition */
       sl++;
       
/*       printf("lc=");
       for (j=1;j<=n;j++) printf("%d ",lc[j]); printf("\n");
*/
       
       zn=1;
       dn=1;
       d=i-l[i];                           /* d=lc[l[i]]-l[i]; */
       for (j=0;j<np;j++) zn*=p[j]+d;
       for (j=0;j<nq;j++) dn*=q[j]+d;
 
       if (XY) z[i]*=zn/(dn*(n+d));
       else {
           for (j=1;j<i;j++) {
               e=l[j]-j+d;
               zn*=e;
               dn*=e+1;
           }
           z[i]*=zn/(dn*l[i]);
       }
       
       holes += (l[i]<l[i-1]) - (l[i]-1<l[i-1]) 
              + (l[i+1]<l[i]) - (l[i+1]<l[i]-1);

       hmn=holes*m*n;
       if ((i-lc[l[i]+1])==1) {
           memcpy(bx+hmn,bx+hmn-m*n,m*n*sizeof(double));
           if (XY) memcpy(by+hmn,by+hmn-m*n,m*n*sizeof(double));
       }
       /* i=lc[l[i]] */

       WorkB(bx,hmn,m,n,lc[1]-1,n-l[i]+1);
       yy=1; for (j=0;j<n;j++) yy*=bx[hmn + (j+lc[n-j])*n + j];
       /*  we have y=schurp(lc,x); */
       
       if (XY) {   
            WorkB(by,hmn,m,n,lc[1]-1,n-l[i]+1);
            for (j=0;j<n;j++) yy*=by[hmn + (j+lc[n-j])*n + j];
       }
       
       ss[sl]+=z[i]*yy; 
       
       z[i+1]=z[i];
       i++;
     }
     else {
       if ((i>1) && (l[i]>0)) holes -= (l[i]<l[i-1]);
       sl-=l[i];
       for (j=1;j<=l[i];j++) lc[j]--;
       l[i]=0;
       i--;
     }
  } /* of while */
  
  *s=0; for (i=0;i<=MAX;i++) *s=*s+ss[i];

  if (nlhs==1) mxFree(ss);
  if (XY) mxFree(by);
  mxFree(bx);
  mxFree(z);
  mxFree(lc);
  mxFree(l);
}
