/* 2-D two-components de-coupled wavefield modeling using low-rank approximation on the base of 
 * original elastic anisotropic (1st-order) velocity-stress wave equation in VTI media.

   Copyright (C) 2014 Tongji University, Shanghai, China 
                      King Abdulah University of Science and Technology, Thuwal, Saudi Arabia
   Authors: Jiubing Cheng
     
   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.
             
   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.
                   
   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <rsf.hh>
#include <assert.h>

/* low rank decomposition  */
#include "vecmatop.hh"
#include "commoninc.hh"
#include "serialize.hh"

using namespace std;

#include "_cjb.h"
#include <rsf.h>
#include <fftw3.h>
/* head files aumatically produced from C programs */
extern "C"{
#include "zero.h"
#include "vti2tti.h"
#include "ricker.h"
#include "kykxkztaper.h"
#include "eigen2x2.h"
#include "fwpvtielowrank1st.h"
}

static std::valarray<float> vp, vs, ep, de, th;
static std::valarray<float> c11, c13, c15, c33, c35, c55;
static double dt1, dt2;
static double dxxh, dzzh;

static std::valarray<double> rkx, rkz, sinx, cosx;

/* dual-domain operators based on low-rank decomp. */

int samplepxx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplepxz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplepzz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplepxx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplepxz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplepzz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

int samplesxx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplesxz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleszz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplesxx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplesxz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleszz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

static void map2c1c(double *d, DblNumMat mat, int m, int n);
/*****************************************************************************************/
int main(int argc, char* argv[])
{
   sf_init(argc,argv);

   clock_t t1, t2, t3;
   float   timespent;

   t1=clock();

   iRSF par(0);
   int seed;
   par.get("seed",seed,time(NULL)); // seed for random number generator
   srand48(seed);

   float eps;
   par.get("eps",eps,1.e-6); // tolerance
       
   int npk;
   par.get("npk",npk,20); // maximum rank

   int   ns;
   float dt;
   par.get("ns",ns);
   par.get("dt",dt);
   dt1 = (double)dt;
   dt2 = (double)dt*dt;

   sf_warning("ns=%d dt=%f",ns,dt);
   sf_warning("npk=%d ",npk);
   sf_warning("eps=%f",eps);
   sf_warning("read velocity model parameters");

   /* setup I files */
   iRSF vp0, vs0("vs0"), epsi("epsi"), del("del"), the("the");

   /* Read/Write axes */
   int nxv, nzv;
   vp0.get("n1",nzv);
   vp0.get("n2",nxv);

   float az, ax;
   vp0.get("o1",az);
   vp0.get("o2",ax);

   float fx, fz;
   fx=ax*1000.0;
   fz=az*1000.0;

   float dx, dz;
   vp0.get("d1",az);
   vp0.get("d2",ax);
   dz = az*1000.0;
   dx = ax*1000.0;

   dxxh=(double)dx*0.5;
   dzzh=(double)dz*0.5;


   /* wave modeling space */
   int nx, nz, nxz;
   nx=nxv;
   nz=nzv;
   nxz=nx*nz;

   sf_warning("nx=%d nz=%d",nx,nz);
   sf_warning("dx=%f dz=%f",dx,dz);
   sf_warning("Warning: Staggered grid spectral need even-based FFT");

   vp.resize(nxz);
   vs.resize(nxz);
   ep.resize(nxz);
   de.resize(nxz);
   th.resize(nxz);
 
   vp0>>vp;
   vs0>>vs;
   epsi>>ep;
   del>>de;
   the>>th;

   float *c_11,*c_13,*c_15,*c_33,*c_35,*c_55;
   c_11 = sf_floatalloc(nxz);
   c_13 = sf_floatalloc(nxz);
   c_15 = sf_floatalloc(nxz);
   c_33 = sf_floatalloc(nxz);
   c_35 = sf_floatalloc(nxz);
   c_55 = sf_floatalloc(nxz);
   float *vp1,*vs1,*ep1,*de1,*th1;
   vp1 = sf_floatalloc(nxz);
   vs1 = sf_floatalloc(nxz);
   ep1 = sf_floatalloc(nxz);
   de1 = sf_floatalloc(nxz);
   th1 = sf_floatalloc(nxz);

   for(int i=0;i<nxz;i++)
   {
       vp1[i] = vp[i];
       vs1[i] = vs[i];
       de1[i] = de[i];
       ep1[i] = ep[i];
       th1[i] = th[i]*SF_PI/180;
   }
   Thomson2stiffness_2d(vp1,vs1,ep1,de1,th1,c_11,c_13,c_15,c_33,c_35,c_55,nx,nz);

   c11.resize(nxz);
   c13.resize(nxz);
   c15.resize(nxz);
   c33.resize(nxz);
   c35.resize(nxz);
   c55.resize(nxz);
   for(int i=0;i<nxz;i++)
   {
       c11[i] =c_11[i];
       c13[i] =c_13[i];
       c15[i] =c_15[i];
       c33[i] =c_33[i];
       c35[i] =c_35[i];
       c55[i] =c_55[i];
   }

   /* Fourier spectra demension */
   int nkz,nkx,nk;
   nkx=nx;
   nkz=nz;
   nk = nkx*nkz;

   float dkz,dkx,kz0,kx0;

   dkx=2*SF_PI/dx/nx;
   dkz=2*SF_PI/dz/nz;

   kx0=-SF_PI/dx;
   kz0=-SF_PI/dz;

   sf_warning("dkx=%f dkz=%f",dkx,dkz);

   rkx.resize(nk);
   rkz.resize(nk);
   sinx.resize(nk);
   cosx.resize(nk);

   float *akx = sf_floatalloc(nk);
   float *akz = sf_floatalloc(nk);

   double kx, kz, rk, k2;
   int    i=0, k=0, ix, iz;
   
   for(ix=0; ix < nkx; ix++)
   {
       kx = kx0+ix*dkx;

       for (iz=0; iz < nkz; iz++)
       {
            kz = kz0+iz*dkz;
            k2 = kx*kx+kz*kz;
            rk = sqrt(k2);

			akx[i] = kx;
			akz[i] = kz;

            rkx[i] = kx;
            rkz[i] = kz;

            sinx[i] = kx/rk;
            cosx[i] = kz/rk;
			if(rk==0) sinx[i]=0.0000001;
			if(rk==0) cosx[i]=0.0000001;

            i++;
       }
   }

   /*****************************************************************************
   *  Calculating de-couplep velocity fields extrapolators
   * ***************************************************************************/
   vector<int> md(nxz), nd(nk);
   for (k=0; k < nxz; k++)  md[k] = k;
   for (k=0; k < nk; k++)  nd[k] = k;

   vector<int> lid, rid;
   DblNumMat mid, mat;

   /********* low rank decomposition of operator: Bpx*Dx+ **********/
   int   m2pxx1, n2pxx1;
   double *ldatapxx1, *fmidpxx1, *rdatapxx1;

   iC( ddlowrank(nxz,nk,samplepxx1,eps,npk,lid,rid,mid) );
   m2pxx1=mid.m();
   n2pxx1=mid.n();
   sf_warning("m2pxx1=%d n2pxx1=%d",m2pxx1, n2pxx1);

   fmidpxx1  = (double*)malloc(sizeof(double)*m2pxx1*n2pxx1);
   ldatapxx1 = (double*)malloc(sizeof(double)*nxz*m2pxx1);
   rdatapxx1 = (double*)malloc(sizeof(double)*n2pxx1*nk);

   map2c1c(fmidpxx1, mid, m2pxx1, n2pxx1);

   iC ( samplepxx1(md,lid,mat) );
   map2c1c(ldatapxx1, mat, nxz, m2pxx1);

   iC ( samplepxx1(rid,nd,mat) );
   map2c1c(rdatapxx1, mat, n2pxx1, nk);
   
   /********* low rank decomposition of operator: Bpx*Dz- + BpxzDx- **********/
   int   m2pxz1, n2pxz1;
   double *ldatapxz1, *fmidpxz1, *rdatapxz1;

   iC( ddlowrank(nxz,nk,samplepxz1,eps,npk,lid,rid,mid) );
   m2pxz1=mid.m();
   n2pxz1=mid.n();
   sf_warning("m2pxz1=%d n2pxz1=%d",m2pxz1, n2pxz1);

   fmidpxz1  = (double*)malloc(sizeof(double)*m2pxz1*n2pxz1);
   ldatapxz1 = (double*)malloc(sizeof(double)*nxz*m2pxz1);
   rdatapxz1 = (double*)malloc(sizeof(double)*n2pxz1*nk);

   map2c1c(fmidpxz1, mid, m2pxz1, n2pxz1);

   iC ( samplepxz1(md,lid,mat) );
   map2c1c(ldatapxz1, mat, nxz, m2pxz1);

   iC ( samplepxz1(rid,nd,mat) );
   map2c1c(rdatapxz1, mat, n2pxz1, nk);
   
   /********* low rank decomposition of operator: Bpxz*Dz+ **********/
   int   m2pzz1, n2pzz1;
   double *ldatapzz1, *fmidpzz1, *rdatapzz1;

   iC( ddlowrank(nxz,nk,samplepzz1,eps,npk,lid,rid,mid) );
   m2pzz1=mid.m();
   n2pzz1=mid.n();
   sf_warning("m2pzz1=%d n2pzz1=%d",m2pzz1, n2pzz1);

   fmidpzz1  = (double*)malloc(sizeof(double)*m2pzz1*n2pzz1);
   ldatapzz1 = (double*)malloc(sizeof(double)*nxz*m2pzz1);
   rdatapzz1 = (double*)malloc(sizeof(double)*n2pzz1*nk);

   map2c1c(fmidpzz1, mid, m2pzz1, n2pzz1);

   iC ( samplepzz1(md,lid,mat) );
   map2c1c(ldatapzz1, mat, nxz, m2pzz1);

   iC ( samplepzz1(rid,nd,mat) );
   map2c1c(rdatapzz1, mat, n2pzz1, nk);
   
   /********* low rank decomposition of operator: Bpxz*Dx+ **********/
   int   m2pxx2, n2pxx2;
   double *ldatapxx2, *fmidpxx2, *rdatapxx2;

   iC( ddlowrank(nxz,nk,samplepxx2,eps,npk,lid,rid,mid) );
   m2pxx2=mid.m();
   n2pxx2=mid.n();
   sf_warning("m2pxx2=%d n2pxx2=%d",m2pxx2, n2pxx2);

   fmidpxx2  = (double*)malloc(sizeof(double)*m2pxx2*n2pxx2);
   ldatapxx2 = (double*)malloc(sizeof(double)*nxz*m2pxx2);
   rdatapxx2 = (double*)malloc(sizeof(double)*n2pxx2*nk);

   map2c1c(fmidpxx2, mid, m2pxx2, n2pxx2);

   iC ( samplepxx2(md,lid,mat) );
   map2c1c(ldatapxx2, mat, nxz, m2pxx2);

   iC ( samplepxx2(rid,nd,mat) );
   map2c1c(rdatapxx2, mat, n2pxx2, nk);
   
   /********* low rank decomposition of operator: Bpxz*Dz- + BpzDx- **********/
   int   m2pxz2, n2pxz2;
   double *ldatapxz2, *fmidpxz2, *rdatapxz2;

   iC( ddlowrank(nxz,nk,samplepxz2,eps,npk,lid,rid,mid) );
   m2pxz2=mid.m();
   n2pxz2=mid.n();
   sf_warning("m2pxz2=%d n2pxz2=%d",m2pxz2, n2pxz2);

   fmidpxz2  = (double*)malloc(sizeof(double)*m2pxz2*n2pxz2);
   ldatapxz2 = (double*)malloc(sizeof(double)*nxz*m2pxz2);
   rdatapxz2 = (double*)malloc(sizeof(double)*n2pxz2*nk);

   map2c1c(fmidpxz2, mid, m2pxz2, n2pxz2);

   iC ( samplepxz2(md,lid,mat) );
   map2c1c(ldatapxz2, mat, nxz, m2pxz2);

   iC ( samplepxz2(rid,nd,mat) );
   map2c1c(rdatapxz2, mat, n2pxz2, nk);
   
   /********* low rank decomposition of operator: Bpz*Dz+ **********/
   int   m2pzz2, n2pzz2;
   double *ldatapzz2, *fmidpzz2, *rdatapzz2;

   iC( ddlowrank(nxz,nk,samplepzz2,eps,npk,lid,rid,mid) );
   m2pzz2=mid.m();
   n2pzz2=mid.n();
   sf_warning("m2pzz2=%d n2pzz2=%d",m2pzz2, n2pzz2);

   fmidpzz2  = (double*)malloc(sizeof(double)*m2pzz2*n2pzz2);
   ldatapzz2 = (double*)malloc(sizeof(double)*nxz*m2pzz2);
   rdatapzz2 = (double*)malloc(sizeof(double)*n2pzz2*nk);

   map2c1c(fmidpzz2, mid, m2pzz2, n2pzz2);

   iC ( samplepzz2(md,lid,mat) );
   map2c1c(ldatapzz2, mat, nxz, m2pzz2);

   iC ( samplepzz2(rid,nd,mat) );
   map2c1c(rdatapzz2, mat, n2pzz2, nk);

   /********* low rank decomposition of operator: Bsx*Dx+ **********/
   int   m2sxx1, n2sxx1;
   double *ldatasxx1, *fmidsxx1, *rdatasxx1;

   iC( ddlowrank(nxz,nk,samplesxx1,eps,npk,lid,rid,mid) );
   m2sxx1=mid.m();
   n2sxx1=mid.n();
   sf_warning("m2sxx1=%d n2sxx1=%d",m2sxx1, n2sxx1);

   fmidsxx1  = (double*)malloc(sizeof(double)*m2sxx1*n2sxx1);
   ldatasxx1 = (double*)malloc(sizeof(double)*nxz*m2sxx1);
   rdatasxx1 = (double*)malloc(sizeof(double)*n2sxx1*nk);

   map2c1c(fmidsxx1, mid, m2sxx1, n2sxx1);

   iC ( samplesxx1(md,lid,mat) );
   map2c1c(ldatasxx1, mat, nxz, m2sxx1);

   iC ( samplesxx1(rid,nd,mat) );
   map2c1c(rdatasxx1, mat, n2sxx1, nk);
   
   /********* low rank decomposition of operator: Bsx*Dz- + BsxzDx- **********/
   int   m2sxz1, n2sxz1;
   double *ldatasxz1, *fmidsxz1, *rdatasxz1;

   iC( ddlowrank(nxz,nk,samplesxz1,eps,npk,lid,rid,mid) );
   m2sxz1=mid.m();
   n2sxz1=mid.n();
   sf_warning("m2sxz1=%d n2sxz1=%d",m2sxz1, n2sxz1);

   fmidsxz1  = (double*)malloc(sizeof(double)*m2sxz1*n2sxz1);
   ldatasxz1 = (double*)malloc(sizeof(double)*nxz*m2sxz1);
   rdatasxz1 = (double*)malloc(sizeof(double)*n2sxz1*nk);

   map2c1c(fmidsxz1, mid, m2sxz1, n2sxz1);

   iC ( samplesxz1(md,lid,mat) );
   map2c1c(ldatasxz1, mat, nxz, m2sxz1);

   iC ( samplesxz1(rid,nd,mat) );
   map2c1c(rdatasxz1, mat, n2sxz1, nk);
   
   /********* low rank decomposition of operator: Bsxz*Dz+ **********/
   int   m2szz1, n2szz1;
   double *ldataszz1, *fmidszz1, *rdataszz1;

   iC( ddlowrank(nxz,nk,sampleszz1,eps,npk,lid,rid,mid) );
   m2szz1=mid.m();
   n2szz1=mid.n();
   sf_warning("m2szz1=%d n2szz1=%d",m2szz1, n2szz1);

   fmidszz1  = (double*)malloc(sizeof(double)*m2szz1*n2szz1);
   ldataszz1 = (double*)malloc(sizeof(double)*nxz*m2szz1);
   rdataszz1 = (double*)malloc(sizeof(double)*n2szz1*nk);

   map2c1c(fmidszz1, mid, m2szz1, n2szz1);

   iC ( sampleszz1(md,lid,mat) );
   map2c1c(ldataszz1, mat, nxz, m2szz1);

   iC ( sampleszz1(rid,nd,mat) );
   map2c1c(rdataszz1, mat, n2szz1, nk);
   
   /********* low rank decomposition of operator: Bsxz*Dx+ **********/
   int   m2sxx2, n2sxx2;
   double *ldatasxx2, *fmidsxx2, *rdatasxx2;

   iC( ddlowrank(nxz,nk,samplesxx2,eps,npk,lid,rid,mid) );
   m2sxx2=mid.m();
   n2sxx2=mid.n();
   sf_warning("m2sxx2=%d n2sxx2=%d",m2sxx2, n2sxx2);

   fmidsxx2  = (double*)malloc(sizeof(double)*m2sxx2*n2sxx2);
   ldatasxx2 = (double*)malloc(sizeof(double)*nxz*m2sxx2);
   rdatasxx2 = (double*)malloc(sizeof(double)*n2sxx2*nk);

   map2c1c(fmidsxx2, mid, m2sxx2, n2sxx2);

   iC ( samplesxx2(md,lid,mat) );
   map2c1c(ldatasxx2, mat, nxz, m2sxx2);

   iC ( samplesxx2(rid,nd,mat) );
   map2c1c(rdatasxx2, mat, n2sxx2, nk);
   
   /********* low rank decomposition of operator: Bsxz*Dz- + BszDx- **********/
   int   m2sxz2, n2sxz2;
   double *ldatasxz2, *fmidsxz2, *rdatasxz2;

   iC( ddlowrank(nxz,nk,samplesxz2,eps,npk,lid,rid,mid) );
   m2sxz2=mid.m();
   n2sxz2=mid.n();
   sf_warning("m2sxz2=%d n2sxz2=%d",m2sxz2, n2sxz2);

   fmidsxz2  = (double*)malloc(sizeof(double)*m2sxz2*n2sxz2);
   ldatasxz2 = (double*)malloc(sizeof(double)*nxz*m2sxz2);
   rdatasxz2 = (double*)malloc(sizeof(double)*n2sxz2*nk);

   map2c1c(fmidsxz2, mid, m2sxz2, n2sxz2);

   iC ( samplesxz2(md,lid,mat) );
   map2c1c(ldatasxz2, mat, nxz, m2sxz2);

   iC ( samplesxz2(rid,nd,mat) );
   map2c1c(rdatasxz2, mat, n2sxz2, nk);
   
   /********* low rank decomposition of operator: Bsz*Dz+ **********/
   int   m2szz2, n2szz2;
   double *ldataszz2, *fmidszz2, *rdataszz2;

   iC( ddlowrank(nxz,nk,sampleszz2,eps,npk,lid,rid,mid) );
   m2szz2=mid.m();
   n2szz2=mid.n();
   sf_warning("m2szz2=%d n2szz2=%d",m2szz2, n2szz2);

   fmidszz2  = (double*)malloc(sizeof(double)*m2szz2*n2szz2);
   ldataszz2 = (double*)malloc(sizeof(double)*nxz*m2szz2);
   rdataszz2 = (double*)malloc(sizeof(double)*n2szz2*nk);

   map2c1c(fmidszz2, mid, m2szz2, n2szz2);

   iC ( sampleszz2(md,lid,mat) );
   map2c1c(ldataszz2, mat, nxz, m2szz2);

   iC ( sampleszz2(rid,nd,mat) );
   map2c1c(rdataszz2, mat, n2szz2, nk);

   t3=clock();
   timespent=(float)(t3-t2)/CLOCKS_PER_SEC;
   sf_warning("CPU time for low-rank decomp 2: %f(second)",timespent);

   /****************begin to calculate wavefield****************/
   /****************begin to calculate wavefield****************/
   /*  wavelet parameter for source definition */
   float A, f0, t0;
   f0=30.0;
   t0=0.04;
   A=1.0;

   sf_warning("fx=%f fz=%f dx=%f dz=%f",fx,fz,dx,dz);
   sf_warning("nx=%d nz=%d ", nx,nz);

   /* source definition */
   int ixs, izs;
   ixs=nxv/2;
   izs=nzv/2;

   /* setup I/O files */
   oRSF Elasticx("out");
   oRSF Elasticz("Elasticz");
   oRSF ElasticPx("ElasticPx");
   oRSF ElasticPz("ElasticPz");
   oRSF ElasticSx("ElasticSx");
   oRSF ElasticSz("ElasticSz");

   Elasticx.put("n1",nkz);
   Elasticx.put("n2",nkx);
   Elasticx.put("d1",dz/1000);
   Elasticx.put("d2",dx/1000);
   Elasticx.put("o1",fz/1000);
   Elasticx.put("o2",fx/1000);

   Elasticz.put("n1",nkz);
   Elasticz.put("n2",nkx);
   Elasticz.put("d1",dz/1000);
   Elasticz.put("d2",dx/1000);
   Elasticz.put("o1",fz/1000);
   Elasticz.put("o2",fx/1000);

   ElasticPx.put("n1",nkz);
   ElasticPx.put("n2",nkx);
   ElasticPx.put("d1",dz/1000);
   ElasticPx.put("d2",dx/1000);
   ElasticPx.put("o1",fz/1000);
   ElasticPx.put("o2",fx/1000);

   ElasticPz.put("n1",nkz);
   ElasticPz.put("n2",nkx);
   ElasticPz.put("d1",dz/1000);
   ElasticPz.put("d2",dx/1000);
   ElasticPz.put("o1",fz/1000);
   ElasticPz.put("o2",fx/1000);

   ElasticSx.put("n1",nkz);
   ElasticSx.put("n2",nkx);
   ElasticSx.put("d1",dz/1000);
   ElasticSx.put("d2",dx/1000);
   ElasticSx.put("o1",fz/1000);
   ElasticSx.put("o2",fx/1000);

   ElasticSz.put("n1",nkz);
   ElasticSz.put("n2",nkx);
   ElasticSz.put("d1",dz/1000);
   ElasticSz.put("d2",dx/1000);
   ElasticSz.put("o1",fz/1000);
   ElasticSz.put("o2",fx/1000);

   /********************* wavefield extrapolation *************************/
   double *ux=(double*)malloc(sizeof(double)*nxz);
   double *uz=(double*)malloc(sizeof(double)*nxz);

   double *txx1=(double*)malloc(sizeof(double)*nxz);
   double *tzz1=(double*)malloc(sizeof(double)*nxz);
   double *txz1=(double*)malloc(sizeof(double)*nxz);
   double *txx2=(double*)malloc(sizeof(double)*nxz);
   double *tzz2=(double*)malloc(sizeof(double)*nxz);
   double *txz2=(double*)malloc(sizeof(double)*nxz);

   double *ttx=(double*)malloc(sizeof(double)*nxz);
   double *ttz=(double*)malloc(sizeof(double)*nxz);
   double *ttxz=(double*)malloc(sizeof(double)*nxz);

   double *px1=(double*)malloc(sizeof(double)*nxz);
   double *pz1=(double*)malloc(sizeof(double)*nxz);
   double *px2=(double*)malloc(sizeof(double)*nxz);
   double *pz2=(double*)malloc(sizeof(double)*nxz);
   double *sx1=(double*)malloc(sizeof(double)*nxz);
   double *sz1=(double*)malloc(sizeof(double)*nxz);
   double *sx2=(double*)malloc(sizeof(double)*nxz);
   double *sz2=(double*)malloc(sizeof(double)*nxz);

   zero1double(ux, nxz);
   zero1double(uz, nxz);
   zero1double(sx1, nxz);
   zero1double(sz1, nxz);
   zero1double(sx2, nxz);
   zero1double(sz2, nxz);
   zero1double(px1, nxz);
   zero1double(pz1, nxz);
   zero1double(px2, nxz);
   zero1double(pz2, nxz);

   zero1double(txx1, nxz);
   zero1double(tzz1, nxz);
   zero1double(txz1, nxz);
   zero1double(txx2, nxz);
   zero1double(tzz2, nxz);
   zero1double(txz2, nxz);

   zero1double(ttx, nxz);
   zero1double(ttz, nxz);
   zero1double(ttxz, nxz);

   int *ijkx = sf_intalloc(nkx);
   int *ijkz = sf_intalloc(nkz);

   ikxikz(ijkx, ijkz, nkx, nkz);

   std::valarray<float> x(nxz);

   for(int it=0;it<ns;it++)
   {
        float t=it*dt;

        if(it%100==0)
                sf_warning("Elastic: it= %d  t=%f(s)",it,t);
 
        /* extrapolation of Txx-componet */
        fwpvti2de1stlr_rsg_double(ldatac11kx,rdatac11kx,fmidc11kx,ttx,ux,ijkx,ijkz,nx,nz,nxz,nk,m2c11kx,n2c11kx,dxxh,dzzh,akx,akz,0);
        fwpvti2de1stlr_rsg_double(ldatac13kz,rdatac13kz,fmidc13kz,ttz,uz,ijkx,ijkz,nx,nz,nxz,nk,m2c13kz,n2c13kz,dxxh,dzzh,akx,akz,0);
        for(i=0;i<nxz;i++)
			txx2[i] = txx1[i]+ dt*(ttx[i]+ttz[i]);

        /* extrapolation of Tzz-componet */
        fwpvti2de1stlr_rsg_double(ldatac13kx,rdatac13kx,fmidc13kx,ttx,ux,ijkx,ijkz,nx,nz,nxz,nk,m2c13kx,n2c13kx,dxxh,dzzh,akx,akz,0);
        fwpvti2de1stlr_rsg_double(ldatac33kz,rdatac33kz,fmidc33kz,ttz,uz,ijkx,ijkz,nx,nz,nxz,nk,m2c33kz,n2c33kz,dxxh,dzzh,akx,akz,0);
        for(i=0;i<nxz;i++)
			tzz2[i] = tzz1[i] + dt*(ttx[i]+ttz[i]);

        /* extrapolation of Txz-componet */
        fwpvti2de1stlr_rsg_double(ldatac44kz,rdatac44kz,fmidc44kz,ttx,ux,ijkx,ijkz,nx,nz,nxz,nk,m2c44kz,n2c44kz,dxxh,dzzh,akx,akz,0);
        fwpvti2de1stlr_rsg_double(ldatac44kx,rdatac44kx,fmidc44kx,ttz,uz,ijkx,ijkz,nx,nz,nxz,nk,m2c44kx,n2c44kx,dxxh,dzzh,akx,akz,0);
        for(i=0;i<nxz;i++)
			txz2[i] =  txz1[i] + dt*(ttx[i]+ttz[i]);

        // 2D exploding force source
        txx2[ixs*nz+izs]+=Ricker(t, f0, t0, A);
        tzz2[ixs*nz+izs]+=Ricker(t, f0, t0, A);

        /* extrapolation of Vpx-componet */
        fwpvti2de1stlr_rsg_double(ldatapxx1,rdatapxx1,fmidpxx1,ttx,txx2,ijkx,ijkz,nx,nz,nxz,nk,m2pxx1,n2pxx1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldatapxz1,rdatapxz1,fmidpxz1,ttxz,txz2,ijkx,ijkz,nx,nz,nxz,nk,m2pxz1,n2pxz1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldatapzz1,rdatapzz1,fmidpzz1,ttz,tzz2,ijkx,ijkz,nx,nz,nxz,nk,m2pzz1,n2pzz1,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			px2[i] = px1[i] + dt*(ttx[i]+ttxz[i]+ttz[i]);

        /* extrapolation of Vpz-componet */
        fwpvti2de1stlr_rsg_double(ldatapxx2,rdatapxx2,fmidpxx2,ttx,txx2,ijkx,ijkz,nx,nz,nxz,nk,m2pxx2,n2pxx2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldatapxz2,rdatapxz2,fmidpxz2,ttxz,txz2,ijkx,ijkz,nx,nz,nxz,nk,m2pxz2,n2pxz2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldatapzz2,rdatapzz2,fmidpzz2,ttz,tzz2,ijkx,ijkz,nx,nz,nxz,nk,m2pzz2,n2pzz2,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			pz2[i] = pz1[i] + dt*(ttx[i]+ttxz[i]+ttz[i]);

        /* extrapolation of Vsx-componet */
        fwpvti2de1stlr_rsg_double(ldatasxx1,rdatasxx1,fmidsxx1,ttx,txx2,ijkx,ijkz,nx,nz,nxz,nk,m2sxx1,n2sxx1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldatasxz1,rdatasxz1,fmidsxz1,ttxz,txz2,ijkx,ijkz,nx,nz,nxz,nk,m2sxz1,n2sxz1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldataszz1,rdataszz1,fmidszz1,ttz,tzz2,ijkx,ijkz,nx,nz,nxz,nk,m2szz1,n2szz1,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			sx2[i] = sx1[i] + dt*(ttx[i]+ttxz[i]+ttz[i]);

        /* extrapolation of Vsz-componet */
        fwpvti2de1stlr_rsg_double(ldatasxx2,rdatasxx2,fmidsxx2,ttx,txx2,ijkx,ijkz,nx,nz,nxz,nk,m2sxx2,n2sxx2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldatasxz2,rdatasxz2,fmidsxz2,ttxz,txz2,ijkx,ijkz,nx,nz,nxz,nk,m2sxz2,n2sxz2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg_double(ldataszz2,rdataszz2,fmidszz2,ttz,tzz2,ijkx,ijkz,nx,nz,nxz,nk,m2szz2,n2szz2,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			sz2[i] = sz1[i] + dt*(ttx[i]+ttxz[i]+ttz[i]);

        for(i=0;i<nxz;i++){
			ux[i] = px2[i] + sx2[i];
			uz[i] = pz2[i] + sz2[i];
		}

		
        /******* output wavefields: components******/
        if(it==ns-1)
        {
              for(i=0;i<nxz;i++) x[i]=ux[i];
              Elasticx<<x;
              for(i=0;i<nxz;i++) x[i]=uz[i];
              Elasticz<<x;
              for(i=0;i<nxz;i++) x[i]=px2[i];
              ElasticPx<<x;
              for(i=0;i<nxz;i++) x[i]=pz2[i];
              ElasticPz<<x;
              for(i=0;i<nxz;i++) x[i]=sx2[i];
              ElasticSx<<x;
              for(i=0;i<nxz;i++) x[i]=sz2[i];
              ElasticSz<<x;
        }
        /******* update the wavefield ********/
        if(it%100==0){
			sf_warning("ux=%f uz=%f ",ux[ixs*nz+izs],uz[ixs*nz+izs]);
			sf_warning("txx=%f tzz=%f txz=%f ",txx2[ixs*nz+izs],tzz2[ixs*nz+izs],txz2[ixs*nz+izs]);
		}
        for(i=0;i<nxz;i++){
                px1[i]=px2[i];
                pz1[i]=pz2[i];

                sx1[i]=sx2[i];
                sz1[i]=sz2[i];

				txx1[i]=txx2[i];
				txz1[i]=txz2[i];
				tzz1[i]=tzz2[i];
            }

   } //* it loop */
   t3=clock();
   timespent=(float)(t3-t2)/CLOCKS_PER_SEC;
   sf_warning("CPU time for wavefield extrapolation.: %f(second)",timespent);

   free(ldatac11kx);
   free(rdatac11kx);
   free(fmidc11kx);

   free(ldatac13kx);
   free(rdatac13kx);
   free(fmidc13kx);

   free(ldatac44kx);
   free(rdatac44kx);
   free(fmidc44kx);

   free(ldatac13kz);
   free(rdatac13kz);
   free(fmidc13kz);

   free(ldatac33kz);
   free(rdatac33kz);
   free(fmidc33kz);

   free(ldatac44kz);
   free(rdatac44kz);
   free(fmidc44kz);

   free(ldatapxx1);
   free(rdatapxx1);
   free(fmidpxx1);

   free(ldatapxx2);
   free(rdatapxx2);
   free(fmidpxx2);

   free(ldatapxz1);
   free(rdatapxz1);
   free(fmidpxz1);

   free(ldatapxz2);
   free(rdatapxz2);
   free(fmidpxz2);

   free(ldatapzz1);
   free(rdatapzz1);
   free(fmidpzz1);

   free(ldatapzz2);
   free(rdatapzz2);
   free(fmidpzz2);

   free(akx);
   free(akz);

   free(ux);
   free(uz);

   free(px1);
   free(px2);
   free(pz1);
   free(pz2);

   free(sx1);
   free(sx2);
   free(sz1);
   free(sz2);

   free(txx1);
   free(tzz1);
   free(txz1);
   free(txx2);
   free(tzz2);
   free(txz2);

   free(ttx);
   free(ttz);
   free(ttxz);

   free(ijkx);
   free(ijkz);

   exit(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C11*kx*exp(-i*kx*dx/2) */
int samplec11kxc15kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;

			c = c11[i]*kkx + c15[i]*kkz;
            
            /* operator: i*c*exp(-i*angle) = i*(c*cos(angle)-i*c*sin(angle) */
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C13*kx*exp(-i*kx*dx/2) */
int samplec13kxc35kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;

            c = c13[i]*kkx + c35[i]*kkz;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C44*kx*exp(i*kx*dx/2) */
int samplec44kxc35kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;

            c = c55[i]*kkx + c35[i]*kkz;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C13*kz*exp(-i*kz*dz/2) */
int samplecc13kzc15kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;

            c = c13[i]*kkz + c15[i]*kkx;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C33*kz*exp(-i*kz*dz/2) */
int samplecc33kzc35kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;

            c = c33[i]*kkz + c35[i]*kkx;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C44*kz*exp(i*kz*dz/2)  */
int samplecc44kzc15kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;
            c = c55[i]*kkz + c15[i]*kkx;
            
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kx*exp(i*kx*dx/2)*Apx*Apx */
int samplepxx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;
            c = u1*u1*kkx*sinc;
            
            resx(a,b) = c;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(-i*kz*dz/2)*Apx*Apx + i*kx*exp(-i*kx*dx/2)*Apx*Apz */
int samplepxz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c1 = u1*u1*kkz;
            c2 = u1*u2*kkx;
            
			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = (c1+c2)*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(i*kz*dz/2)*Apx*Apz */
int samplepzz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];
            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c = u1*u2*kkz;
            
			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kx*exp(i*kx*dx/2)*Apx*Apz */
int samplepxx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];
            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c = u1*u2*kkx;
			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(-i*kz*dz/2)*Apx*Apz + i*kx*exp(-i*kx*dx/2)*Apz*Apz */
int samplepxz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c1 = u1*u2*kkz;
            c2 = u2*u2*kkx;
			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = (c1+c2)*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(i*kz*dz/2)*Apz*Apz */
int samplepzz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];
            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            /* get the closest direction to k */
            if(u1*sx + u2*cx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c = u2*u2*kkz;
			double lam,sinc;
			lam = sqrt(va[0])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kx*exp(i*kx*dx/2)*Asx*Asx */
int samplesxx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];
            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[1][0];
            u2=ve[1][1];

            /* get the closest direction to k */
			if(u1*cx - u2*sx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c = u1*u1*kkx;
			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(-i*kz*dz/2)*Asx*Asx + i*kx*exp(-i*kx*dx/2)*Asx*Asz */
int samplesxz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[1][0];
            u2=ve[1][1];

            /* get the closest direction to k */
			if(u1*cx - u2*sx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c1 = u1*u1*kkz;
            c2 = u1*u2*kkx;
            
			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = (c1+c2)*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(i*kz*dz/2)*Asx*Asz */
int sampleszz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];
            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[1][0];
            u2=ve[1][1];

            /* get the closest direction to k */
			if(u1*cx - u2*sx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c = u1*u2*kkz;
            
			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kx*exp(i*kx*dx/2)*Asx*Asz */
int samplesxx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];
            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[1][0];
            u2=ve[1][1];

            /* get the closest direction to k */
			if(u1*cx - u2*sx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c = u1*u2*kkx;
            
			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(-i*kz*dz/2)*Asx*Asz + i*kx*exp(-i*kx*dx/2)*Asz*Asz */
int samplesxz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[1][0];
            u2=ve[1][1];

            /* get the closest direction to k */
			if(u1*cx - u2*sx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c1 = u1*u2*kkz;
            c2 = u2*u2*kkx;
			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = (c1+c2)*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*kz*exp(i*kz*dz/2)*Asz*Asz */
int sampleszz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = rkx[cs[b]];
            cx = rkz[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            // vector decomposition operators based on polarization
            a11= c11[i]*sx*sx+c55[i]*cx*cx+2*c15[i]*cx*sx;
            a12= (c13[i]+c55[i])*sx*cx+c15[i]*sx*sx+c35[i]*cx*cx;
            a22= c55[i]*sx*sx+c33[i]*cx*cx+2*c35[i]*sx*cx;

            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[0][0];
            u2=ve[0][1];
            aa[0][0] = a11;
            aa[0][1] = a12;
            aa[1][0] = a12;
            aa[1][1] = a22;

            dsolveSymmetric22(aa, ve, va);

            u1=ve[1][0];
            u2=ve[1][1];

            /* get the closest direction to k */
			if(u1*cx - u2*sx <0) {
               u1 = -u1;
               u2 = -u2;
            }

            c = u2*u2*kkz;
            
			double lam,sinc;
			lam = sqrt(va[1])*0.5*dt1;
			sinc = sin(lam)/lam;
            resx(a,b) = c*sinc;
              
         }// b loop
    }// a loop

    return 0;
}

static void map2c1c(double *d, DblNumMat mat, int m, int n)
{
   int i, j, k;

   k=0;
   for (i=0; i < m; i++)
   for (j=0; j < n; j++)
   {
        d[k] = (double)mat(i,j);
        k++;
   }

}
