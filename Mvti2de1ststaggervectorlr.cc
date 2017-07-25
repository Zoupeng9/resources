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
#include "ricker.h"
#include "kykxkztaper.h"
#include "eigen2x2.h"
#include "fwpvtielowrank1st.h"
}

static std::valarray<float> vp, vs, ep, de;
static double dt1, dt2;
static double dxxh, dzzh;

static std::valarray<double> rkx, rkz, sinx, cosx;

/* dual-domain operators based on low-rank decomp. */
int samplec11kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec13kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec44kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

int samplec13kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec33kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec44kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

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

static void map2c1c(float *d, DblNumMat mat, int m, int n);
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
   iRSF vp0, vs0("vs0"), epsi("epsi"), del("del");

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
 
   vp0>>vp;
   vs0>>vs;
   epsi>>ep;
   del>>de;

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
   *  Calculating stress fields extrapolators
   * ***************************************************************************/
   vector<int> md(nxz), nd(nk);
   for (k=0; k < nxz; k++)  md[k] = k;
   for (k=0; k < nk; k++)  nd[k] = k;

   vector<int> lid, rid;
   DblNumMat mid, mat;

   /********* low rank decomposition of operator: C11*Kx **********/
   int   m2c11kx, n2c11kx;
   float *ldatac11kx, *fmidc11kx, *rdatac11kx;

   iC( ddlowrank(nxz,nk,samplec11kx,eps,npk,lid,rid,mid) );
   m2c11kx=mid.m();
   n2c11kx=mid.n();
   sf_warning("m2c11kx=%d n2c11kx=%d",m2c11kx, n2c11kx);

   int j;

   fmidc11kx  = sf_floatalloc(m2c11kx*n2c11kx);
   ldatac11kx = sf_floatalloc(nxz*m2c11kx);
   rdatac11kx = sf_floatalloc(n2c11kx*nk);

   map2c1c(fmidc11kx, mid, m2c11kx, n2c11kx);

   iC ( samplec11kx(md,lid,mat) );
   map2c1c(ldatac11kx, mat, nxz, m2c11kx);

   iC ( samplec11kx(rid,nd,mat) );
   map2c1c(rdatac11kx, mat, n2c11kx, nk);
   
   /********* low rank decomposition of operator: C13*Kx **********/
   int   m2c13kx, n2c13kx;
   float *ldatac13kx, *fmidc13kx, *rdatac13kx;

   iC( ddlowrank(nxz,nk,samplec13kx,eps,npk,lid,rid,mid) );
   m2c13kx=mid.m();
   n2c13kx=mid.n();
   sf_warning("m2c13kx=%d n2c13kx=%d",m2c13kx, n2c13kx);

   fmidc13kx  = sf_floatalloc(m2c13kx*n2c13kx);
   ldatac13kx = sf_floatalloc(nxz*m2c13kx);
   rdatac13kx = sf_floatalloc(n2c13kx*nk);

   map2c1c(fmidc13kx, mid, m2c13kx, n2c13kx);

   iC ( samplec13kx(md,lid,mat) );
   map2c1c(ldatac13kx, mat, nxz, m2c13kx);

   iC ( samplec13kx(rid,nd,mat) );
   map2c1c(rdatac13kx, mat, n2c13kx, nk);

   /********* low rank decomposition of operator: C44*Kx **********/
   int   m2c44kx, n2c44kx;
   float *ldatac44kx, *fmidc44kx, *rdatac44kx;

   iC( ddlowrank(nxz,nk,samplec44kx,eps,npk,lid,rid,mid) );
   m2c44kx=mid.m();
   n2c44kx=mid.n();
   sf_warning("m2c44kx=%d n2c44kx=%d",m2c44kx, n2c44kx);

   for(j=0;j<m2c13kx*n2c13kx;j++)

   fmidc44kx  = sf_floatalloc(m2c44kx*n2c44kx);
   ldatac44kx = sf_floatalloc(nxz*m2c44kx);
   rdatac44kx = sf_floatalloc(n2c44kx*nk);

   map2c1c(fmidc44kx, mid, m2c44kx, n2c44kx);

   iC ( samplec44kx(md,lid,mat) );
   map2c1c(ldatac44kx, mat, nxz, m2c44kx);

   iC ( samplec44kx(rid,nd,mat) );
   map2c1c(rdatac44kx, mat, n2c44kx, nk);

   /********* low rank decomposition of operator: C13*Kz **********/
   int   m2c13kz, n2c13kz;
   float *ldatac13kz, *fmidc13kz, *rdatac13kz;

   iC( ddlowrank(nxz,nk,samplec13kz,eps,npk,lid,rid,mid) );
   m2c13kz=mid.m();
   n2c13kz=mid.n();
   sf_warning("m2c13kz=%d n2c13kz=%d",m2c13kz, n2c13kz);

   fmidc13kz  = sf_floatalloc(m2c13kz*n2c13kz);
   ldatac13kz = sf_floatalloc(nxz*m2c13kz);
   rdatac13kz = sf_floatalloc(n2c13kz*nk);

   map2c1c(fmidc13kz, mid, m2c13kz, n2c13kz);

   iC ( samplec13kz(md,lid,mat) );
   map2c1c(ldatac13kz, mat, nxz, m2c13kz);

   iC ( samplec13kz(rid,nd,mat) );
   map2c1c(rdatac13kz, mat, n2c13kz, nk);

   /********* low rank decomposition of operator: C33*Kz **********/
   int   m2c33kz, n2c33kz;
   float *ldatac33kz, *fmidc33kz, *rdatac33kz;

   iC( ddlowrank(nxz,nk,samplec33kz,eps,npk,lid,rid,mid) );
   m2c33kz=mid.m();
   n2c33kz=mid.n();
   sf_warning("m2c33kz=%d n2c33kz=%d",m2c33kz, n2c33kz);

   fmidc33kz  = sf_floatalloc(m2c33kz*n2c33kz);
   ldatac33kz = sf_floatalloc(nxz*m2c33kz);
   rdatac33kz = sf_floatalloc(n2c33kz*nk);

   map2c1c(fmidc33kz, mid, m2c33kz, n2c33kz);

   iC ( samplec33kz(md,lid,mat) );
   map2c1c(ldatac33kz, mat, nxz, m2c33kz);

   iC ( samplec33kz(rid,nd,mat) );
   map2c1c(rdatac33kz, mat, n2c33kz, nk);

   /********* low rank decomposition of operator: C44*Kz **********/
   int   m2c44kz, n2c44kz;
   float *ldatac44kz, *fmidc44kz, *rdatac44kz;

   iC( ddlowrank(nxz,nk,samplec44kz,eps,npk,lid,rid,mid) );
   m2c44kz=mid.m();
   n2c44kz=mid.n();
   sf_warning("m2c44kz=%d n2c44kz=%d",m2c44kz, n2c44kz);

   fmidc44kz  = sf_floatalloc(m2c44kz*n2c44kz);
   ldatac44kz = sf_floatalloc(nxz*m2c44kz);
   rdatac44kz = sf_floatalloc(n2c44kz*nk);

   map2c1c(fmidc44kz, mid, m2c44kz, n2c44kz);

   iC ( samplec44kz(md,lid,mat) );
   map2c1c(ldatac44kz, mat, nxz, m2c44kz);

   iC ( samplec44kz(rid,nd,mat) );
   map2c1c(rdatac44kz, mat, n2c44kz, nk);

   t2=clock();
   timespent=(float)(t2-t1)/CLOCKS_PER_SEC;
   sf_warning("CPU time for low-rank decomp 1: %f(second)",timespent);

   /*****************************************************************************
   *  Calculating de-couplep velocity fields extrapolators
   * ***************************************************************************/

   /********* low rank decomposition of operator: Bpx*Dx+ **********/
   int   m2pxx1, n2pxx1;
   float *ldatapxx1, *fmidpxx1, *rdatapxx1;

   iC( ddlowrank(nxz,nk,samplepxx1,eps,npk,lid,rid,mid) );
   m2pxx1=mid.m();
   n2pxx1=mid.n();
   sf_warning("m2pxx1=%d n2pxx1=%d",m2pxx1, n2pxx1);

   fmidpxx1  = sf_floatalloc(m2pxx1*n2pxx1);
   ldatapxx1 = sf_floatalloc(nxz*m2pxx1);
   rdatapxx1 = sf_floatalloc(n2pxx1*nk);

   map2c1c(fmidpxx1, mid, m2pxx1, n2pxx1);

   iC ( samplepxx1(md,lid,mat) );
   map2c1c(ldatapxx1, mat, nxz, m2pxx1);

   iC ( samplepxx1(rid,nd,mat) );
   map2c1c(rdatapxx1, mat, n2pxx1, nk);
   
   /********* low rank decomposition of operator: Bpx*Dz- + BpxzDx- **********/
   int   m2pxz1, n2pxz1;
   float *ldatapxz1, *fmidpxz1, *rdatapxz1;

   iC( ddlowrank(nxz,nk,samplepxz1,eps,npk,lid,rid,mid) );
   m2pxz1=mid.m();
   n2pxz1=mid.n();
   sf_warning("m2pxz1=%d n2pxz1=%d",m2pxz1, n2pxz1);

   fmidpxz1  = sf_floatalloc(m2pxz1*n2pxz1);
   ldatapxz1 = sf_floatalloc(nxz*m2pxz1);
   rdatapxz1 = sf_floatalloc(n2pxz1*nk);

   map2c1c(fmidpxz1, mid, m2pxz1, n2pxz1);

   iC ( samplepxz1(md,lid,mat) );
   map2c1c(ldatapxz1, mat, nxz, m2pxz1);

   iC ( samplepxz1(rid,nd,mat) );
   map2c1c(rdatapxz1, mat, n2pxz1, nk);
   
   /********* low rank decomposition of operator: Bpxz*Dz+ **********/
   int   m2pzz1, n2pzz1;
   float *ldatapzz1, *fmidpzz1, *rdatapzz1;

   iC( ddlowrank(nxz,nk,samplepzz1,eps,npk,lid,rid,mid) );
   m2pzz1=mid.m();
   n2pzz1=mid.n();
   sf_warning("m2pzz1=%d n2pzz1=%d",m2pzz1, n2pzz1);

   fmidpzz1  = sf_floatalloc(m2pzz1*n2pzz1);
   ldatapzz1 = sf_floatalloc(nxz*m2pzz1);
   rdatapzz1 = sf_floatalloc(n2pzz1*nk);

   map2c1c(fmidpzz1, mid, m2pzz1, n2pzz1);

   iC ( samplepzz1(md,lid,mat) );
   map2c1c(ldatapzz1, mat, nxz, m2pzz1);

   iC ( samplepzz1(rid,nd,mat) );
   map2c1c(rdatapzz1, mat, n2pzz1, nk);
   
   /********* low rank decomposition of operator: Bpxz*Dx+ **********/
   int   m2pxx2, n2pxx2;
   float *ldatapxx2, *fmidpxx2, *rdatapxx2;

   iC( ddlowrank(nxz,nk,samplepxx2,eps,npk,lid,rid,mid) );
   m2pxx2=mid.m();
   n2pxx2=mid.n();
   sf_warning("m2pxx2=%d n2pxx2=%d",m2pxx2, n2pxx2);

   fmidpxx2  = sf_floatalloc(m2pxx2*n2pxx2);
   ldatapxx2 = sf_floatalloc(nxz*m2pxx2);
   rdatapxx2 = sf_floatalloc(n2pxx2*nk);

   map2c1c(fmidpxx2, mid, m2pxx2, n2pxx2);

   iC ( samplepxx2(md,lid,mat) );
   map2c1c(ldatapxx2, mat, nxz, m2pxx2);

   iC ( samplepxx2(rid,nd,mat) );
   map2c1c(rdatapxx2, mat, n2pxx2, nk);
   
   /********* low rank decomposition of operator: Bpxz*Dz- + BpzDx- **********/
   int   m2pxz2, n2pxz2;
   float *ldatapxz2, *fmidpxz2, *rdatapxz2;

   iC( ddlowrank(nxz,nk,samplepxz2,eps,npk,lid,rid,mid) );
   m2pxz2=mid.m();
   n2pxz2=mid.n();
   sf_warning("m2pxz2=%d n2pxz2=%d",m2pxz2, n2pxz2);

   fmidpxz2  = sf_floatalloc(m2pxz2*n2pxz2);
   ldatapxz2 = sf_floatalloc(nxz*m2pxz2);
   rdatapxz2 = sf_floatalloc(n2pxz2*nk);

   map2c1c(fmidpxz2, mid, m2pxz2, n2pxz2);

   iC ( samplepxz2(md,lid,mat) );
   map2c1c(ldatapxz2, mat, nxz, m2pxz2);

   iC ( samplepxz2(rid,nd,mat) );
   map2c1c(rdatapxz2, mat, n2pxz2, nk);
   
   /********* low rank decomposition of operator: Bpz*Dz+ **********/
   int   m2pzz2, n2pzz2;
   float *ldatapzz2, *fmidpzz2, *rdatapzz2;

   iC( ddlowrank(nxz,nk,samplepzz2,eps,npk,lid,rid,mid) );
   m2pzz2=mid.m();
   n2pzz2=mid.n();
   sf_warning("m2pzz2=%d n2pzz2=%d",m2pzz2, n2pzz2);

   fmidpzz2  = sf_floatalloc(m2pzz2*n2pzz2);
   ldatapzz2 = sf_floatalloc(nxz*m2pzz2);
   rdatapzz2 = sf_floatalloc(n2pzz2*nk);

   map2c1c(fmidpzz2, mid, m2pzz2, n2pzz2);

   iC ( samplepzz2(md,lid,mat) );
   map2c1c(ldatapzz2, mat, nxz, m2pzz2);

   iC ( samplepzz2(rid,nd,mat) );
   map2c1c(rdatapzz2, mat, n2pzz2, nk);

   /********* low rank decomposition of operator: Bsx*Dx+ **********/
   int   m2sxx1, n2sxx1;
   float *ldatasxx1, *fmidsxx1, *rdatasxx1;

   iC( ddlowrank(nxz,nk,samplesxx1,eps,npk,lid,rid,mid) );
   m2sxx1=mid.m();
   n2sxx1=mid.n();
   sf_warning("m2sxx1=%d n2sxx1=%d",m2sxx1, n2sxx1);

   fmidsxx1  = sf_floatalloc(m2sxx1*n2sxx1);
   ldatasxx1 = sf_floatalloc(nxz*m2sxx1);
   rdatasxx1 = sf_floatalloc(n2sxx1*nk);

   map2c1c(fmidsxx1, mid, m2sxx1, n2sxx1);

   iC ( samplesxx1(md,lid,mat) );
   map2c1c(ldatasxx1, mat, nxz, m2sxx1);

   iC ( samplesxx1(rid,nd,mat) );
   map2c1c(rdatasxx1, mat, n2sxx1, nk);
   
   /********* low rank decomposition of operator: Bsx*Dz- + BsxzDx- **********/
   int   m2sxz1, n2sxz1;
   float *ldatasxz1, *fmidsxz1, *rdatasxz1;

   iC( ddlowrank(nxz,nk,samplesxz1,eps,npk,lid,rid,mid) );
   m2sxz1=mid.m();
   n2sxz1=mid.n();
   sf_warning("m2sxz1=%d n2sxz1=%d",m2sxz1, n2sxz1);

   fmidsxz1  = sf_floatalloc(m2sxz1*n2sxz1);
   ldatasxz1 = sf_floatalloc(nxz*m2sxz1);
   rdatasxz1 = sf_floatalloc(n2sxz1*nk);

   map2c1c(fmidsxz1, mid, m2sxz1, n2sxz1);

   iC ( samplesxz1(md,lid,mat) );
   map2c1c(ldatasxz1, mat, nxz, m2sxz1);

   iC ( samplesxz1(rid,nd,mat) );
   map2c1c(rdatasxz1, mat, n2sxz1, nk);
   
   /********* low rank decomposition of operator: Bsxz*Dz+ **********/
   int   m2szz1, n2szz1;
   float *ldataszz1, *fmidszz1, *rdataszz1;

   iC( ddlowrank(nxz,nk,sampleszz1,eps,npk,lid,rid,mid) );
   m2szz1=mid.m();
   n2szz1=mid.n();
   sf_warning("m2szz1=%d n2szz1=%d",m2szz1, n2szz1);

   fmidszz1  = sf_floatalloc(m2szz1*n2szz1);
   ldataszz1 = sf_floatalloc(nxz*m2szz1);
   rdataszz1 = sf_floatalloc(n2szz1*nk);

   map2c1c(fmidszz1, mid, m2szz1, n2szz1);

   iC ( sampleszz1(md,lid,mat) );
   map2c1c(ldataszz1, mat, nxz, m2szz1);

   iC ( sampleszz1(rid,nd,mat) );
   map2c1c(rdataszz1, mat, n2szz1, nk);
   
   /********* low rank decomposition of operator: Bsxz*Dx+ **********/
   int   m2sxx2, n2sxx2;
   float *ldatasxx2, *fmidsxx2, *rdatasxx2;

   iC( ddlowrank(nxz,nk,samplesxx2,eps,npk,lid,rid,mid) );
   m2sxx2=mid.m();
   n2sxx2=mid.n();
   sf_warning("m2sxx2=%d n2sxx2=%d",m2sxx2, n2sxx2);

   fmidsxx2  = sf_floatalloc(m2sxx2*n2sxx2);
   ldatasxx2 = sf_floatalloc(nxz*m2sxx2);
   rdatasxx2 = sf_floatalloc(n2sxx2*nk);

   map2c1c(fmidsxx2, mid, m2sxx2, n2sxx2);

   iC ( samplesxx2(md,lid,mat) );
   map2c1c(ldatasxx2, mat, nxz, m2sxx2);

   iC ( samplesxx2(rid,nd,mat) );
   map2c1c(rdatasxx2, mat, n2sxx2, nk);
   
   /********* low rank decomposition of operator: Bsxz*Dz- + BszDx- **********/
   int   m2sxz2, n2sxz2;
   float *ldatasxz2, *fmidsxz2, *rdatasxz2;

   iC( ddlowrank(nxz,nk,samplesxz2,eps,npk,lid,rid,mid) );
   m2sxz2=mid.m();
   n2sxz2=mid.n();
   sf_warning("m2sxz2=%d n2sxz2=%d",m2sxz2, n2sxz2);

   fmidsxz2  = sf_floatalloc(m2sxz2*n2sxz2);
   ldatasxz2 = sf_floatalloc(nxz*m2sxz2);
   rdatasxz2 = sf_floatalloc(n2sxz2*nk);

   map2c1c(fmidsxz2, mid, m2sxz2, n2sxz2);

   iC ( samplesxz2(md,lid,mat) );
   map2c1c(ldatasxz2, mat, nxz, m2sxz2);

   iC ( samplesxz2(rid,nd,mat) );
   map2c1c(rdatasxz2, mat, n2sxz2, nk);
   
   /********* low rank decomposition of operator: Bsz*Dz+ **********/
   int   m2szz2, n2szz2;
   float *ldataszz2, *fmidszz2, *rdataszz2;

   iC( ddlowrank(nxz,nk,sampleszz2,eps,npk,lid,rid,mid) );
   m2szz2=mid.m();
   n2szz2=mid.n();
   sf_warning("m2szz2=%d n2szz2=%d",m2szz2, n2szz2);

   fmidszz2  = sf_floatalloc(m2szz2*n2szz2);
   ldataszz2 = sf_floatalloc(nxz*m2szz2);
   rdataszz2 = sf_floatalloc(n2szz2*nk);

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
   float *ux=sf_floatalloc(nxz);
   float *uz=sf_floatalloc(nxz);

   float *txx=sf_floatalloc(nxz);
   float *tzz=sf_floatalloc(nxz);
   float *txz=sf_floatalloc(nxz);

   float *ttx=sf_floatalloc(nxz);
   float *ttz=sf_floatalloc(nxz);
   float *ttxz=sf_floatalloc(nxz);

   float *px1=sf_floatalloc(nxz);
   float *pz1=sf_floatalloc(nxz);
   float *px2=sf_floatalloc(nxz);
   float *pz2=sf_floatalloc(nxz);
   float *px3=sf_floatalloc(nxz);
   float *pz3=sf_floatalloc(nxz);
   float *sx1=sf_floatalloc(nxz);
   float *sz1=sf_floatalloc(nxz);
   float *sx2=sf_floatalloc(nxz);
   float *sz2=sf_floatalloc(nxz);
   float *sx3=sf_floatalloc(nxz);
   float *sz3=sf_floatalloc(nxz);

   zero1float(ux, nxz);
   zero1float(uz, nxz);
   zero1float(sx1, nxz);
   zero1float(sz1, nxz);
   zero1float(sx2, nxz);
   zero1float(sz2, nxz);
   zero1float(sx3, nxz);
   zero1float(sz3, nxz);
   zero1float(px1, nxz);
   zero1float(pz1, nxz);
   zero1float(px2, nxz);
   zero1float(pz2, nxz);
   zero1float(px3, nxz);
   zero1float(pz3, nxz);

   zero1float(txx, nxz);
   zero1float(tzz, nxz);
   zero1float(txz, nxz);

   zero1float(ttx, nxz);
   zero1float(ttz, nxz);
   zero1float(ttxz, nxz);

   int *ijkx = sf_intalloc(nkx);
   int *ijkz = sf_intalloc(nkz);

   ikxikz(ijkx, ijkz, nkx, nkz);

   std::valarray<float> x(nxz);

   /* Setting Stability Conditions, by Chenlong Wang & Zedong Wu */
   float fmax = 3*f0;
   float kxm, kzm, kxzm;
   float amin, bmin;
   amin = 99999999999;
   bmin = 99999999999;
   float c11, c33, c44;
   float c1144, c3344;
    i=0;
   for (ix=0; ix<nx; ix++)
    for (iz=0; iz<nz; iz++)
    {
        c33 = vp[i] * vp[i];
        c44 = vs[i] * vs[i];
        c11 = (1+2*ep[i])*c33;
        c1144 = c11 + c44;
        c3344 = c33 + c44;

        if (c1144<amin)
            amin = c1144;
        if (c3344<bmin)
            bmin = c3344;
        i++;
   }
   float kxmax = kx0 + nkx*dkx;
   float kzmax = kz0 + nkz*dkz;
   kxm = 2*sqrt(2)*SF_PI*fmax/sqrt(amin);
   kzm = 2*sqrt(2)*SF_PI*fmax/sqrt(bmin);
   float abmin = MIN(amin, bmin);
   kxzm = 2*sqrt(2)*SF_PI*fmax/sqrt(abmin);

   cerr<<"max kx="<<kxmax<<endl;
   cerr<<"max kz="<<kzmax<<endl;
   cerr<<"kxm="<<kxm<<endl;
   cerr<<"kzm="<<kzm<<endl;
   cerr<<"kxzm="<<kxzm<<endl;
 
   for(int it=0;it<ns;it++)
   {
        float t=it*dt;

        if(it%100==0)
                sf_warning("Elastic: it= %d  t=%f(s)",it,t);
 
        /* extrapolation of Txx-componet */
        fwpvti2de1stlr_rsg(ldatac11kx,rdatac11kx,fmidc11kx,ttx,ux,ijkx,ijkz,nx,nz,nxz,nk,m2c11kx,n2c11kx,dxxh,dzzh,akx,akz,0);
        fwpvti2de1stlr_rsg(ldatac13kz,rdatac13kz,fmidc13kz,ttz,uz,ijkx,ijkz,nx,nz,nxz,nk,m2c13kz,n2c13kz,dxxh,dzzh,akx,akz,0);
        for(i=0;i<nxz;i++)
			txx[i] = ttx[i]+ttz[i];

        /* extrapolation of Tzz-componet */
        fwpvti2de1stlr_rsg(ldatac13kx,rdatac13kx,fmidc13kx,ttx,ux,ijkx,ijkz,nx,nz,nxz,nk,m2c13kx,n2c13kx,dxxh,dzzh,akx,akz,0);
        fwpvti2de1stlr_rsg(ldatac33kz,rdatac33kz,fmidc33kz,ttz,uz,ijkx,ijkz,nx,nz,nxz,nk,m2c33kz,n2c33kz,dxxh,dzzh,akx,akz,0);
        for(i=0;i<nxz;i++)
			tzz[i] = ttx[i]+ttz[i];

        /* extrapolation of Txz-componet */
        fwpvti2de1stlr_rsg(ldatac44kz,rdatac44kz,fmidc44kz,ttx,ux,ijkx,ijkz,nx,nz,nxz,nk,m2c44kz,n2c44kz,dxxh,dzzh,akx,akz,0);
        fwpvti2de1stlr_rsg(ldatac44kx,rdatac44kx,fmidc44kx,ttz,uz,ijkx,ijkz,nx,nz,nxz,nk,m2c44kx,n2c44kx,dxxh,dzzh,akx,akz,0);
        for(i=0;i<nxz;i++)
			txz[i] = ttx[i]+ttz[i];

        // 2D exploding force source
        txx[ixs*nz+izs]+=Ricker(t, f0, t0, A);
        tzz[ixs*nz+izs]+=Ricker(t, f0, t0, A);

        /* extrapolation of Vpx-componet */
        fwpvti2de1stlr_rsg(ldatapxx1,rdatapxx1,fmidpxx1,ttx,txx,ijkx,ijkz,nx,nz,nxz,nk,m2pxx1,n2pxx1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatapxz1,rdatapxz1,fmidpxz1,ttxz,txz,ijkx,ijkz,nx,nz,nxz,nk,m2pxz1,n2pxz1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatapzz1,rdatapzz1,fmidpzz1,ttz,tzz,ijkx,ijkz,nx,nz,nxz,nk,m2pzz1,n2pzz1,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			px3[i] = 2*px2[i] - px1[i] + dt*dt*(ttx[i]+ttxz[i]+ttz[i]);

        /* extrapolation of Vpz-componet */
        fwpvti2de1stlr_rsg(ldatapxx2,rdatapxx2,fmidpxx2,ttx,txx,ijkx,ijkz,nx,nz,nxz,nk,m2pxx2,n2pxx2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatapxz2,rdatapxz2,fmidpxz2,ttxz,txz,ijkx,ijkz,nx,nz,nxz,nk,m2pxz2,n2pxz2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatapzz2,rdatapzz2,fmidpzz2,ttz,tzz,ijkx,ijkz,nx,nz,nxz,nk,m2pzz2,n2pzz2,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			pz3[i] = 2*pz2[i] - pz1[i] + dt*dt*(ttx[i]+ttxz[i]+ttz[i]);

        /* extrapolation of Vsx-componet */
        fwpvti2de1stlr_rsg(ldatasxx1,rdatasxx1,fmidsxx1,ttx,txx,ijkx,ijkz,nx,nz,nxz,nk,m2sxx1,n2sxx1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatasxz1,rdatasxz1,fmidsxz1,ttxz,txz,ijkx,ijkz,nx,nz,nxz,nk,m2sxz1,n2sxz1,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldataszz1,rdataszz1,fmidszz1,ttz,tzz,ijkx,ijkz,nx,nz,nxz,nk,m2szz1,n2szz1,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			sx3[i] = 2*sx2[i] - sx1[i] + dt*dt*(ttx[i]+ttxz[i]+ttz[i]);

        /* extrapolation of Vsz-componet */
        fwpvti2de1stlr_rsg(ldatasxx2,rdatasxx2,fmidsxx2,ttx,txx,ijkx,ijkz,nx,nz,nxz,nk,m2sxx2,n2sxx2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatasxz2,rdatasxz2,fmidsxz2,ttxz,txz,ijkx,ijkz,nx,nz,nxz,nk,m2sxz2,n2sxz2,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldataszz2,rdataszz2,fmidszz2,ttz,tzz,ijkx,ijkz,nx,nz,nxz,nk,m2szz2,n2szz2,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++) 
			sz3[i] = 2*sz2[i] - sz1[i] + dt*dt*(ttx[i]+ttxz[i]+ttz[i]);

        for(i=0;i<nxz;i++){
			ux[i] = px3[i] + sx3[i];
			uz[i] = pz3[i] + sz3[i];
		}

				  /*
        for(i=-1;i<=1;i++)
        for(j=-1;j<=1;j++)
        {
             if(fabs(i)+fabs(j)==2)
             {
                 // ux2[(ixs+i)*nz+(izs+j)]+=i*dt*Ricker(t, f0, t0, A);
                 // uz2[(ixs+i)*nz+(izs+j)]+=j*dt*Ricker(t, f0, t0, A);
                  txx2[(ixs+i)*nz+(izs+j)]+=i*Ricker(t, f0, t0, A);
                  tzz2[(ixs+i)*nz+(izs+j)]+=j*Ricker(t, f0, t0, A);
                  if(it%10==0)sf_warning("ux2=%f uz2=%f ",ux2[(ixs+i)*nz+(izs+j)],uz2[(ixs+i)*nz+(izs+j)]);
             }
        }
		*/
        /******* output wavefields: components******/
        if(it==ns-1)
        {
              for(i=0;i<nxz;i++) x[i]=ux[i];
              Elasticx<<x;
              for(i=0;i<nxz;i++) x[i]=uz[i];
              Elasticz<<x;
              for(i=0;i<nxz;i++) x[i]=px3[i];
              ElasticPx<<x;
              for(i=0;i<nxz;i++) x[i]=pz3[i];
              ElasticPz<<x;
              for(i=0;i<nxz;i++) x[i]=sx3[i];
              ElasticSx<<x;
              for(i=0;i<nxz;i++) x[i]=sz3[i];
              ElasticSz<<x;
        }
        /******* update the wavefield ********/
        if(it%100==0){
			sf_warning("ux=%f uz=%f ",ux[ixs*nz+izs],uz[ixs*nz+izs]);
			sf_warning("txx=%f tzz=%f txz=%f ",txx[ixs*nz+izs],tzz[ixs*nz+izs],txz[ixs*nz+izs]);
		}
        for(i=0;i<nxz;i++){
                px1[i]=px2[i];
                px2[i]=px3[i];
                pz1[i]=pz2[i];
                pz2[i]=pz3[i];

                sx1[i]=sx2[i];
                sx2[i]=sx3[i];
                sz1[i]=sz2[i];
                sz2[i]=sz3[i];
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
   free(px3);
   free(pz1);
   free(pz2);
   free(pz3);

   free(sx1);
   free(sx2);
   free(sx3);
   free(sz1);
   free(sz2);
   free(sz3);

   free(txx);
   free(tzz);
   free(txz);

   free(ttx);
   free(ttz);
   free(ttxz);

   free(ijkx);
   free(ijkz);

   exit(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C11*kx*exp(-i*kx*dx/2) */
int samplec11kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double  c11, c33;
	double  c;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            if(kkx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c11=ep2*c33;

			c = c11*kkx;
            
            /* operator: i*c*exp(-i*angle) = i*(c*cos(angle)-i*c*sin(angle) */
            resx(a,b) = c;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C13*kx*exp(-i*kx*dx/2) */
int samplec13kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double  c44, c33, c13c44, c13;
	double  c;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            if(kkx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));
			c13=c13c44-c44;

            c = c13*kkx;
            resx(a,b) = c;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C44*kx*exp(i*kx*dx/2) */
int samplec44kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double  c44;
    double  c;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vs2 = vs[i]*vs[i];

        c44=vs2;

        for(int b=0; b<nc; b++)
        {
            double kkx =rkx[cs[b]];
            if(kkx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c = c44*kkx;
            resx(a,b) = c;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C13*kz*exp(-i*kz*dz/2) */
int samplec13kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double  c44, c33, c13c44, c13;
	double  c;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            if(kkz==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));
			c13=c13c44-c44;

            c = c13*kkz;
            resx(a,b) = c;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C33*kz*exp(-i*kz*dz/2) */
int samplec33kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double  c33;
	double  c;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            if(kkz==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;

            c = c33*kkz;
            resx(a,b) = c;
              
         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator: i*C44*kz*exp(i*kz*dz/2)  */
int samplec44kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double  c44;
	double  c;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vs2 = vs[i]*vs[i];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            if(kkz==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c44=vs2;
            c = c44*kkz;
            resx(a,b) = c;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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

            c = u1*u1*kkx;
            
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            
            resx(a,b) = c1+c2;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            
            resx(a,b) = c;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            resx(a,b) = c;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            resx(a,b) = c1+c2;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            resx(a,b) = c;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            resx(a,b) = c;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            
            resx(a,b) = c1+c2;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            
            resx(a,b) = c;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            
            resx(a,b) = c;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c1, c2, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkx = rkx[cs[b]];
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            resx(a,b) = c1+c2;
              
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

    double  c11, c44, c33, c13c44;
	double  aa[2][2],ve[2][2],va[2];  /*matrix, eigeinvector and eigeinvalues*/
	double  a11, a12, a22, u1, u2;
	double  c, sx, cx;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];
        double vp2 = vp[i]*vp[i];
        double vs2 = vs[i]*vs[i];
        double de2 = 1.0+2*de[i];;
		double ep2 = 1.0+2*ep[i];

        for(int b=0; b<nc; b++)
        {
            double kkz = rkz[cs[b]];
            sx = sinx[cs[b]];
            cx = cosx[cs[b]];
            if(sx==0&&cx==0)
            {
               resx(a,b) = 0.0;
               continue;
            }

            c33=vp2;
            c44=vs2;
			c11=ep2*c33;
            c13c44=sqrt((de2*c33-c44)*(c33-c44));

            // vector decomposition operators based on polarization
            a11= c11*sx*sx+c44*cx*cx;
            a12= c13c44*sx*cx;
            a22= c44*sx*sx+c33*cx*cx;

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
            
            resx(a,b) = c;
              
         }// b loop
    }// a loop

    return 0;
}

static void map2c1c(float *d, DblNumMat mat, int m, int n)
{
   int i, j, k;

   k=0;
   for (i=0; i < m; i++)
   for (j=0; j < n; j++)
   {
        d[k] = (float)mat(i,j);
        k++;
   }

}
