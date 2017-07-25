/* 2-D two-components wavefield modeling using low-rank approximation on the base of 
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
#include "fwpvtielowrank1st.h"
}

static std::valarray<float> vp, vs, ep, de;
static double dt1, dt2;
static double dxxh, dzzh;

static std::valarray<double> rkx, rkz;

/* dual-domain operators based on low-rank decomp. */
int samplec11kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec13kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec44kx(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

int samplec13kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec33kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int samplec44kz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

void dz1stplus(float *z, float *z1, int *ijkx, int *ijkz, int nx, int nz, int nk);
void dz1stminus(float *z, float *z1, int *ijkx, int *ijkz, int nx, int nz, int nk);
void dx1stplus(float *x, float *x1, int *ijkx, int *ijkz, int nx, int nz, int nk);
void dx1stminus(float *x, float *x1, int *ijkx, int *ijkz, int nx, int nz, int nk);

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
   sf_warning("fx=%f fz=%f",fx,fz);
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

   float *akx = sf_floatalloc(nk);
   float *akz = sf_floatalloc(nk);

   float *kx = sf_floatalloc(nk);
   float *kz = sf_floatalloc(nk);
   int    i=0, k=0, ix, iz;
   
   for(ix=0; ix < nkx; ix++)
       for (iz=0; iz < nkz; iz++)
       {
			akx[i] = kx0+ix*dkx;
			akz[i] = kz0+iz*dkz;

            rkx[i] = kx0+ix*dkx;
            rkz[i] = kz0+iz*dkz;

           if(nx%2==0)
           {
               if(ix<=nx/2)
                   kx[i] = ix*dkx;
               else
                   kx[i] = (ix-nx)*dkx;
           }
           else
           {
               if(ix<=nx/2)
                   kx[i] = ix*dkx;
               else
                   kx[i] = (ix-nx)*dkx;
           }
           if(nz%2==0)
           {
               if(iz<=nz/2)
                   kz[i] = iz*dkz;
               else
                   kz[i] = (iz-nz)*dkz;
           }
           else
           {
               if(iz<=nz/2)
                   kz[i] = iz*dkz;
               else
                   kz[i] = (iz-nz)*dkz;
           }
            i++;
       }

   /*****************************************************************************
   *  Calculating polarization deviation operator for wave-mode separation
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
   sf_warning("CPU time for low-rank decomp: %f(second)",timespent);

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

   /********************* wavefield extrapolation *************************/
   float *vx1=sf_floatalloc(nxz);
   float *vz1=sf_floatalloc(nxz);
   float *vx2=sf_floatalloc(nxz);
   float *vz2=sf_floatalloc(nxz);


   float *txx1=sf_floatalloc(nxz);
   float *tzz1=sf_floatalloc(nxz);
   float *txz1=sf_floatalloc(nxz);
   float *txx2=sf_floatalloc(nxz);
   float *tzz2=sf_floatalloc(nxz);
   float *txz2=sf_floatalloc(nxz);

   float *ttx=sf_floatalloc(nxz);
   float *ttz=sf_floatalloc(nxz);

   zero1float(vx1, nxz);
   zero1float(vz1, nxz);
   zero1float(vx2, nxz);
   zero1float(vz2, nxz);

   zero1float(txx1, nxz);
   zero1float(tzz1, nxz);
   zero1float(txz1, nxz);

   zero1float(ttx, nxz);
   zero1float(ttz, nxz);

   int *ijkx = sf_intalloc(nkx);
   int *ijkz = sf_intalloc(nkz);

   ikxikz(ijkx, ijkz, nkx, nkz);

   std::valarray<float> x(nxz);

   for(int it=0;it<ns;it++)
   {
        float t=it*dt;

        if(it%100==0)
		{
                sf_warning("Elastic: it= %d  t=%f(s)",it,t);
				sf_warning("Elastic: vx= %f  vz=%f",vx2[nxz/2],vz2[nxz/2]);
		}
 
        /* extrapolation of Txx-componet */
        fwpvti2de1stlr_rsg(ldatac11kx,rdatac11kx,fmidc11kx,ttx,vx2,ijkx,ijkz,nx,nz,nxz,nk,m2c11kx,n2c11kx,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatac13kz,rdatac13kz,fmidc13kz,ttz,vz2,ijkx,ijkz,nx,nz,nxz,nk,m2c13kz,n2c13kz,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++)
			txx2[i] = dt*(ttx[i]+ttz[i]) + txx1[i];
        zero1float(ttx, nxz);
        zero1float(ttz, nxz);

        /* extrapolation of Tzz-componet */
        fwpvti2de1stlr_rsg(ldatac13kx,rdatac13kx,fmidc13kx,ttx,vx2,ijkx,ijkz,nx,nz,nxz,nk,m2c13kx,n2c13kx,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatac33kz,rdatac33kz,fmidc33kz,ttz,vz2,ijkx,ijkz,nx,nz,nxz,nk,m2c33kz,n2c33kz,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++)
			tzz2[i] = dt*(ttx[i]+ttz[i]) + tzz1[i];
        zero1float(ttx, nxz);
        zero1float(ttz, nxz);

        /* extrapolation of Txz-componet */
        fwpvti2de1stlr_rsg(ldatac44kz,rdatac44kz,fmidc44kz,ttx,vx2,ijkx,ijkz,nx,nz,nxz,nk,m2c44kz,n2c44kz,dxxh,dzzh,akx,akz,1);
        fwpvti2de1stlr_rsg(ldatac44kx,rdatac44kx,fmidc44kx,ttz,vz2,ijkx,ijkz,nx,nz,nxz,nk,m2c44kx,n2c44kx,dxxh,dzzh,akx,akz,1);
        for(i=0;i<nxz;i++)
			txz2[i] = dt*(ttx[i]+ttz[i]) + txz1[i];

        zero1float(ttx, nxz);
        zero1float(ttz, nxz);

        // 2D exploding force source
        txx2[ixs*nz+izs]+=Ricker(t, f0, t0, A);
        txx2[(ixs+1)*nz+izs+1]+=Ricker(t, f0, t0, A);
        tzz2[ixs*nz+izs]+=Ricker(t, f0, t0, A);
        tzz2[(ixs+1)*nz+izs+1]+=Ricker(t, f0, t0, A);

        /* extrapolation of Vx-componet */
        dx1stplus(txx2,ttx,ijkx,ijkz,nx,nz,nk);
        dz1stminus(txz2,ttz,ijkx,ijkz,nx,nz,nk);
        for(i=0;i<nxz;i++)
			vx2[i] = vx1[i] + dt*(ttx[i]+ttz[i]);

        zero1float(ttx, nxz);
        zero1float(ttz, nxz);

        /* extrapolation of Vz-componet */
        dx1stminus(txz2,ttx,ijkx,ijkz,nx,nz,nk);
        dz1stplus(tzz2,ttz,ijkx,ijkz,nx,nz,nk);
        for(i=0;i<nxz;i++)
			vz2[i] = vz1[i] + dt*(ttx[i]+ttz[i]);

        /******* output wavefields: components******/
       if(it==ns-1)
        {
              for(i=0;i<nxz;i++) x[i]=vx2[i];
              Elasticx<<x;

              for(i=0;i<nxz;i++) x[i]=vz2[i];
              Elasticz<<x;
        }

        /******* update the wavefield ********/
        for(i=0;i<nxz;i++)
		{
			vx1[i]=vx2[i];
            vz1[i]=vz2[i];
			txx1[i]=txx2[i];
			tzz1[i]=tzz2[i];
			txz1[i]=txz2[i];
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

   free(akx);
   free(akz);

   free(vx1);
   free(vx2);
   free(vz1);
   free(vz2);

   free(txx1);
   free(tzz1);
   free(txz1);
   free(txx2);
   free(tzz2);
   free(txz2);

   free(ttx);
   free(ttz);

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
/* operation apply to stress component */
void dx1stplus(float *x, float *x1, int *ijkx, int *ijkz, int nx, int nz, int nk)
{
  int index, ikx, ikz, k, i;
  float kx,kz,angle;

  sf_complex *xin, *xout;
  sf_complex c;

  fftwf_plan xp;
  fftwf_plan xpi;

  xin=sf_complexalloc(nk);
  xout=sf_complexalloc(nk);

  xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

  xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

  /* FFT: from (x,z) to (kx, kz) domain */
  for(i=0;i<nk;i++) xin[i]=sf_cmplx(x[i],0.0);

  fftwf_execute(xp);

  k=0;
  for(ikx=0;ikx<nx;ikx++)
  {
       int ixnz=ijkx[ikx]*nz;
       for(ikz=0;ikz<nz;ikz++)
       {
         index = ixnz + ijkz[ikz];
         kx = rkx[index];
         kz = rkz[index];
		 angle = kx*dxxh - kz*dzzh;
		 //angle = kx*dxxh;

         c = sf_cmplx(-kx*sin(angle), kx*cos(angle));

         xin[k] = sf_cmul(c, xout[k]);
         		 
		 k++;
       }// ikz loop
  }//ikx loop

  // (kx,kz) to (x, z) domain
  fftwf_execute(xpi);

  for(k=0;k<nk;k++)
    x1[k] = xout[k].r/nk;
    //x1[k] = creal(xout[k])/nk;
  
  fftwf_destroy_plan(xp);
  fftwf_destroy_plan(xpi);
  free(xin);
  free(xout);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operation apply to stress component */
void dx1stminus(float *x, float *x1, int *ijkx, int *ijkz, int nx, int nz, int nk)
{
  int index, ikx, ikz, k, i;
  float kx, kz, angle;

  sf_complex *xin, *xout;
  sf_complex c;

  fftwf_plan xp;
  fftwf_plan xpi;

  xin=sf_complexalloc(nk);
  xout=sf_complexalloc(nk);

  xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

  xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

  /* FFT: from (x,z) to (kx, kz) domain */
  for(i=0;i<nk;i++) xin[i]=sf_cmplx(x[i],0.0);

  fftwf_execute(xp);

  k=0;
  for(ikx=0;ikx<nx;ikx++)
  {
       int ixnz=ijkx[ikx]*nz;
       for(ikz=0;ikz<nz;ikz++)
       {
         index = ixnz + ijkz[ikz];
         kx = rkx[index];
         kz = rkz[index];
		 angle = kx*dxxh - kz*dzzh;
		 //angle = -kx*dxxh;

         c = sf_cmplx(-kx*sin(angle), kx*cos(angle));

         xin[k] = sf_cmul(c, xout[k]);
         		 
		 k++;
       }// ikz loop
  }//ikx loop

  // (kx,kz) to (x, z) domain
  fftwf_execute(xpi);

  for(k=0;k<nk;k++)
    x1[k] = xout[k].r/nk;
    //x1[k] = creal(xout[k])/nk;
  
  fftwf_destroy_plan(xp);
  fftwf_destroy_plan(xpi);
  free(xin);
  free(xout);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operation apply to stress component */
void dz1stplus(float *z, float *z1, int *ijkx, int *ijkz, int nx, int nz, int nk)
{
  int index, ikx, ikz, k, i;
  float kx, kz, angle;

  sf_complex *xin, *xout;
  sf_complex c;

  fftwf_plan xp;
  fftwf_plan xpi;

  xin=sf_complexalloc(nk);
  xout=sf_complexalloc(nk);

  xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

  xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

  /* FFT: from (x,z) to (kx, kz) domain */
  for(i=0;i<nk;i++) xin[i]=sf_cmplx( z[i],0.0);

  fftwf_execute(xp);

  k=0;
  for(ikx=0;ikx<nx;ikx++)
  {
       int ixnz=ijkx[ikx]*nz;
       for(ikz=0;ikz<nz;ikz++)
       {
         index = ixnz + ijkz[ikz];
         kx = rkx[index];
         kz = rkz[index];
		 angle = kx*dxxh - kz*dzzh;
		 //angle = kz*dzzh;

         c = sf_cmplx(-kz*sin(angle), kz*cos(angle));
         
         xin[k] = sf_cmul(c, xout[k]);
         		 
		 k++;
       }// ikz loop
  }//ikx loop

  // (kx,kz) to (x, z) domain
  fftwf_execute(xpi);

  for(k=0;k<nk;k++)
    z1[k] = xout[k].r/nk;
    //z1[k] = creal(xout[k])/nk;
  
  fftwf_destroy_plan(xp);
  fftwf_destroy_plan(xpi);
  free(xin);
  free(xout);

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operation apply to stress component */
void dz1stminus(float *z, float *z1, int *ijkx, int *ijkz, int nx, int nz, int nk)
{
  int index, ikx, ikz, k, i;
  float kx, kz, angle;

  sf_complex *xin, *xout;
  sf_complex c;

  fftwf_plan xp;
  fftwf_plan xpi;

  xin=sf_complexalloc(nk);
  xout=sf_complexalloc(nk);

  xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

  xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

  /* FFT: from (x,z) to (kx, kz) domain */
  for(i=0;i<nk;i++) xin[i]=sf_cmplx( z[i],0.0);

  fftwf_execute(xp);

  k=0;
  for(ikx=0;ikx<nx;ikx++)
  {
       int ixnz=ijkx[ikx]*nz;
       for(ikz=0;ikz<nz;ikz++)
       {
         index = ixnz + ijkz[ikz];
         kx = rkx[index];
         kz = rkz[index];
		 angle = kx*dxxh -  kz*dzzh;
		 //angle = -kz*dzzh;

         c = sf_cmplx(-kz*sin(angle), kz*cos(angle));
         
         xin[k] = sf_cmul(c, xout[k]);
         		 
		 k++;
       }// ikz loop
  }//ikx loop

  // (kx,kz) to (x, z) domain
  fftwf_execute(xpi);

  for(k=0;k<nk;k++)
    z1[k] = xout[k].r/nk;
    //z1[k] = creal(xout[k])/nk;
  
  fftwf_destroy_plan(xp);
  fftwf_destroy_plan(xpi);
  free(xin);
  free(xout);

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
