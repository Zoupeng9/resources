/* 3-D two-components elastic wavefield extrapolation 
   using low-rank approximate PS solution on the base of 
   displacement wave equation in TI media.

   Copyright (C) 2014 Tongji University, Shanghai, China 
   Authors: Jiubing Cheng
   modified by Peng Zou in 2015
     
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

/* prepared head files by myself */
#include "_cjb.h"
#include<fftw3.h>
#include<omp.h>

/* head files aumatically produced from C programs */
extern "C"{
#include "zero.h"
#include "ricker.h"
#include "kykxkztaper.h"
#include "fwpvtielowrank.h"
#include "decomplowrank.h"
#include "vti2tti.h"
#include "eigen3x3.h"
}

static float *c11,*c12,*c13,*c14,*c15,*c16,
			      *c22,*c23,*c24,*c25,*c26,
				       *c33,*c34,*c35,*c36,
					        *c44,*c45,*c46,
						         *c55,*c56,
							          *c66;
static double dt1, dt2;

static std::valarray<double> rkx,rky,rkz,rk2;
static std::valarray<float> vp, vs, ep, de, ga, th, ph;

/* dual-domain operators based on low-rank decomp. */
int sampleaxx(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleayy(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleazz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleaxy(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleaxz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleayz(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

static void map2d1d(float *d, DblNumMat mat, int m, int n);
/*****************************************************************************************/
int main(int argc, char* argv[])
{
   sf_init(argc,argv);
   fftwf_init_threads();
   omp_set_num_threads(10);
   
   clock_t t1, t2, t3;
   float   timespent;

   t1=clock();

   int i,j,k;

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
   dt1=(double)dt;
   dt2=(double)(dt*dt);

   sf_warning("ns=%d dt=%f",ns,dt);
   sf_warning("npk=%d ",npk);
   sf_warning("eps=%f",eps);
   sf_warning("read velocity model parameters");

   /* setup I files */
   iRSF vp0, vs0("vs0"), epsi("epsi"), del("del"),gam("gam"),the("the"),phi("phi");

   /* Read/Write axes */
   int nxv,nyv,nzv;
   vp0.get("n1",nzv);
   vp0.get("n2",nxv);
   vp0.get("n3",nyv);

   float az, ax,ay;
   vp0.get("o1",az);
   vp0.get("o2",ax);
   vp0.get("o3",ay);

   float fx,fy,fz;
   fx=ax*1000.0;
   fy=ay*1000.0;
   fz=az*1000.0;

   float dx,dy,dz;
   vp0.get("d1",az);
   vp0.get("d2",ax);
   vp0.get("d3",ay);
   dz = az*1000.0;
   dx = ax*1000.0;
   dy = ay*1000.0;

   /* wave modeling space */
   int nx,ny,nz,nxz,nxyz;
   nx=nxv;
   ny=nyv;
   nz=nzv;
   nxz=nx*nz;
   nxyz=nx*ny*nz;

   sf_warning("nx=%d ny=%d nz=%d",nx,ny,nz);
   sf_warning("dx=%f dy=%f dz=%f",dx,dy,dz);

   sf_warning("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
   sf_warning("Warning: 2nd-order spectral need odd-based FFT");
   sf_warning("Warning: 2nd-order spectral need odd-based FFT");
   sf_warning("Warning: 2nd-order spectral need odd-based FFT");
   sf_warning("Warning: 2nd-order spectral need odd-based FFT");
   sf_warning("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");

   vp.resize(nxyz);
   vs.resize(nxyz);
   ep.resize(nxyz);
   de.resize(nxyz);
   ga.resize(nxyz);
   th.resize(nxyz);
   ph.resize(nxyz);
 
   c11 = sf_floatalloc(nxyz);
   c12 = sf_floatalloc(nxyz);
   c13 = sf_floatalloc(nxyz);
   c14 = sf_floatalloc(nxyz);
   c15 = sf_floatalloc(nxyz);
   c16 = sf_floatalloc(nxyz);
   c22 = sf_floatalloc(nxyz);
   c23 = sf_floatalloc(nxyz);
   c24 = sf_floatalloc(nxyz);
   c25 = sf_floatalloc(nxyz);
   c26 = sf_floatalloc(nxyz);
   c33 = sf_floatalloc(nxyz);
   c34 = sf_floatalloc(nxyz);
   c35 = sf_floatalloc(nxyz);
   c36 = sf_floatalloc(nxyz);
   c44 = sf_floatalloc(nxyz);
   c45 = sf_floatalloc(nxyz);
   c46 = sf_floatalloc(nxyz);
   c55 = sf_floatalloc(nxyz);
   c56 = sf_floatalloc(nxyz);
   c66 = sf_floatalloc(nxyz);
   
  /* vp0>>vp;
   vs0>>vs;
   epsi>>ep;
   del>>de;
   gam>>ga;
   the>>th;
   phi>>ph;

   for(i=0;i<nxyz;i++)
   {
	   th[i] *= SF_PI/180.0;
	   ph[i] *= SF_PI/180.0;
   }

   float *vp1,*vs1,*ep1,*de1,*ga1,*th1,*ph1;
   vp1 = sf_floatalloc(nxyz);
   vs1 = sf_floatalloc(nxyz);
   ep1 = sf_floatalloc(nxyz);
   de1 = sf_floatalloc(nxyz);
   ga1 = sf_floatalloc(nxyz);
   th1 = sf_floatalloc(nxyz);
   ph1 = sf_floatalloc(nxyz);

   for(int i=0;i<nxyz;i++)
   {
	   vp1[i] = vp[i];
	   vs1[i] = vs[i];
	   de1[i] = de[i];
	   ep1[i] = ep[i];
	   ga1[i] = ga[i];
	   th1[i] = th[i];
	   ph1[i] = ph[i];
   }
   Thomson2stiffness_3d(vp1,vs1,ep1,de1,ga1,th1,ph1,c11,c12,c13,c14,c15,c16,
					c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,
					c56,c66,nx, ny, nz);*/
   for(i=0;i<nxyz;i++)
   {
       c11[i] = 10e6;
       c12[i] = 3.5e6;
       c13[i] = 2.5e6;
       c14[i] = -5e6;
       c15[i] = 0.1e6;
       c16[i] = 0.3e6;
       c22[i] = 8e6;
       c23[i] = 1.5e6;
       c24[i] = 0.2e6;
       c25[i] = -0.1e6;
       c26[i] = -0.15e6;
       c33[i] = 6e6;
       c34[i] = 1e6;
       c35[i] = 0.4e6;
       c36[i] = 0.24e6;
       c44[i] = 5e6;
       c45[i] = 0.35e6;
       c46[i] = 0.525e6;
       c55[i] = 4e6;
       c56[i] = -1e6;
       c66[i] = 3e6;
   }

/*	free(vp1);
	free(vs1);
	free(ep1);
	free(de1);
	free(ga1);
	free(th1);
	free(ph1);*/


   /* Fourier spectra demension */
   int nkz,nkx,nky,nk;
   nkx=nx;
   nky=ny;
   nkz=nz;
   nk = nkx*nky*nkz;

   for(i=0;i<nk;i++)
   {
	   if(c33[i]<0||c44[i]<0||c66[i]<0)
	   {
		   sf_warning("stifiness matrix doesn't positive");
		   exit(0);
	   }
	   if((c11[i]-c66[i])*c33[i]-c13[i]*c13[i]<0)
	   {
		   sf_warning("stifiness matrix doesn't positive");
		   exit(0);
	   }
   }

   float dkz,dkx,dky,kz0,kx0,ky0;

   dkx=2*SF_PI/dx/nx;
   dky=2*SF_PI/dy/ny;
   dkz=2*SF_PI/dz/nz;

   kx0=-SF_PI/dx;
   ky0=-SF_PI/dy;
   kz0=-SF_PI/dz;

   sf_warning("dkx=%f dky=%f dkz=%f",dkx,dky,dkz);

   rkx.resize(nk);
   rky.resize(nk);
   rkz.resize(nk);
   rk2.resize(nk);


   double kx, kz,ky,rk, k2;
   int ix,iy,iz;
   i = 0;
   for(iy=0; iy < nky; iy++)
   {
     ky = ky0+iy*dky;

     for(ix=0; ix < nkx; ix++)
     {
       kx = kx0+ix*dkx;

         for (iz=0; iz < nkz; iz++)
         {
            kz = kz0+iz*dkz;

            k2 = ky*ky+kx*kx+kz*kz;
            rk = sqrt(k2);

            rky[i] = ky/rk;
            rkx[i] = kx/rk;
            rkz[i] = kz/rk;
            rk2[i] = k2;
            i++;
         }
      }
   }

   vector<int> md(nxyz), nd(nk);
   for (k=0; k < nxyz; k++)  md[k] = k;
   for (k=0; k < nk; k++)  nd[k] = k;

   vector<int> lid, rid;
   DblNumMat mid, mat;

   /********* low rank decomposition of operator Axx  **********/
   int   m2axx, n2axx;
   float *ldataaxx, *fmidaxx, *rdataaxx;

   iC( ddlowrank(nxyz,nk,sampleaxx,eps,npk,lid,rid,mid) );
   m2axx=mid.m();
   n2axx=mid.n();
   sf_warning("m2axx=%d n2axx=%d",m2axx, n2axx);

   fmidaxx  = sf_floatalloc(m2axx*n2axx);
   ldataaxx = sf_floatalloc(nxyz*m2axx);
   rdataaxx = sf_floatalloc(n2axx*nk);

   map2d1d(fmidaxx, mid, m2axx, n2axx);

   iC ( sampleaxx(md,lid,mat) );
   map2d1d(ldataaxx, mat, nxyz, m2axx);

   iC ( sampleaxx(rid,nd,mat) );
   map2d1d(rdataaxx, mat, n2axx, nk);

   /********* low rank decomposition of operator Ayy  **********/
   int   m2ayy, n2ayy;
   float *ldataayy, *fmidayy, *rdataayy;

   iC( ddlowrank(nxyz,nk,sampleayy,eps,npk,lid,rid,mid) );
   m2ayy=mid.m();
   n2ayy=mid.n();
   sf_warning("m2ayy=%d n2ayy=%d",m2ayy, n2ayy);

   fmidayy  = sf_floatalloc(m2ayy*n2ayy);
   ldataayy = sf_floatalloc(nxyz*m2ayy);
   rdataayy = sf_floatalloc(n2ayy*nk);

   map2d1d(fmidayy, mid, m2ayy, n2ayy);

   iC ( sampleayy(md,lid,mat) );
   map2d1d(ldataayy, mat, nxyz, m2ayy);

   iC ( sampleayy(rid,nd,mat) );
   map2d1d(rdataayy, mat, n2ayy, nk);

   /********* low rank decomposition of operator Azz  **********/
   int   m2azz, n2azz;
   float *ldataazz, *fmidazz, *rdataazz;

   iC( ddlowrank(nxyz,nk,sampleazz,eps,npk,lid,rid,mid) );
   m2azz=mid.m();
   n2azz=mid.n();
   sf_warning("m2azz=%d n2azz=%d",m2azz, n2azz);

   fmidazz  = sf_floatalloc(m2azz*n2azz);
   ldataazz = sf_floatalloc(nxyz*m2azz);
   rdataazz = sf_floatalloc(n2azz*nk);

   map2d1d(fmidazz, mid, m2azz, n2azz);

   iC ( sampleazz(md,lid,mat) );
   map2d1d(ldataazz, mat, nxyz, m2azz);

   iC ( sampleazz(rid,nd,mat) );
   map2d1d(rdataazz, mat, n2azz, nk);

   /********* low rank decomposition of operator Axy  **********/
   int   m2axy, n2axy;
   float *ldataaxy, *fmidaxy, *rdataaxy;

   iC( ddlowrank(nxyz,nk,sampleaxy,eps,npk,lid,rid,mid) );
   m2axy=mid.m();
   n2axy=mid.n();
   sf_warning("m2axy=%d n2axy=%d",m2axy, n2axy);

   fmidaxy  = sf_floatalloc(m2axy*n2axy);
   ldataaxy = sf_floatalloc(nxyz*m2axy);
   rdataaxy = sf_floatalloc(n2axy*nk);

   map2d1d(fmidaxy, mid, m2axy, n2axy);

   iC ( sampleaxy(md,lid,mat) );
   map2d1d(ldataaxy, mat, nxyz, m2axy);

   iC ( sampleaxy(rid,nd,mat) );
   map2d1d(rdataaxy, mat, n2axy, nk);

   /********* low rank decomposition of operator Axz  **********/
   int   m2axz, n2axz;
   float *ldataaxz, *fmidaxz, *rdataaxz;

   iC( ddlowrank(nxyz,nk,sampleaxz,eps,npk,lid,rid,mid) );
   m2axz=mid.m();
   n2axz=mid.n();
   sf_warning("m2axz=%d n2axz=%d",m2axz, n2axz);

   fmidaxz  = sf_floatalloc(m2axz*n2axz);
   ldataaxz = sf_floatalloc(nxyz*m2axz);
   rdataaxz = sf_floatalloc(n2axz*nk);

   map2d1d(fmidaxz, mid, m2axz, n2axz);

   iC ( sampleaxz(md,lid,mat) );
   map2d1d(ldataaxz, mat, nxyz, m2axz);

   iC ( sampleaxz(rid,nd,mat) );
   map2d1d(rdataaxz, mat, n2axz, nk);

   /********* low rank decomposition of operator Ayz  **********/
   int   m2ayz, n2ayz;
   float *ldataayz, *fmidayz, *rdataayz;

   iC( ddlowrank(nxyz,nk,sampleayz,eps,npk,lid,rid,mid) );
   m2ayz=mid.m();
   n2ayz=mid.n();
   sf_warning("m2ayz=%d n2ayz=%d",m2ayz, n2ayz);

   fmidayz  = sf_floatalloc(m2ayz*n2ayz);
   ldataayz = sf_floatalloc(nxyz*m2ayz);
   rdataayz = sf_floatalloc(n2ayz*nk);

   map2d1d(fmidayz, mid, m2ayz, n2ayz);

   iC ( sampleayz(md,lid,mat) );
   map2d1d(ldataayz, mat, nxyz, m2ayz);

   iC ( sampleayz(rid,nd,mat) );
   map2d1d(rdataayz, mat, n2ayz, nk);

   /****************End of Calculating Projection Deviation Operator****************/

   t2=clock();
   timespent=(float)(t2-t1)/CLOCKS_PER_SEC;
   sf_warning("CPU time for low-rank decomp: %f(second)",timespent);

   /****************begin to calculate wavefield****************/
   /****************begin to calculate wavefield****************/
   /*  wavelet parameter for source definition */
   float A, f0, t0;
   f0=30.0;
   t0=0.04;
   A=1.0;

   sf_warning("fx=%f fy=%f fz=%f ",fx,fy,fz);
   sf_warning("dx=%f dy=%f dz=%f ",dx,dy,dz);
   sf_warning("nx=%d ny=%d nz=%d ",nx,ny,nz);

   /* source definition */
   int ixs, izs,iys;
   ixs=nxv/2;
   izs=nzv/2;
   iys=nyv/2;
   sf_warning("ixs=%d iys=%d izs=%d ", ixs,iys,izs);

   /* setup I/O files */
   oRSF Elasticx("out");
   oRSF Elasticy("Elasticy");
   oRSF Elasticz("Elasticz");

   Elasticx.put("n1",nz);
   Elasticx.put("n2",nx);
   Elasticx.put("n3",ny);
   Elasticx.put("d1",dz/1000);
   Elasticx.put("d2",dx/1000);
   Elasticx.put("d3",dy/1000);
   Elasticx.put("o1",fz/1000);
   Elasticx.put("o2",fx/1000);
   Elasticx.put("o3",fy/1000);

   Elasticy.put("n1",nz);
   Elasticy.put("n2",nx);
   Elasticy.put("n3",ny);
   Elasticy.put("d1",dz/1000);
   Elasticy.put("d2",dx/1000);
   Elasticy.put("o1",fz/1000);
   Elasticy.put("o2",fx/1000);

   Elasticz.put("n1",nkz);
   Elasticz.put("n2",nkx);
   Elasticz.put("d1",dz/1000);
   Elasticz.put("d2",dx/1000);
   Elasticz.put("d3",dy/1000);
   Elasticz.put("o1",fz/1000);
   Elasticz.put("o2",fx/1000);
   Elasticz.put("o3",fy/1000);

   /********************* wavefield extrapolation *************************/
   float *ux1=sf_floatalloc(nxyz);
   float *ux2=sf_floatalloc(nxyz);
   float *ux3=sf_floatalloc(nxyz);
   float *uy1=sf_floatalloc(nxyz);
   float *uy2=sf_floatalloc(nxyz);
   float *uy3=sf_floatalloc(nxyz);
   float *uz1=sf_floatalloc(nxyz);
   float *uz2=sf_floatalloc(nxyz);
   float *uz3=sf_floatalloc(nxyz);

   float *pp=sf_floatalloc(nxyz);

   zero1float(ux1, nxyz);
   zero1float(ux2, nxyz);
   zero1float(ux3, nxyz);
   zero1float(uy1, nxyz);
   zero1float(uy2, nxyz);
   zero1float(uy3, nxyz);
   zero1float(uz1, nxyz);
   zero1float(uz2, nxyz);
   zero1float(uz3, nxyz);

   int *ijkx = sf_intalloc(nkx);
   int *ijkz = sf_intalloc(nkz);
   int *ijky = sf_intalloc(nky);

   ikxikyikz(ijkx, ijky, ijkz, nkx, nky, nkz);

   std::valarray<float> x(nxyz);

	int nbd = 20;
	float alpha = 0.005;
    float *decay = sf_floatalloc(nxyz);
    for (iy = 0; iy < ny; iy++) {
        for (ix = 0; ix < nx; ix++) {
            for (iz=0; iz < nz; iz++) {
                i = iz+nz *(ix+nx *iy);
                decay[i]=1.0;
                if(iz<nbd)
                    decay[i] *= exp(-pow(alpha*(nbd-iz)*dz,2));
                else if(iz>(nz-1-nbd))
                    decay[i] *= exp(-pow(alpha*(iz-nz+nbd+1)*dz,2));
                if(ix<nbd)
                    decay[i] *= exp(-pow(alpha*(nbd-ix)*dx,2));
                else if(ix>(nx-1-nbd))
                    decay[i] *= exp(-pow(alpha*(ix-nx+nbd+1)*dx,2));
                if(iy<nbd)
                    decay[i] *= exp(-pow(alpha*(nbd-iy)*dy,2));
                else if(iy>(ny-1-nbd))
                    decay[i] *= exp(-pow(alpha*(iy-ny+nbd+1)*dy,2));
            }
        }
    }

   int iii;
   for(int it=0;it<ns;it++)
   {
        float t=it*dt;

        if(it%10==0)
                sf_warning("Elastic: it= %d  t=%f(s)",it,t);
 
         // 3D exploding force source (e.g., Wu's PhD)
         for(k=-1;k<=1;k++)
         for(i=-1;i<=1;i++)
         for(j=-1;j<=1;j++)
         {
             if(fabs(k)+fabs(i)+fabs(j)==3)
             {
                 iii=(iys+k)*nxz+(ixs+i)*nz+(izs+j);
                 uy2[iii]+=k*Ricker(t, f0, t0, A);
                 ux2[iii]+=i*Ricker(t, f0, t0, A);
                 uz2[iii]+=j*Ricker(t, f0, t0, A);
             }
        }


        if(it%10==0) sf_warning("ux=%f uy=%f uz=%f ",ux2[iii],uy2[iii],uz2[iii]);

        /* extrapolation of Ux-componet */
        fwpvti3delowrank(ldataaxx,rdataaxx,fmidaxx,pp,ux2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2axx,n2axx);
        for(i=0;i<nxyz;i++) ux3[i] = pp[i];
        fwpvti3delowrank(ldataaxy,rdataaxy,fmidaxy,pp,uy2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2axy,n2axy);
        for(i=0;i<nxyz;i++) ux3[i] += pp[i];
        fwpvti3delowrank(ldataaxz,rdataaxz,fmidaxz,pp,uz2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2axz,n2axz);
        for(i=0;i<nxyz;i++)
			ux3[i] = decay[i]*(ux3[i] +  pp[i]) - pow(decay[i],2)*ux1[i];

        /* extrapolation of Uy-componet */
        fwpvti3delowrank(ldataayy,rdataayy,fmidayy,pp,uy2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2ayy,n2ayy);
        for(i=0;i<nxyz;i++) uy3[i] = pp[i];
        fwpvti3delowrank(ldataaxy,rdataaxy,fmidaxy,pp,ux2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2axy,n2axy);
        for(i=0;i<nxyz;i++) uy3[i] += pp[i];
        fwpvti3delowrank(ldataayz,rdataayz,fmidayz,pp,uz2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2ayz,n2ayz);
        for(i=0;i<nxyz;i++)
			uy3[i] = decay[i]*(uy3[i] + pp[i]) - pow(decay[i],2)*uy1[i];

        /* extrapolation of Uz-componet */
        fwpvti3delowrank(ldataazz,rdataazz,fmidazz,pp,uz2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2azz,n2azz);
        for(i=0;i<nxyz;i++) uz3[i] = pp[i];
        fwpvti3delowrank(ldataaxz,rdataaxz,fmidaxz,pp,ux2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2axz,n2axz);
        for(i=0;i<nxyz;i++) uz3[i] += pp[i];
        fwpvti3delowrank(ldataayz,rdataayz,fmidayz,pp,uy2,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2ayz,n2ayz);
        for(i=0;i<nxyz;i++)
			uz3[i] = decay[i]*(uz3[i] + pp[i]) - pow(decay[i],2)*uz1[i];

        /******* update the wavefield ********/
        for(i=0;i<nxyz;i++){
                ux1[i]=ux2[i];
                ux2[i]=ux3[i];
                uy1[i]=uy2[i];
                uy2[i]=uy3[i];
                uz1[i]=uz2[i];
                uz2[i]=uz3[i];
            }
        /******* output wavefields: components******/
        if(it==ns-1)
        {
              for(i=0;i<nxyz;i++) x[i]=ux3[i];
              Elasticx<<x;
              for(i=0;i<nxyz;i++) x[i]=uy3[i];
              Elasticy<<x;
              for(i=0;i<nxyz;i++) x[i]=uz3[i];
              Elasticz<<x;
        }
   } //* it loop */

   t3=clock();
   timespent=(float)(t3-t2)/CLOCKS_PER_SEC;
   sf_warning("CPU time for wavefield extrapolation.: %f(second)",timespent);

   timespent=(float)(t3-t1)/(ns*CLOCKS_PER_SEC);
   sf_warning("CPU time for every time extrapolation (including low-rank decom.): %f(second)",timespent);

   free(ldataaxx);
   free(fmidaxx);
   free(rdataaxx);

   free(ldataayy);
   free(fmidayy);
   free(rdataayy);

   free(ldataazz);
   free(fmidazz);
   free(rdataazz);

   free(ldataaxy);
   free(fmidaxy);
   free(rdataaxy);

   free(ldataaxz);
   free(fmidaxz);
   free(rdataaxz);

   free(ldataayz);
   free(fmidayz);
   free(rdataayz);

   free(ux1);
   free(ux2);
   free(ux3);
   free(uy1);
   free(uy2);
   free(uy3);
   free(uz1);
   free(uz2);
   free(uz3);

   free(pp);

   free(ijkx);
   free(ijkz);

   exit(0);
}

double A[3][3],Q[3][3],w[3]; //Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
int info;

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator 1 to extrapolate based on low-rank decomp. */
int sampleaxx(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;
	double lam1,lam2,lam3,sinclam1,sinclam2,sinclam3;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kx = rkx[cs[b]];
            double ky = rky[cs[b]];
            double kz = rkz[cs[b]];
            double k2 = rk2[cs[b]];
            if(kx==0.0&&ky==0.0&&kz==0.0)
            {
               resx(a,b) = 2.0;
               continue;
            }

           double kx2=kx*kx*k2;
           double ky2=ky*ky*k2;
           double kz2=kz*kz*k2;
		   double kxky=kx*ky*k2;
		   double kxkz=kx*kz*k2;
		   double kykz=ky*kz*k2;

           a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
           a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
           a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
           a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
           a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
           a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           A[0][0] = a11;
           A[0][1] = a12;
           A[0][2] = a13;
           A[1][0] = A[0][1];
           A[1][1] = a22;
           A[1][2] = a23;
           A[2][0] = A[0][2];
           A[2][1] = A[1][2];
           A[2][2] = a33;

           info = dsyevd3(A,Q,w);
           if(info == -1)
           {
               sf_warning("Error in Calculation the eigenvalues and normalized eigenvectors");
               exit(0);
           }
           u1 = Q[0][0];
           u2 = Q[1][0];
           u3 = Q[2][0];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1 = -u1;
               u2 = -u2;
               u3 = -u3;
           }

/*   A[0][0] = 1.0;
   A[0][1] = .0;
   A[0][2] = .0;
   A[1][0] = A[0][1];
   A[1][1] = 1.0;
   A[1][2] = .0;
   A[2][0] = A[0][2];
   A[2][1] = A[1][2];
   A[2][2] = 4;

   for(int ii=0;ii<3;ii++)
   {
       for(int jj=0;jj<3;jj++)
           sf_warning("%f\t",A[ii][jj]);
       sf_warning("\n");
   }

           info = dsyevd3(A,Q,w);
           if(info == -1)
           {
               sf_warning("Error in Calculation the eigenvalues and normalized eigenvectors");
               exit(0);
           }

sf_warning("%f\t%f\t%f\n",w[0],w[1],w[2]);
   for(int ii=0;ii<3;ii++)
   {
       for(int jj=0;jj<3;jj++)
           sf_warning("%f\t",Q[jj][ii]);
       sf_warning("\n");
   }

exit(0);*/

           lam1 = sqrt(w[0])*0.5*dt1;
           lam2 = sqrt(w[1])*0.5*dt1;
           lam3 = sqrt(w[2])*0.5*dt1;
           sinclam1 = sin(lam1)*sin(lam1)/lam1/lam1;
           sinclam2 = sin(lam2)*sin(lam2)/lam2/lam2;
           sinclam3 = sin(lam3)*sin(lam3)/lam3/lam3;

           a11 = Q[0][0]*Q[0][0]*w[0]*sinclam1 + Q[0][1]*Q[0][1]*w[1]*sinclam2 + Q[0][2]*Q[0][2]*w[2]*sinclam3;

            // wavefield extrapolator
            resx(a,b) = 2.0 - dt2*a11; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator 2 to extrapolate based on low-rank decomp. */
int sampleayy(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double a11, a12, a22, a33, a13, a23;
    double u1, u2, u3;
    double lam1,lam2,lam3,sinclam1,sinclam2,sinclam3;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kx = rkx[cs[b]];
            double ky = rky[cs[b]];
            double kz = rkz[cs[b]];
            double k2 = rk2[cs[b]];
            if(kx==0.0&&ky==0.0&&kz==0.0)
            {
               resx(a,b) = 2.0;
               continue;
            }

           double kx2=kx*kx*k2;
           double ky2=ky*ky*k2;
           double kz2=kz*kz*k2;
		   double kxky=kx*ky*k2;
		   double kxkz=kx*kz*k2;
		   double kykz=ky*kz*k2;

           a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
           a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
           a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
           a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
           a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
           a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           A[0][0] = a11;
           A[0][1] = a12;
           A[0][2] = a13;
           A[1][0] = A[0][1];
           A[1][1] = a22;
           A[1][2] = a23;
           A[2][0] = A[0][2];
           A[2][1] = A[1][2];
           A[2][2] = a33;

           info = dsyevd3(A,Q,w);
           if(info == -1)
           {
               sf_warning("Error in Calculation the eigenvalues and normalized eigenvectors");
               exit(0);
           }
           u1 = Q[0][0];
           u2 = Q[1][0];
           u3 = Q[2][0];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1 = -u1;
               u2 = -u2;
               u3 = -u3;
           } 

           lam1 = sqrt(w[0])*0.5*dt1;
           lam2 = sqrt(w[1])*0.5*dt1;
           lam3 = sqrt(w[2])*0.5*dt1;
           sinclam1 = sin(lam1)*sin(lam1)/lam1/lam1;
           sinclam2 = sin(lam2)*sin(lam2)/lam2/lam2;
           sinclam3 = sin(lam3)*sin(lam3)/lam3/lam3;

           // wavefield extrapolator
           a22 = Q[1][0]*Q[1][0]*w[0]*sinclam1 + Q[1][1]*Q[1][1]*w[1]*sinclam2 + Q[1][2]*Q[1][2]*w[2]*sinclam3;

           resx(a,b) = 2.0 - dt2*a22; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator 3 to extrapolate based on low-rank decomp. */
int sampleazz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double a11, a12, a22, a33, a13, a23;
    double u1, u2, u3;
    double lam1,lam2,lam3,sinclam1,sinclam2,sinclam3;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kx = rkx[cs[b]];
            double ky = rky[cs[b]];
            double kz = rkz[cs[b]];
            double k2 = rk2[cs[b]];
            if(kx==0.0&&ky==0.0&&kz==0.0)
            {
               resx(a,b) = 2.0;
               continue;
            }

           double kx2=kx*kx*k2;
           double ky2=ky*ky*k2;
           double kz2=kz*kz*k2;
		   double kxky=kx*ky*k2;
		   double kxkz=kx*kz*k2;
		   double kykz=ky*kz*k2;

           a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
           a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
           a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
           a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
           a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
           a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           A[0][0] = a11;
           A[0][1] = a12;
           A[0][2] = a13;
           A[1][0] = A[0][1];
           A[1][1] = a22;
           A[1][2] = a23;
           A[2][0] = A[0][2];
           A[2][1] = A[1][2];
           A[2][2] = a33;

           info = dsyevd3(A,Q,w);
           if(info == -1)
           {
               sf_warning("Error in Calculation the eigenvalues and normalized eigenvectors");
               exit(0);
           }
           u1 = Q[0][0];
           u2 = Q[1][0];
           u3 = Q[2][0];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1 = -u1;
               u2 = -u2; 
               u3 = -u3; 
           }
           lam1 = sqrt(w[0])*0.5*dt1;
           lam2 = sqrt(w[1])*0.5*dt1;
           lam3 = sqrt(w[2])*0.5*dt1;
           sinclam1 = sin(lam1)*sin(lam1)/lam1/lam1;
           sinclam2 = sin(lam2)*sin(lam2)/lam2/lam2;
           sinclam3 = sin(lam3)*sin(lam3)/lam3/lam3;

           a33 = Q[2][0]*Q[2][0]*w[0]*sinclam1 + Q[2][1]*Q[2][1]*w[1]*sinclam2 + Q[2][2]*Q[2][2]*w[2]*sinclam3;

           // wavefield extrapolator
           resx(a,b) = 2.0 - dt2*a33;

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator 4 to extrapolate based on low-rank decomp. */
int sampleaxy(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double a11, a12, a22, a33, a13, a23;
    double u1, u2, u3;
    double lam1,lam2,lam3,sinclam1,sinclam2,sinclam3;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kx = rkx[cs[b]];
            double ky = rky[cs[b]];
            double kz = rkz[cs[b]];
            double k2 = rk2[cs[b]];
            if(kx==0.0&&ky==0.0&&kz==0.0)
            {
               resx(a,b) = 0.0;
               continue;
            }

           double kx2=kx*kx*k2;
           double ky2=ky*ky*k2;
           double kz2=kz*kz*k2;
		   double kxky=kx*ky*k2;
		   double kxkz=kx*kz*k2;
		   double kykz=ky*kz*k2;

           a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
           a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
           a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
           a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
           a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
           a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           A[0][0] = a11;
           A[0][1] = a12;
           A[0][2] = a13;
           A[1][0] = A[0][1];
           A[1][1] = a22;
           A[1][2] = a23;
           A[2][0] = A[0][2];
           A[2][1] = A[1][2];
           A[2][2] = a33;

           info = dsyevd3(A,Q,w);
           if(info == -1)
           {
               sf_warning("Error in Calculation the eigenvalues and normalized eigenvectors");
               exit(0);
           }
           u1 = Q[0][0];
           u2 = Q[1][0];
           u3 = Q[2][0];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1 = -u1;
               u2 = -u2; 
               u3 = -u3; 
           }
           lam1 = sqrt(w[0])*0.5*dt1;
           lam2 = sqrt(w[1])*0.5*dt1;
           lam3 = sqrt(w[2])*0.5*dt1;
           sinclam1 = sin(lam1)*sin(lam1)/lam1/lam1;
           sinclam2 = sin(lam2)*sin(lam2)/lam2/lam2;
           sinclam3 = sin(lam3)*sin(lam3)/lam3/lam3;

           a12 = Q[0][0]*Q[1][0]*w[0]*sinclam1 + Q[0][1]*Q[1][1]*w[1]*sinclam2 + Q[0][2]*Q[1][2]*w[2]*sinclam3;

            // wavefield extrapolator
		   resx(a,b) = -dt2*a12;

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator 5 to extrapolate based on low-rank decomp. */
int sampleaxz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double a11, a12, a22, a33, a13, a23;
    double u1, u2, u3;
    double lam1,lam2,lam3,sinclam1,sinclam2,sinclam3;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kx = rkx[cs[b]];
            double ky = rky[cs[b]];
            double kz = rkz[cs[b]];
            double k2 = rk2[cs[b]];
            if(kx==0.0&&ky==0.0&&kz==0.0)
            {
               resx(a,b) = 0.0;
               continue;
            }

           double kx2=kx*kx*k2;
           double ky2=ky*ky*k2;
           double kz2=kz*kz*k2;
		   double kxky=kx*ky*k2;
		   double kxkz=kx*kz*k2;
		   double kykz=ky*kz*k2;

           a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
           a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
           a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
           a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
           a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
           a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           A[0][0] = a11;
           A[0][1] = a12;
           A[0][2] = a13;
           A[1][0] = A[0][1];
           A[1][1] = a22;
           A[1][2] = a23;
           A[2][0] = A[0][2];
           A[2][1] = A[1][2];
           A[2][2] = a33;

           info = dsyevd3(A,Q,w);
           if(info == -1)
           {
               sf_warning("Error in Calculation the eigenvalues and normalized eigenvectors");
               exit(0);
           }
           u1 = Q[0][0];
           u2 = Q[1][0];
           u3 = Q[2][0];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1 = -u1;
               u2 = -u2;
               u3 = -u3;
           }

           lam1 = sqrt(w[0])*0.5*dt1;
           lam2 = sqrt(w[1])*0.5*dt1;
           lam3 = sqrt(w[2])*0.5*dt1;
           sinclam1 = sin(lam1)*sin(lam1)/lam1/lam1;
           sinclam2 = sin(lam2)*sin(lam2)/lam2/lam2;
           sinclam3 = sin(lam3)*sin(lam3)/lam3/lam3;

           a13 = Q[0][0]*Q[2][0]*w[0]*sinclam1 + Q[0][1]*Q[2][1]*w[1]*sinclam2 + Q[0][2]*Q[2][2]*w[2]*sinclam3;

            // wavefield extrapolator
            resx(a,b) = -dt2*a13;

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
/* operator 6 to extrapolate based on low-rank decomp. */
int sampleayz(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

    double a11, a12, a22, a33, a13, a23;
    double u1, u2, u3;
    double lam1,lam2,lam3,sinclam1,sinclam2,sinclam3;

    for(int a=0; a<nr; a++) 
    {
        int i=rs[a];

        for(int b=0; b<nc; b++)
        {
            double kx = rkx[cs[b]];
            double ky = rky[cs[b]];
            double kz = rkz[cs[b]];
            double k2 = rk2[cs[b]];
            if(kx==0.0&&ky==0.0&&kz==0.0)
            {
               resx(a,b) = 0.0;
               continue;
            }

           double kx2=kx*kx*k2;
           double ky2=ky*ky*k2;
           double kz2=kz*kz*k2;
		   double kxky=kx*ky*k2;
		   double kxkz=kx*kz*k2;
		   double kykz=ky*kz*k2;

           a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
           a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
           a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
           a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
           a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
           a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           A[0][0] = a11;
           A[0][1] = a12;
           A[0][2] = a13;
           A[1][0] = A[0][1];
           A[1][1] = a22;
           A[1][2] = a23;
           A[2][0] = A[0][2];
           A[2][1] = A[1][2];
           A[2][2] = a33;

           info = dsyevd3(A,Q,w);
           if(info == -1)
           {
               sf_warning("Error in Calculation the eigenvalues and normalized eigenvectors");
               exit(0);
           }
           u1 = Q[0][0];
           u2 = Q[1][0];
           u3 = Q[2][0];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1 = -u1;
               u2 = -u2; 
               u3 = -u3; 
           }

           lam1 = sqrt(w[0])*0.5*dt1;
           lam2 = sqrt(w[1])*0.5*dt1;
           lam3 = sqrt(w[2])*0.5*dt1;
           sinclam1 = sin(lam1)*sin(lam1)/lam1/lam1;
           sinclam2 = sin(lam2)*sin(lam2)/lam2/lam2;
           sinclam3 = sin(lam3)*sin(lam3)/lam3/lam3;

           a23 = Q[1][0]*Q[2][0]*w[0]*sinclam1 + Q[1][1]*Q[2][1]*w[1]*sinclam2 + Q[1][2]*Q[2][2]*w[2]*sinclam3;

           // wavefield extrapolator
           resx(a,b) = -dt2*a23;

         }// b loop
    }// a loop

    return 0;
}

static void map2d1d(float *d, DblNumMat mat, int m, int n)
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
