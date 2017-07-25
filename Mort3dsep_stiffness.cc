/* 3-D elastic wavefield decoupling using low-rank approximate PS solution 
   in arbitrary anisotropic media. The API use ORT media Thomsen parameter
   with tilt angle and azimuth to convert into 21 stifnesses.

   Copyright (C) 2016 Tongji University, Shanghai, China 
   Authors: Peng Zou
     
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
}

static float *c11,*c12,*c13,*c14,*c15,*c16,
			      *c22,*c23,*c24,*c25,*c26,
				       *c33,*c34,*c35,*c36,
					        *c44,*c45,*c46,
						         *c55,*c56,
							          *c66;

static std::valarray<double> rkx,rky,rkz,rk2;
static std::valarray<float> vp, vs, ep1, de1, ga1, ep2, de2, ga2, de3, th, ph;
static std::valarray<float> vx, vy, vz;

/* dual-domain operators based on low-rank decomp. */
int sampleopx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopx3(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopy1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopy2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopy3(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleopz3(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

int sampleosx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosx3(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosy1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosy2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosy3(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx);
int sampleosz3(vector<int>& rs, vector<int>& cs, DblNumMat& resx);

static void map2d1d(double *d, DblNumMat mat, int m, int n);
/* definition for LAPACK SVD ROUTINEs */
char    jobz='V';  // for SVD 
char    uplo='U';  // for SVD 
int     M=3;       // for SVD 
int     LDA=M;     // for SVD 
int     LWORK=8*M; // for SVD 
int     INFO;      // for SVD 
double  Chr[9], ww[9], work[24];  // Lapack SVD array 

/*****************************************************************************************/
int main(int argc, char* argv[])
{
   sf_init(argc,argv);
   fftwf_init_threads();
   omp_set_num_threads(10);
   
   clock_t t1, t2, t3;
   float   timespent;

   t1=clock();

   int i,k;

   iRSF par(0);
   int seed;
   par.get("seed",seed,time(NULL)); // seed for random number generator
   srand48(seed);

   float eps;
   par.get("eps",eps,1.e-4); // tolerance
       
   int npk;
   par.get("npk",npk,20); // maximum rank

   sf_warning("npk=%d ",npk);
   sf_warning("eps=%f",eps);
   sf_warning("read velocity model parameters");

   /* setup I files */
   iRSF vp0, vs0("vs0"), epsi1("epsi1"), del1("del1"), gam1("gam1"), the("the"), phi("phi");
   iRSF epsi2("epsi2"), del2("del2"), gam2("gam2"), del3("del3");	
   iRSF Elasticvx("Elasticvx"), Elasticvy("Elasticvy"), Elasticvz("Elasticvz");

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

   vp.resize(nxyz);
   vs.resize(nxyz);
   ep1.resize(nxyz);
   de1.resize(nxyz);
   ga1.resize(nxyz);
   ep2.resize(nxyz);
   de2.resize(nxyz);
   ga2.resize(nxyz);
   de3.resize(nxyz);
   th.resize(nxyz);
   ph.resize(nxyz);
   vx.resize(nxyz);
   vy.resize(nxyz);
   vz.resize(nxyz);

 
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
   
   vp0>>vp;
   vs0>>vs;
   epsi1>>ep1;
   del1>>de1;
   gam1>>ga1;
   epsi2>>ep2;
   del2>>de2;
   gam2>>ga2;
   del3>>de3;
   the>>th;
   phi>>ph;
   Elasticvx>>vx;
   Elasticvy>>vy;
   Elasticvz>>vz;

   for(i=0;i<nxyz;i++)
   {
	   th[i] *= SF_PI/180.0;
	   ph[i] *= SF_PI/180.0;
   }

   float *vp_1, *vs_1, *ep1_1, *de1_1, *ga1_1, *ep2_1;
   float *de2_1, *ga2_1, *de3_1, *th_1, *ph_1;
   vp_1 = sf_floatalloc(nxyz);
   vs_1 = sf_floatalloc(nxyz);
   ep1_1 = sf_floatalloc(nxyz);
   de1_1 = sf_floatalloc(nxyz);
   ga1_1 = sf_floatalloc(nxyz);
   ep2_1 = sf_floatalloc(nxyz);
   de2_1 = sf_floatalloc(nxyz);
   ga2_1 = sf_floatalloc(nxyz);
   de3_1 = sf_floatalloc(nxyz);
   th_1 = sf_floatalloc(nxyz);
   ph_1 = sf_floatalloc(nxyz);

   for(int i=0;i<nxyz;i++)
   {
	   vp_1[i] = vp[i];
	   vs_1[i] = vs[i];
	   de1_1[i] = de1[i];
	   ep1_1[i] = ep1[i];
	   ga1_1[i] = ga1[i];
	   de2_1[i] = de2[i];
	   ep2_1[i] = ep2[i];
	   ga2_1[i] = ga2[i];
	   de3_1[i] = de3[i];
	   th_1[i] = th[i];
	   ph_1[i] = ph[i];
   }
   Thomson2stiffness_ort(vp_1,vs_1,ep1_1,ep2_1,de1_1,de2_1,de3_1,ga1_1,ga2_1,th_1,ph_1,c11,c12,c13,c14,c15,c16,
						c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66,nx, ny, nz);
	free(vp_1);
	free(vs_1);
	free(ep1_1);
	free(de1_1);
	free(ga1_1);
	free(ep2_1);
	free(de2_1);
	free(ga2_1);
	free(de3_1);
	free(th_1);
	free(ph_1);

   sf_warning("c11=%g",c11[1]);
   sf_warning("c12=%g",c12[1]);
   sf_warning("c13=%g",c13[1]);
   sf_warning("c22=%g",c22[1]);
   sf_warning("c23=%g",c23[1]);
   sf_warning("c33=%g",c33[1]);
   sf_warning("c44=%g",c44[1]);
   sf_warning("c55=%g",c55[1]);
   sf_warning("c66=%g",c66[1]);
   sf_warning("c16=%g",c16[1]);
   sf_warning("c26=%g",c26[1]);
   sf_warning("c36=%g",c36[1]);
   sf_warning("c45=%g",c45[1]);

   /* Fourier spectra demension */
   int nkz,nkx,nky,nk;
   nkx=nx;
   nky=ny;
   nkz=nz;
   nk = nkx*nky*nkz;

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

   /********* qP-wave low rank decomposition of operator BxAx + BxyAxy + BxzAxz applying to ux for updating ux **********/
   int   m2opx1, n2opx1;
   double *ldataopx1, *fmidopx1, *rdataopx1;

   int jump = 50;
   vector<int> ms, ns, js;
   ms.resize(3); ms[0] = nz; ms[1] = nx; ms[2] = ny;
   ns.resize(3); ns[0] = nz; ns[1] = nx; ns[2] = ny;
   js.resize(3); js[0] = jump;  js[1] = jump;  js[2] = jump;

   //iC( ddlowrank(nxyz,nk,sampleopx1,eps,npk,lid,rid,mid) );
   iC( ddlowrank(ms,ns,js,sampleopx1,eps,npk,lid,rid,mid) );
   m2opx1=mid.m();
   n2opx1=mid.n();
   sf_warning("m2opx1=%d n2opx1=%d",m2opx1, n2opx1);

   fmidopx1  = (double*)malloc(sizeof(double)*m2opx1*n2opx1);
   ldataopx1 = (double*)malloc(sizeof(double)*nxyz*m2opx1);
   rdataopx1 = (double*)malloc(sizeof(double)*n2opx1*nk);

   map2d1d(fmidopx1, mid, m2opx1, n2opx1);

   iC ( sampleopx1(md,lid,mat) );
   map2d1d(ldataopx1, mat, nxyz, m2opx1);

   iC ( sampleopx1(rid,nd,mat) );
   map2d1d(rdataopx1, mat, n2opx1, nk);

   /********* qP-wave low rank decomposition of operator BxAxy + BxyAy + BxzAyz applying to uy for updating ux **********/
   int   m2opx2, n2opx2;
   double *ldataopx2, *fmidopx2, *rdataopx2;

   //iC( ddlowrank(nxyz,nk,sampleopx2,eps,npk,lid,rid,mid) );
   iC( ddlowrank(ms,ns,js,sampleopx2,eps,npk,lid,rid,mid) );
   m2opx2=mid.m();
   n2opx2=mid.n();
   sf_warning("m2opx2=%d n2opx2=%d",m2opx2, n2opx2);

   fmidopx2  = (double*)malloc(sizeof(double)*m2opx2*n2opx2);
   ldataopx2 = (double*)malloc(sizeof(double)*nxyz*m2opx2);
   rdataopx2 = (double*)malloc(sizeof(double)*n2opx2*nk);

   map2d1d(fmidopx2, mid, m2opx2, n2opx2);

   iC ( sampleopx2(md,lid,mat) );
   map2d1d(ldataopx2, mat, nxyz, m2opx2);

   iC ( sampleopx2(rid,nd,mat) );
   map2d1d(rdataopx2, mat, n2opx2, nk);

   /********* qP-wave low rank decomposition of operator BxAxz + BxyAyz + BxzAz applying to uz for updating ux **********/
   int   m2opx3, n2opx3;
   double *ldataopx3, *fmidopx3, *rdataopx3;

   //iC( ddlowrank(nxyz,nk,sampleopx3,eps,npk,lid,rid,mid) );
   iC( ddlowrank(ms,ns,js,sampleopx3,eps,npk,lid,rid,mid) );
   m2opx3=mid.m();
   n2opx3=mid.n();
   sf_warning("m2opx3=%d n2opx3=%d",m2opx3, n2opx3);

   fmidopx3  = (double*)malloc(sizeof(double)*m2opx3*n2opx3);
   ldataopx3 = (double*)malloc(sizeof(double)*nxyz*m2opx3);
   rdataopx3 = (double*)malloc(sizeof(double)*n2opx3*nk);

   map2d1d(fmidopx3, mid, m2opx3, n2opx3);

   iC ( sampleopx3(md,lid,mat) );
   map2d1d(ldataopx3, mat, nxyz, m2opx3);

   iC ( sampleopx3(rid,nd,mat) );
   map2d1d(rdataopx3, mat, n2opx3, nk);

   t2=clock();
   timespent=(float)(t2-t1)/CLOCKS_PER_SEC;
   sf_warning("CPU time for low-rank decomp: %f(second)",timespent);

   /* setup I/O files */
   oRSF ElasticPx("ElasticPx");
   oRSF ElasticPy("ElasticPy");
   oRSF ElasticPz("ElasticPz");
   oRSF ElasticSx("ElasticSx");
   oRSF ElasticSy("ElasticSy");
   oRSF ElasticSz("ElasticSz");

   ElasticPx.put("n1",nz);
   ElasticPx.put("n2",nx);
   ElasticPx.put("n3",ny);
   ElasticPx.put("d1",dz/1000);
   ElasticPx.put("d2",dx/1000);
   ElasticPx.put("d3",dy/1000);
   ElasticPx.put("o1",fz/1000);
   ElasticPx.put("o2",fx/1000);
   ElasticPx.put("o3",fy/1000);
   
   ElasticPy.put("n1",nz);
   ElasticPy.put("n2",nx);
   ElasticPy.put("n3",ny);
   ElasticPy.put("d1",dz/1000);
   ElasticPy.put("d2",dx/1000);
   ElasticPy.put("d3",dy/1000);
   ElasticPy.put("o1",fz/1000);
   ElasticPy.put("o2",fx/1000);
   ElasticPy.put("o3",fy/1000);

   ElasticPz.put("n1",nz);
   ElasticPz.put("n2",nx);
   ElasticPz.put("n3",ny);
   ElasticPz.put("d1",dz/1000);
   ElasticPz.put("d2",dx/1000);
   ElasticPz.put("d3",dy/1000);
   ElasticPz.put("o1",fz/1000);
   ElasticPz.put("o2",fx/1000);
   ElasticPz.put("o3",fy/1000);

   ElasticSx.put("n1",nz);
   ElasticSx.put("n2",nx);
   ElasticSx.put("n3",ny);
   ElasticSx.put("d1",dz/1000);
   ElasticSx.put("d2",dx/1000);
   ElasticSx.put("d3",dy/1000);
   ElasticSx.put("o1",fz/1000);
   ElasticSx.put("o2",fx/1000);
   ElasticSx.put("o3",fy/1000);

   ElasticSy.put("n1",nz);
   ElasticSy.put("n2",nx);
   ElasticSy.put("n3",ny);
   ElasticSy.put("d1",dz/1000);
   ElasticSy.put("d2",dx/1000);
   ElasticSy.put("d3",dy/1000);
   ElasticSy.put("o1",fz/1000);
   ElasticSy.put("o2",fx/1000);
   ElasticSy.put("o3",fy/1000);

   ElasticSz.put("n1",nz);
   ElasticSz.put("n2",nx);
   ElasticSz.put("n3",ny);
   ElasticSz.put("d1",dz/1000);
   ElasticSz.put("d2",dx/1000);
   ElasticSz.put("d3",dy/1000);
   ElasticSz.put("o1",fz/1000);
   ElasticSz.put("o2",fx/1000);
   ElasticSz.put("o3",fy/1000);

   /********************* wavefield extrapolation *************************/
   double *px1=(double*)malloc(sizeof(double)*nxyz);
   double *px2=(double*)malloc(sizeof(double)*nxyz);
   double *px3=(double*)malloc(sizeof(double)*nxyz);
   double *py1=(double*)malloc(sizeof(double)*nxyz);
   double *py2=(double*)malloc(sizeof(double)*nxyz);
   double *py3=(double*)malloc(sizeof(double)*nxyz);
   double *pz1=(double*)malloc(sizeof(double)*nxyz);
   double *pz2=(double*)malloc(sizeof(double)*nxyz);
   double *pz3=(double*)malloc(sizeof(double)*nxyz);
   
   double *sx1=(double*)malloc(sizeof(double)*nxyz);
   double *sx2=(double*)malloc(sizeof(double)*nxyz);
   double *sx3=(double*)malloc(sizeof(double)*nxyz);
   double *sy1=(double*)malloc(sizeof(double)*nxyz);
   double *sy2=(double*)malloc(sizeof(double)*nxyz);
   double *sy3=(double*)malloc(sizeof(double)*nxyz);
   double *sz1=(double*)malloc(sizeof(double)*nxyz);
   double *sz2=(double*)malloc(sizeof(double)*nxyz);
   double *sz3=(double*)malloc(sizeof(double)*nxyz);
   
   double *ux=(double*)malloc(sizeof(double)*nxyz);
   double *uy=(double*)malloc(sizeof(double)*nxyz);
   double *uz=(double*)malloc(sizeof(double)*nxyz);

   double *pp=(double*)malloc(sizeof(double)*nxyz);
   double *ppp=(double*)malloc(sizeof(double)*nxyz);
   
   zero1double(px1, nxyz);
   zero1double(px2, nxyz);
   zero1double(px3, nxyz);
   zero1double(py1, nxyz);
   zero1double(py2, nxyz);
   zero1double(py3, nxyz);
   zero1double(pz1, nxyz);
   zero1double(pz2, nxyz);
   zero1double(pz3, nxyz);

   zero1double(sx1, nxyz);
   zero1double(sx2, nxyz);
   zero1double(sx3, nxyz);
   zero1double(sy1, nxyz);
   zero1double(sy2, nxyz);
   zero1double(sy3, nxyz);
   zero1double(sz1, nxyz);
   zero1double(sz2, nxyz);
   zero1double(sz3, nxyz);

   zero1double(ux, nxyz);
   zero1double(uy, nxyz);
   zero1double(uz, nxyz);

   int *ijkx = sf_intalloc(nkx);
   int *ijkz = sf_intalloc(nkz);
   int *ijky = sf_intalloc(nky);

   ikxikyikz(ijkx, ijky, ijkz, nkx, nky, nkz);

   std::valarray<float> x(nxyz);

   for(i=0;i<nxyz;i++)
   {
	   ux[i] = vx[i];
	   uy[i] = vy[i];
	   uz[i] = vz[i];
   }

       /* extrapolation of Upx-componet */
        fwpvti3delowrank_double(ldataopx1,rdataopx1,fmidopx1,pp,ux,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2opx1,n2opx1);
        for(i=0;i<nxyz;i++) px3[i] = pp[i];
        fwpvti3delowrank_double(ldataopx2,rdataopx2,fmidopx2,pp,uy,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2opx2,n2opx2);
        for(i=0;i<nxyz;i++) px3[i] += pp[i];
        fwpvti3delowrank_double(ldataopx3,rdataopx3,fmidopx3,pp,uz,ijkx,ijky,ijkz,nx,ny,nz,nxyz,nk,m2opx3,n2opx3);
		for(i=0;i<nxyz;i++) px3[i] += pp[i];
		

        for(i=0;i<nxyz;i++) x[i]=px3[i];
        ElasticPx<<x;
        for(i=0;i<nxyz;i++) x[i]=py3[i];
        ElasticPy<<x;
        for(i=0;i<nxyz;i++) x[i]=pz3[i];
        ElasticPz<<x;
        for(i=0;i<nxyz;i++) x[i]=sx3[i];
        ElasticSx<<x;
        for(i=0;i<nxyz;i++) x[i]=sy3[i];
        ElasticSy<<x;
        for(i=0;i<nxyz;i++) x[i]=sz3[i];
        ElasticSz<<x;

   t3=clock();
   timespent=(float)(t3-t2)/CLOCKS_PER_SEC;
   sf_warning("CPU time for wavefield decoupling.: %f(second)",timespent);


   free(ldataopx1);
   free(fmidopx1);
   free(rdataopx1);

   free(ldataopx2);
   free(fmidopx2);
   free(rdataopx2);

   free(ldataopx3);
   free(fmidopx3);
   free(rdataopx3);


   free(ijkx);
   free(ijky);
   free(ijkz);

   exit(0);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xx*w_xx+d_xy*w_xy+d_xz*w_xz applying to ux **********//
int sampleopx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u1*u1; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xx*w_xy+d_xy*w_yy+d_xz*w_yz applying to uy **********/
int sampleopx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u1*u2; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xx*w_xz+d_xy*w_yz+dxz*w_zz applying to uz **********/
int sampleopx3(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u1*u3; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xy*w_xx+d_yy*w_xy+d_yz*wxz applying to ux **********//
int sampleopy1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u1*u2; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xy*w_xy+d_yy*w_yy+d_yz*w_yz applying to uy **********/
int sampleopy2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u2*u2; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xy*w_xz+d_yy*w_yz+d_yz*w_zz applying to uz **********/
int sampleopy3(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u2*u3; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xz*w_xx+d_yz*w_xy+d_yz*w_xz  applying to ux **********//
int sampleopz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u3*u1; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xz*w_xy+d_yz*w_yy+d_zz*w_yz applying to uy **********/
int sampleopz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u3*u2; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qP-wave low rank decomposition of operator d_xz*w_xz+d_yz*w_yz+d_zz*w_zz applying to uz **********/
int sampleopz3(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u3*u3; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xx*w_xx+d_xy*w_xy+d_xz*w_xz applying to ux **********//
int sampleosx1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = (u2*u2+u3*u3); 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xx*w_xy+d_xy*w_yy+d_xz*w_yz applying to uy **********/
int sampleosx2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = -u1*u2; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xx*w_xz+d_xy*w_yz+dxz*w_zz applying to uz **********/
int sampleosx3(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = -u1*u3; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xy*w_xx+d_yy*w_xy+d_yz*wxz applying to ux **********//
int sampleosy1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = -u1*u2; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xy*w_xy+d_yy*w_yy+d_yz*w_yz applying to uy **********/
int sampleosy2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u1*u1+u3*u3; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xy*w_xz+d_yy*w_yz+d_yz*w_zz applying to uz **********/
int sampleosy3(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = -u2*u3; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xz*w_xx+d_yz*w_xy+d_yz*w_xz  applying to ux **********//
int sampleosz1(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = -u3*u1; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xz*w_xy+d_yz*w_yy+d_zz*w_yz applying to uy **********/
int sampleosz2(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = -u3*u2; 

         }// b loop
    }// a loop

    return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//********* qS-wave low rank decomposition of operator d_xz*w_xz+d_yz*w_yz+d_zz*w_zz applying to uz **********/
int sampleosz3(vector<int>& rs, vector<int>& cs, DblNumMat& resx)
{
    int nr = rs.size();
    int nc = cs.size();

    resx.resize(nr,nc);

    setvalue(resx,0.0);

	double a11, a12, a22, a33, a13, a23;
	double u1, u2, u3;

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

           double kx2 = kx*kx*k2;
           double ky2 = ky*ky*k2;
           double kz2 = kz*kz*k2;
		   double kxky = kx*ky*k2;
		   double kxkz = kx*kz*k2;
		   double kykz = ky*kz*k2;

		   a11 = c11[i]*kx2 + c66[i]*ky2 + c55[i]*kz2 + 2.0*(c56[i]*kykz + c15[i]*kxkz + c16[i]*kxky);
		   a22 = c66[i]*kx2 + c22[i]*ky2 + c44[i]*kz2 + 2.0*(c24[i]*kykz + c46[i]*kxkz + c26[i]*kxky);
		   a33 = c55[i]*kx2 + c44[i]*ky2 + c33[i]*kz2 + 2.0*(c34[i]*kykz + c35[i]*kxkz + c45[i]*kxky);
		   a12 = c16[i]*kx2 + c26[i]*ky2 + c45[i]*kz2 + (c46[i]+c25[i])*kykz + (c14[i]+c56[i])*kxkz + (c12[i]+c66[i])*kxky;
		   a13 = c15[i]*kx2 + c46[i]*ky2 + c35[i]*kz2 + (c45[i]+c36[i])*kykz + (c13[i]+c55[i])*kxkz + (c14[i]+c56[i])*kxky;
		   a23 = c56[i]*kx2 + c24[i]*ky2 + c34[i]*kz2 + (c44[i]+c23[i])*kykz + (c36[i]+c45[i])*kxkz + (c25[i]+c46[i])*kxky;

           Chr[0] = a11;
           Chr[4] = a22;
           Chr[8] = a33;
           Chr[1] = Chr[3] = a12;
           Chr[2] = Chr[6] = a13;
           Chr[5] = Chr[7] = a23;
           // LAPACK's ssyev routine (slow but accurate) 
           dsyev_(&jobz, &uplo, &M, Chr, &LDA, ww, work, &LWORK, &INFO);

           u1=Chr[6];
           u2=Chr[7];
           u3=Chr[8];

           if(u1*kx + u2*ky+ u3*kz < 0.) {
               u1=-Chr[6];
               u2=-Chr[7];
               u3=-Chr[8];
           }

            resx(a,b) = u1*u1+u2*u2; 

         }// b loop
    }// a loop

    return 0;
}

static void map2d1d(double *d, DblNumMat mat, int m, int n)
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
