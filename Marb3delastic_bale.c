/* 3D three-components general anisotropic  lastic wavefield extrapolation 
   using  rotated staggered pseudo-spectral method with velocity-stress wave 
   equation in heterogeneous media

   Copyright (C) 2015 Tongji University, Shanghai, China 
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

#include <rsf.h>
#include <assert.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>
#include <alloc.h>
#include <omp.h>
#include <stdio.h>
#include "puthead.h"
#include "derivative.h"

/* prepared head files by myself */
#include "_cjb.h"
#include "ricker.h"
#include "kykxkztaper.h"

/*****************************************************************************************/
void staggerPS1order3Ddx_xy(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1);
void staggerPS1order3Ddx_xz(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1);
void staggerPS1order3Ddx_yz(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1);
void staggerPS1order3Ddx_xyz(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1);

int main(int argc, char* argv[])
{
   fftwf_init_threads();
   omp_set_num_threads(30);
   sf_init(argc,argv);

   clock_t t1, t2;
   float   timespent;

   t1=clock();

   int i,k;
   
   int nt,flag;
   float dt;
   if (!sf_getint("nt",&nt)) nt = 500;
   if (!sf_getint("flag",&flag)) flag = 1;
   if (!sf_getfloat("dt",&dt)) dt = 0.001;

   sf_warning("nt=%d dt=%f",nt,dt);


   /* setup I/O files */
	sf_file Fo1,Fo2,Fo3;

	Fo1 = sf_output("Elasticvx");  //particle velocities x-component 
	Fo2 = sf_output("Elasticvy");  //particle velocities y-component 
	Fo3 = sf_output("Elasticvz");  //particle velocities z-component 

   /* wave modeling space */
   int nx,ny,nz,nxyz;
   float dx,dy,dz;
   if (!sf_getint("nx",&nx)) nx = 200;
   if (!sf_getint("ny",&ny)) ny = 200;
   if (!sf_getint("nz",&nz)) nz = 200;
   if (!sf_getfloat("dx",&dx)) dx = 10.0;
   if (!sf_getfloat("dy",&dy)) dy = 10.0;
   if (!sf_getfloat("dz",&dz)) dz = 10.0;
   sf_warning("nx=%d ny=%d nz=%d",nx,ny,nz);
   sf_warning("dx=%f dy=%f dz=%f",dx,dy,dz);
   nxyz = nx*ny*nz;

   puthead3x(Fo1,nz,nx,ny,dz/1000.0,dx/1000.0,dy/1000.,0,0,0);
   puthead3x(Fo2,nz,nx,ny,dz/1000.0,dx/1000.0,dy/1000.,0,0,0);
   puthead3x(Fo3,nz,nx,ny,dz/1000.0,dx/1000.0,dy/1000.,0,0,0);

   float *vx1,*vx2,*vz1,*vz2,*vy1,*vy2;
   vx1 = alloc1float(nxyz);
   vx2 = alloc1float(nxyz);
   vy1 = alloc1float(nxyz);
   vy2 = alloc1float(nxyz);
   vz1 = alloc1float(nxyz);
   vz2 = alloc1float(nxyz);
   zero1float(vx1,nxyz);
   zero1float(vx2,nxyz);
   zero1float(vy1,nxyz);
   zero1float(vy2,nxyz);
   zero1float(vz1,nxyz);
   zero1float(vz2,nxyz);

   float *Txx1,*Txx2,*Tyy1,*Tyy2,*Tzz1,*Tzz2;
   float *Txz1,*Txz2,*Tyz1,*Tyz2,*Txy1,*Txy2;

   Txx1 = alloc1float(nxyz);
   Txx2 = alloc1float(nxyz);
   Tyy1 = alloc1float(nxyz);
   Tyy2 = alloc1float(nxyz);
   Tzz1 = alloc1float(nxyz);
   Tzz2 = alloc1float(nxyz);
   Txy1 = alloc1float(nxyz);
   Txy2 = alloc1float(nxyz);
   Txz1 = alloc1float(nxyz);
   Txz2 = alloc1float(nxyz);
   Tyz1 = alloc1float(nxyz);
   Tyz2 = alloc1float(nxyz);

   zero1float(Txx1,nxyz);
   zero1float(Txx2,nxyz);
   zero1float(Tyy1,nxyz);
   zero1float(Tyy2,nxyz);
   zero1float(Tzz1,nxyz);
   zero1float(Tzz2,nxyz);
   zero1float(Txy1,nxyz);
   zero1float(Txy2,nxyz);
   zero1float(Txz1,nxyz);
   zero1float(Txz2,nxyz);
   zero1float(Tyz1,nxyz);
   zero1float(Tyz2,nxyz);

   float *c11,*c12,*c13,*c14,*c15,*c16,*c22,*c23,*c24,*c25,*c26,
		 *c33,*c34,*c35,*c36,*c44,*c45,*c46,*c55,*c56,*c66,*rho;
   c11 = alloc1float(nxyz);
   c12 = alloc1float(nxyz);
   c13 = alloc1float(nxyz);
   c14 = alloc1float(nxyz);
   c15 = alloc1float(nxyz);
   c16 = alloc1float(nxyz);
   c22 = alloc1float(nxyz);
   c23 = alloc1float(nxyz);
   c24 = alloc1float(nxyz);
   c25 = alloc1float(nxyz);
   c26 = alloc1float(nxyz);
   c33 = alloc1float(nxyz);
   c34 = alloc1float(nxyz);
   c35 = alloc1float(nxyz);
   c36 = alloc1float(nxyz);
   c44 = alloc1float(nxyz);
   c45 = alloc1float(nxyz);
   c46 = alloc1float(nxyz);
   c55 = alloc1float(nxyz);
   c56 = alloc1float(nxyz);
   c66 = alloc1float(nxyz);
   rho = alloc1float(nxyz);

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
	   rho[i] = 1.0;
   }

   int ix,iy,iz;
   i = 0;
/*   for(iy=0; iy < ny; iy++) 
	   for(ix=0; ix < nx; ix++)
		   for (iz=0; iz < nz; iz++)
		   {
			   if(iy>20&&iy<40&&ix>20&&ix<40&&iz>20&&iz<40)
			   {
				   c11[i] = 1000*10e6;
    	           c12[i] = 1000*3.5e6;
		           c13[i] = 1000*2.5e6;
		           c14[i] = 0;//1000*-5e6;
				   c15[i] = 0;//1000*0.1e6;
				   c16[i] = 0;//1000*0.3e6;
				   c22[i] = 1000*8e6;
				   c23[i] = 1000*1.5e6;
				   c24[i] = 0;//1000*0.2e6;
				   c25[i] = 0;//1000*-0.1e6;
				   c26[i] = 0;//1000*-0.15e6;
				   c33[i] = 1000*6e6;
				   c34[i] = 0;//1000*1e6;
				   c35[i] = 0;//1000*0.4e6;
				   c36[i] = 1000*0.24e6;
				   c44[i] = 1000*5e6;
				   c45[i] = 1000*0.35e6;
				   c46[i] = 0;//1000*0.525e6;
				   c55[i] = 1000*4e6;
				   c56[i] = 0;//1000*-1e6;
				   c66[i] = 1000*3e6;
				   rho[i] = 2000.0;
			   }
			   else
			   {
				   c11[i] = 2.25e10;
    	           c12[i] = 1.25e10;
		           c13[i] = 1.25e10;
		           c14[i] = 0;
				   c15[i] = 0;
				   c16[i] = 0;
				   c22[i] = 2.25e10;
				   c23[i] = 1.25e10;
				   c24[i] = 0;
				   c25[i] = 0;
				   c26[i] = 0;
				   c33[i] = 2.25e10;
				   c34[i] = 0;
				   c35[i] = 0;
				   c36[i] = 0;
				   c44[i] = 1e10;
				   c45[i] = 0;
				   c46[i] = 0;
				   c55[i] = 1e10;
				   c56[i] = 0;
				   c66[i] = 1e10;
				   rho[i] = 2500.0;
			   }
			   i++;
		   }*/

   float dkz,dkx,dky;
   float *kx = alloc1float(nxyz);
   float *ky = alloc1float(nxyz);
   float *kz = alloc1float(nxyz);

   dkx=2*SF_PI/dx/nx;
   dky=2*SF_PI/dy/ny;
   dkz=2*SF_PI/dz/nz;

   sf_warning("dkx=%f dky=%f dkz=%f",dkx,dky,dkz);

   i = 0;
   for(iy=0; iy < ny; iy++) 
	   for(ix=0; ix < nx; ix++)
		   for (iz=0; iz < nz; iz++)
		   {
			   if(ny%2==0)
			   {
			    if(iy<=ny/2)
					ky[i] = iy*dky;
				else
					ky[i] = (iy-ny)*dky;
			   }
			   else
			   {
			    if(iy<=ny/2)
					ky[i] = iy*dky;
				else
					ky[i] = (iy-ny)*dky;
			   }
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

   float *d1,*d2,*d3,*d4,*d5,*d6,*d7,*d8,*d9; //derivation variable
   d1 = alloc1float(nxyz);
   d2 = alloc1float(nxyz);
   d3 = alloc1float(nxyz);
   d4 = alloc1float(nxyz);
   d5 = alloc1float(nxyz);
   d6 = alloc1float(nxyz);
   d7 = alloc1float(nxyz);
   d8 = alloc1float(nxyz);
   d9 = alloc1float(nxyz);

   float *d11,*d22,*d33,*d44,*d55,*d66,*d77,*d88,*d99; //derivation variable
   d11 = alloc1float(nxyz);
   d22 = alloc1float(nxyz);
   d33 = alloc1float(nxyz);
   d44 = alloc1float(nxyz);
   d55 = alloc1float(nxyz);
   d66 = alloc1float(nxyz);
   d77 = alloc1float(nxyz);
   d88 = alloc1float(nxyz);
   d99 = alloc1float(nxyz);

   float *dd1,*dd2,*dd3,*dd4,*dd5,*dd6,*dd7,*dd8,*dd9; 
   float *dd10, *dd11,*dd12,*dd13,*dd14,*dd15; 
   dd1 = alloc1float(nxyz);
   dd2 = alloc1float(nxyz);
   dd3 = alloc1float(nxyz);
   dd4 = alloc1float(nxyz);
   dd5 = alloc1float(nxyz);
   dd6 = alloc1float(nxyz);
   dd7 = alloc1float(nxyz);
   dd8 = alloc1float(nxyz);
   dd9 = alloc1float(nxyz);
   dd10 = alloc1float(nxyz);
   dd11 = alloc1float(nxyz);
   dd12 = alloc1float(nxyz);
   dd13 = alloc1float(nxyz);
   dd14 = alloc1float(nxyz);
   dd15 = alloc1float(nxyz);

   for(k=0;k<nt;k++)
   {
	   if(flag==0)
	   {
		   RstaggerPS1order3Ddx(vx2,d1,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,1); //dvx/dx
		   RstaggerPS1order3Ddx(vx2,d2,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,1); //dvx/dy
		   RstaggerPS1order3Ddx(vx2,d3,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,1); //dvx/dz
		   RstaggerPS1order3Ddx(vy2,d4,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,1); //dvy/dx
		   RstaggerPS1order3Ddx(vy2,d5,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,1); //dvy/dy
		   RstaggerPS1order3Ddx(vy2,d6,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,1); //dvy/dz
		   RstaggerPS1order3Ddx(vz2,d7,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,1); //dvz/dx
		   RstaggerPS1order3Ddx(vz2,d8,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,1); //dvz/dy
		   RstaggerPS1order3Ddx(vz2,d9,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,1);//dvz/dz
	   }
	   else 
	   {
           staggerPS1order3Ddx(vx2,d1,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,1); //dvx/dx
           staggerPS1order3Ddx(vx2,d2,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,0); //dvx/dy
           staggerPS1order3Ddx(vx2,d3,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,0); //dvx/dz
           staggerPS1order3Ddx(vy2,d4,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,0); //dvy/dx
           staggerPS1order3Ddx(vy2,d5,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,1); //dvy/dy
           staggerPS1order3Ddx(vy2,d6,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,0); //dvy/dz
           staggerPS1order3Ddx(vz2,d7,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,0); //dvz/dx
           staggerPS1order3Ddx(vz2,d8,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,0); //dvz/dy
           staggerPS1order3Ddx(vz2,d9,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,1);//dvz/dz
	   }
           staggerPS1order3Ddx_xy(vx2,d22,nx,ny,nz,dx,dy,dz,kx,ky,kz,1); //dvx/dy
           staggerPS1order3Ddx_xz(vx2,d33,nx,ny,nz,dx,dy,dz,kx,ky,kz,1); //dvx/dz
           staggerPS1order3Ddx_xy(vy2,d44,nx,ny,nz,dx,dy,dz,kx,ky,kz,0); //dvy/dx
           staggerPS1order3Ddx_yz(vy2,d66,nx,ny,nz,dx,dy,dz,kx,ky,kz,1); //dvy/dz
           staggerPS1order3Ddx_xz(vz2,d77,nx,ny,nz,dx,dy,dz,kx,ky,kz,0); //dvz/dx
           staggerPS1order3Ddx_yz(vz2,d88,nx,ny,nz,dx,dy,dz,kx,ky,kz,0); //dvz/dy

	   for(i=0;i<nxyz;i++)
	   {
		   dd1[i] = c14[i]*d1[i];
		   dd2[i] = c24[i]*d5[i];
		   dd3[i] = c34[i]*d9[i];
		   dd4[i] = c45[i]*(d33[i]+d77[i]);
		   dd5[i] = c46[i]*(d22[i]+d44[i]);
		   dd6[i] = c15[i]*d1[i];
		   dd7[i] = c25[i]*d5[i];
		   dd8[i] = c35[i]*d9[i];
		   dd9[i] = c45[i]*(d66[i]+d88[i]);
		   dd10[i] = c56[i]*(d22[i]+d44[i]);
		   dd11[i] = c16[i]*d1[i];
		   dd12[i] = c26[i]*d5[i];
		   dd13[i] = c36[i]*d9[i];
		   dd14[i] = c46[i]*(d66[i]+d88[i]);
		   dd15[i] = c56[i]*(d33[i]+d77[i]);
	   }

           staggerPS1order3Ddx_xyz(dd1,dd1,nx,ny,nz,dx,dy,dz,kx,ky,kz,0); 
           staggerPS1order3Ddx_xyz(dd2,dd2,nx,ny,nz,dx,dy,dz,kx,ky,kz,0); 
           staggerPS1order3Ddx_xyz(dd3,dd3,nx,ny,nz,dx,dy,dz,kx,ky,kz,0);
           staggerPS1order3Ddx_xyz(dd4,dd4,nx,ny,nz,dx,dy,dz,kx,ky,kz,0);
           staggerPS1order3Ddx_xyz(dd5,dd5,nx,ny,nz,dx,dy,dz,kx,ky,kz,0);
           staggerPS1order3Ddx_xyz(dd6,dd6,nx,ny,nz,dx,dy,dz,kx,ky,kz,1);
           staggerPS1order3Ddx_xyz(dd7,dd7,nx,ny,nz,dx,dy,dz,kx,ky,kz,1);
           staggerPS1order3Ddx_xyz(dd8,dd8,nx,ny,nz,dx,dy,dz,kx,ky,kz,1);
           staggerPS1order3Ddx_xyz(dd9,dd9,nx,ny,nz,dx,dy,dz,kx,ky,kz,1);
           staggerPS1order3Ddx_xyz(dd10,dd10,nx,ny,nz,dx,dy,dz,kx,ky,kz,1);
           staggerPS1order3Ddx_xyz(dd11,dd11,nx,ny,nz,dx,dy,dz,kx,ky,kz,2);
           staggerPS1order3Ddx_xyz(dd12,dd12,nx,ny,nz,dx,dy,dz,kx,ky,kz,2);
           staggerPS1order3Ddx_xyz(dd13,dd13,nx,ny,nz,dx,dy,dz,kx,ky,kz,2);
           staggerPS1order3Ddx_xyz(dd14,dd14,nx,ny,nz,dx,dy,dz,kx,ky,kz,2);
           staggerPS1order3Ddx_xyz(dd15,dd15,nx,ny,nz,dx,dy,dz,kx,ky,kz,2);

	   for(i=0;i<nxyz;i++)
	   {
		   Txx2[i] = Txx1[i] + dt*(c11[i]*d1[i]+c12[i]*d5[i]+c13[i]*d9[i]+c14[i]*(d66[i]+d88[i])
				     +c15[i]*(d33[i]+d77[i])+c16[i]*(d22[i]+d44[i]));
		   Tyy2[i] = Tyy1[i] + dt*(c12[i]*d1[i]+c22[i]*d5[i]+c23[i]*d9[i]+c24[i]*(d66[i]+d88[i])
				     +c25[i]*(d33[i]+d77[i])+c26[i]*(d22[i]+d44[i]));
		   Tzz2[i] = Tzz1[i] + dt*(c13[i]*d1[i]+c23[i]*d5[i]+c33[i]*d9[i]+c34[i]*(d66[i]+d88[i])
				     +c35[i]*(d33[i]+d77[i])+c36[i]*(d22[i]+d44[i]));

		   Tyz2[i] = Tyz1[i] + dt*(dd1[i]+dd2[i]+dd3[i]+dd4[i]+dd5[i]+c44[i]*(d6[i]+d8[i]));

		   Txz2[i] = Txz1[i] + dt*(dd6[i]+dd7[i]+dd8[i]+dd9[i]+dd10[i]+c55[i]*(d3[i]+d7[i]));

		   Txy2[i] = Txy1[i] + dt*(dd11[i]+dd12[i]+dd13[i]+dd14[i]+dd15[i]+c66[i]*(d2[i]+d4[i]));
	   }

	   Txx2[nz*nx*(ny/2)+nx/2*nz+nz/2] += Ricker(k*dt,20,0.04,10);
	   Tyy2[nz*nx*(ny/2)+nx/2*nz+nz/2] += Ricker(k*dt,20,0.04,10);
	   Tzz2[nz*nx*(ny/2)+nx/2*nz+nz/2] += Ricker(k*dt,20,0.04,10); //source term

	   if(k%10==0)
	   sf_warning("k=%d",k);

	   if(flag==0)
	   {
		   RstaggerPS1order3Ddx(Txx2,d1,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,0); //dTxx/dx
		   RstaggerPS1order3Ddx(Txy2,d2,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,0);//dTxy/dy
		   RstaggerPS1order3Ddx(Txz2,d3,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,0); //dTxz/dz
		   RstaggerPS1order3Ddx(Txy2,d4,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,0); //dTxy/dx
		   RstaggerPS1order3Ddx(Tyy2,d5,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,0); //dTyy/dy
		   RstaggerPS1order3Ddx(Tyz2,d6,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,0); //dTyz/dz
		   RstaggerPS1order3Ddx(Txz2,d7,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,0); //dTxz/dx
		   RstaggerPS1order3Ddx(Tyz2,d8,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,0); //dTyz/dy
		   RstaggerPS1order3Ddx(Tzz2,d9,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,0); //dTzz/dz
	   }
	   else
	   {
           staggerPS1order3Ddx(Txx2,d1,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,0); //dTxx/dx
           staggerPS1order3Ddx(Txy2,d2,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,1);//dTxy/dy
           staggerPS1order3Ddx(Txz2,d3,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,1); //dTxz/dz
           staggerPS1order3Ddx(Txy2,d4,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,1); //dTxy/dx
           staggerPS1order3Ddx(Tyy2,d5,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,0); //dTyy/dy
           staggerPS1order3Ddx(Tyz2,d6,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,1); //dTyz/dz
           staggerPS1order3Ddx(Txz2,d7,nx,ny,nz,dx,dy,dz,kx,ky,kz,0,1); //dTxz/dx
           staggerPS1order3Ddx(Tyz2,d8,nx,ny,nz,dx,dy,dz,kx,ky,kz,1,1); //dTyz/dy
           staggerPS1order3Ddx(Tzz2,d9,nx,ny,nz,dx,dy,dz,kx,ky,kz,2,0); //dTzz/dz

	   }
	   for(i=0;i<nxyz;i++)
	   {
		   vx2[i] = vx1[i] + dt*(d1[i]+d2[i]+d3[i])/rho[i];
		   vy2[i] = vy1[i] + dt*(d4[i]+d5[i]+d6[i])/rho[i];
		   vz2[i] = vz1[i] + dt*(d7[i]+d8[i]+d9[i])/rho[i];
	   }

	   for(i=0;i<nxyz;i++)
	   {
		   vx1[i] = vx2[i];
		   vy1[i] = vy2[i];
		   vz1[i] = vz2[i];
		   Txx1[i] = Txx2[i];
		   Tyy1[i] = Tyy2[i];
		   Tzz1[i] = Tzz2[i];
		   Txy1[i] = Txy2[i];
		   Txz1[i] = Txz2[i];
		   Tyz1[i] = Tyz2[i];
	   }
   } // end of nt loop

   for(i=0;i<nxyz;i++)
   {
	   sf_floatwrite(&vx2[i],1,Fo1);
	   sf_floatwrite(&vy2[i],1,Fo2);
	   sf_floatwrite(&vz2[i],1,Fo3);
   }

   t2 = clock();
   timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
   sf_warning("costime %fs",timespent);

   free(kx);
   free(ky);
   free(kz);

   free(Txx1);
   free(Txx2);
   free(Tyy1);
   free(Tyy2);
   free(Tzz1);
   free(Tzz2);
   free(Txy1);
   free(Txy2);
   free(Txz1);
   free(Txz2);
   free(Tyz1);
   free(Tyz2);

   free(vx1);
   free(vx2);
   free(vy1);
   free(vy2);
   free(vz1);
   free(vz2);

   return 0;
}

void staggerPS1order3Ddx_xy(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1)
/*< RstaggerPS1order3Ddx caculate 1order derivation of p using rotated-staggered PS method 
             if flag1==0,caculate dp/dx flag==1 dp/dy,else caculate dp/dz 
               if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,ix,iy,iz,nxyz;
       nxyz = nx*ny*nz;

       xin=sf_complexalloc(nxyz);
       xout=sf_complexalloc(nxyz);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       i = 0;
       if(flag1 == 0) //dp/dx
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = I*kx[i]*cexpf(I*(-ky[i]*dy)/2.0)*xout[i];
                               i++;
                       }
       }
       else // dp/dy
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = I*ky[i]*cexpf(I*(-kx[i]*dx)/2.0)*xout[i];
                               i++;
                       }
       }

       fftwf_execute(xpi);

       for(i=0;i<nxyz;i++)
              dp[i] = creal(xout[i])/nxyz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif
}

void staggerPS1order3Ddx_xz(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1)
/*< RstaggerPS1order3Ddx caculate 1order derivation of p using rotated-staggered PS method 
             if flag1==0,caculate dp/dx flag==1 dp/dy,else caculate dp/dz 
               if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,ix,iy,iz,nxyz;
       nxyz = nx*ny*nz;

       xin=sf_complexalloc(nxyz);
       xout=sf_complexalloc(nxyz);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       i = 0;
       if(flag1 == 0) //dp/dx
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = I*kx[i]*cexpf(I*(-kz[i]*dz)/2.0)*xout[i];
                               i++;
                       }
       }
       else // dp/dz
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = I*kz[i]*cexpf(I*(-kx[i]*dx)/2.0)*xout[i];
                               i++;
                       }
       }

       fftwf_execute(xpi);

       for(i=0;i<nxyz;i++)
              dp[i] = creal(xout[i])/nxyz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif
}

void staggerPS1order3Ddx_yz(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1)
/*< RstaggerPS1order3Ddx caculate 1order derivation of p using rotated-staggered PS method 
             if flag1==0,caculate dp/dx flag==1 dp/dy,else caculate dp/dz 
               if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,ix,iy,iz,nxyz;
       nxyz = nx*ny*nz;

       xin=sf_complexalloc(nxyz);
       xout=sf_complexalloc(nxyz);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       i = 0;
       if(flag1 == 0) //dp/dy
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = I*ky[i]*cexpf(I*(-kz[i]*dz)/2.0)*xout[i];
                               i++;
                       }
       }
       else // dp/dz
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = I*kz[i]*cexpf(I*(-ky[i]*dy)/2.0)*xout[i];
                               i++;
                       }
       }

       fftwf_execute(xpi);

       for(i=0;i<nxyz;i++)
              dp[i] = creal(xout[i])/nxyz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif
}

void staggerPS1order3Ddx_xyz(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1)
/*< RstaggerPS1order3Ddx caculate 1order derivation of p using rotated-staggered PS method 
             if flag1==0,caculate dp/dx flag==1 dp/dy,else caculate dp/dz 
               if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,ix,iy,iz,nxyz;
       nxyz = nx*ny*nz;

       xin=sf_complexalloc(nxyz);
       xout=sf_complexalloc(nxyz);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       i = 0;
       if(flag1 == 0) //Syz
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = cexpf(I*(ky[i]*dy+kz[i]*dz)/2.0)*xout[i];
                               i++;
                       }
       }
	   else if(flag1 == 1) //Sxz
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = cexpf(I*(kx[i]*dx+kz[i]*dz)/2.0)*xout[i];
                               i++;
                       }
       }
       else // Sxy
       {
               for(iy=0;iy<ny;iy++)
                   for(ix=0;ix<nx;ix++)
                       for(iz=0;iz<nz;iz++)
                       {
                               xin[i] = cexpf(I*(kx[i]*dx+ky[i]*dy)/2.0)*xout[i];
                               i++;
                       }
       }

       fftwf_execute(xpi);

       for(i=0;i<nxyz;i++)
              dp[i] = creal(xout[i])/nxyz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif
}

