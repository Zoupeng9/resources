/* 2-D two-components vertical transverse isotropic  visco-elastic wavefield
   extrapolation using  pseudo-spectral method with velocity-stress wave 
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
#include <stdio.h>
#include <omp.h>
#include "puthead.h"
#include "derivative.h"

/* prepared head files by myself */
#include "_cjb.h"
#include "ricker.h"
#include "vti2tti.h"
#include "kykxkztaper.h"

/*****************************************************************************************/

int main(int argc, char* argv[])
{
   sf_init(argc,argv);
   fftwf_init_threads();
   omp_set_num_threads(30);

   clock_t t1, t2;
   float   timespent;

   t1=clock();

   int i,iz,ix,k;
   
   int nt;
   float dt;
   if (!sf_getint("nt",&nt)) nt = 500;
   if (!sf_getfloat("dt",&dt)) dt = 0.001;

   sf_warning("nt=%d dt=%f",nt,dt);

   int nxpad,nzpad;
   if (!sf_getint("nxpad",&nxpad)) nxpad = 20;
   if (!sf_getint("nzpad",&nzpad)) nzpad = 20;
   sf_warning("nxpad=%d  nzpad=%d",nxpad,nzpad);

   sf_warning("read anisotropic elastic parameters"); 

   /* setup I/O files */
	sf_file Fvp0,Fvs0,Fepsi,Fdel,Fthe;
	sf_file Fo1,Fo2,Fo3,Fo4;

	Fvp0 = sf_input("vp0"); //Vp0
	Fvs0 = sf_input("vs0"); //Vs0
	Fepsi = sf_input("epsi"); //epsi
	Fdel = sf_input("del"); //del
	Fthe = sf_input("the"); //the 

	Fo1 = sf_output("Elasticvx");  //particle velocities x-component 
	Fo2 = sf_output("Elasticvz");  //particle velocities z-component 
	Fo3 = sf_output("recordvx"); // surface record of x-component
	Fo4 = sf_output("recordvz"); // surface record of z-component

   /* Read/Write axes */
   sf_axis ax,az;
   int nxv, nzv;
   float dx,dz,fx,fz;
   az = sf_iaxa(Fvp0,1); nzv = sf_n(az); dz = sf_d(az)*1000.0;
   ax = sf_iaxa(Fvp0,2); nxv = sf_n(ax); dx = sf_d(ax)*1000.0;
   fz = sf_o(az)*1000.0;
   fx = sf_o(ax)*1000.0;

   sf_warning("nx=%d nz=%d",nxv,nzv);
   sf_warning("dx=%f dz=%f",dx,dz);

   /* wave modeling space */
   int nx, nz, nxz;
   nx=nxv;
   nz=nzv;
   nxz=nx*nz;

   puthead2(Fo1,nz,nx,dz/1000.0,fx/1000.0,dx/1000.,fx/1000.0);
   puthead2(Fo2,nz,nx,dz/1000.0,fx/1000.0,dx/1000.,fx/1000.0);
   puthead2(Fo3,nt,nx-2*nxpad,dz/1000.0,fx/1000.0,dx/1000.,fx/1000.0);
   puthead2(Fo4,nt,nx-2*nxpad,dz/1000.0,fx/1000.0,dx/1000.,fx/1000.0);

   float *vx1,*vx2,*vz1,*vz2;
   vx1 = alloc1float(nxz);
   vx2 = alloc1float(nxz);
   vz1 = alloc1float(nxz);
   vz2 = alloc1float(nxz);
   zero1float(vx1,nxz);
   zero1float(vx2,nxz);
   zero1float(vz1,nxz);
   zero1float(vz2,nxz);

   float *Txx1,*Txx2,*Txz1,*Txz2,*Tzz1,*Tzz2;
   Txx1 = alloc1float(nxz);
   Txx2 = alloc1float(nxz);
   Tzz1 = alloc1float(nxz);
   Tzz2 = alloc1float(nxz);
   Txz1 = alloc1float(nxz);
   Txz2 = alloc1float(nxz);

   zero1float(Txx1,nxz);
   zero1float(Txx2,nxz);
   zero1float(Txz1,nxz);
   zero1float(Txz2,nxz);
   zero1float(Tzz1,nxz);
   zero1float(Tzz2,nxz);

   float *recordvx, *recordvz;
   recordvx = alloc1float(nt*(nx-2*nxpad));
   recordvz = alloc1float(nt*(nx-2*nxpad));

   float *d1,*d2,*d3,*d4; //derivation variable
   d1 = alloc1float(nxz);
   d2 = alloc1float(nxz);
   d3 = alloc1float(nxz);
   d4 = alloc1float(nxz);

   float *vp0,*vs0,*epsi,*del,*the; //input model parameter
   vp0 = alloc1float(nxz);
   vs0 = alloc1float(nxz);
   epsi = alloc1float(nxz);
   del = alloc1float(nxz);
   the = alloc1float(nxz);
	   
   /*read model parameter*/
   sf_floatread(vp0,nxz,Fvp0);
   sf_floatread(vs0,nxz,Fvs0);
   sf_floatread(epsi,nxz,Fepsi);
   sf_floatread(del,nxz,Fdel);
   sf_floatread(the,nxz,Fthe);


   float *c11,*c13,*c15,*c33,*c35,*c55;
   c11 = alloc1float(nx*nz);
   c13 = alloc1float(nx*nz);
   c15 = alloc1float(nx*nz);
   c33 = alloc1float(nx*nz);
   c35 = alloc1float(nx*nz);
   c55 = alloc1float(nx*nz);

   for(i=0;i<nxz;i++)
	   the[i] *= SF_PI/180;

   Thomson2stiffness_2d(vp0,vs0,epsi,del,the,c11,c13,c15,c33,c35,c55,nx,nz);

   float dkz,dkx;
   float *kx = alloc1float(nxz);
   float *kz = alloc1float(nxz);

   dkx=2*SF_PI/dx/nx;
   dkz=2*SF_PI/dz/nz;

   sf_warning("dkx=%f dkz=%f",dkx,dkz);
   i = 0;
   for(ix=0; ix < nx; ix++)
	   for (iz=0; iz < nz; iz++)
	   {
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
			   
   float *dx1,*dz1,*ax1,*az1,*bx,*bz,*alphax,*alphaz;
   dx1 = alloc1float(nxz);
   ax1 = alloc1float(nxz);
   bx = alloc1float(nxz);
   alphax = alloc1float(nxz);
   dz1 = alloc1float(nxz);
   az1 = alloc1float(nxz);
   bz = alloc1float(nxz);
   alphaz = alloc1float(nxz);
   zero1float(dx1,nxz);
   zero1float(ax1,nxz);
   zero1float(bx,nxz);
   zero1float(alphax,nxz);
   zero1float(dz1,nxz);
   zero1float(az1,nxz);
   zero1float(bz,nxz);
   zero1float(alphaz,nxz);

   float kax,kaz,alphax_max,alphaz_max,d0x,d0z;
   kax = 1.;
   kaz = 1.;
   alphax_max = SF_PI*20.;
   alphaz_max = SF_PI*20.;
   d0x = -3.0*4500*log(0.001)/2./nxpad/dx;
   d0z = -3.0*4500*log(0.001)/2./nzpad/dz;

   i = 0;
   for(ix=0;ix<nx;ix++)
       for(iz=0;iz<nz;iz++)
       {
           if(ix<nxpad)
           {
               alphax[i] = alphax_max*ix/nxpad;
               dx1[i] = d0x*pow((nxpad-ix)*1.0/nxpad,2);
               bx[i] = exp(-(dx1[i]/kax+alphax[i])*dt);
               ax1[i] = dx1[i]/kax/(dx1[i]+kax*alphax[i]+0.001)*(bx[i]-1.);
           }
           if(ix>=nx-nxpad)
           {
               alphax[i] = alphax_max*(nx-ix)/nxpad;
               dx1[i] = d0x*pow((ix+nxpad-nx)*1.0/nxpad,2);
               bx[i] = exp(-(dx1[i]/kax+alphax[i])*dt);
               ax1[i] = dx1[i]/kax/(dx1[i]+kax*alphax[i]+0.001)*(bx[i]-1.);
           }
           if(iz<nzpad)
           {
               alphaz[i] = alphaz_max*iz/nzpad;
               dz1[i] = d0z*pow((nzpad-iz)*1.0/nzpad,2);
               bz[i] = exp(-(dz1[i]/kaz+alphaz[i])*dt);
               az1[i] = dz1[i]/kaz/(dz1[i]+kaz*alphaz[i]+0.001)*(bz[i]-1.);
           }
           if(iz>=nz-nzpad)
           {
               alphaz[i] = alphaz_max*(nz-iz)/nzpad;
               dz1[i] = d0z*pow((iz+nzpad-nz)*1.0/nzpad,2);
               bz[i] = exp(-(dz1[i]/kaz+alphaz[i])*dt);
               az1[i] = dz1[i]/kaz/(dz1[i]+kaz*alphaz[i]+0.001)*(bz[i]-1.);
           }
           bx[i] = exp(-(dx1[i]/kax+alphax[i])*dt);
           bz[i] = exp(-(dz1[i]/kaz+alphaz[i])*dt);
		   i++;
      }

   float *phi_Txx_x1,*phi_Txz_x1,*phi_Txz_z1,*phi_Tzz_z1;
   float *phi_Txx_x2,*phi_Txz_x2,*phi_Txz_z2,*phi_Tzz_z2;
   phi_Txx_x1 = alloc1float(nxz);
   phi_Txz_x1 = alloc1float(nxz);
   phi_Txz_z1 = alloc1float(nxz);
   phi_Tzz_z1 = alloc1float(nxz);
   phi_Txx_x2 = alloc1float(nxz);
   phi_Txz_x2 = alloc1float(nxz);
   phi_Txz_z2 = alloc1float(nxz);
   phi_Tzz_z2 = alloc1float(nxz);
   zero1float(phi_Txx_x1,nxz);
   zero1float(phi_Txz_x1,nxz);
   zero1float(phi_Txz_z1,nxz);
   zero1float(phi_Tzz_z1,nxz);
   zero1float(phi_Txx_x2,nxz);
   zero1float(phi_Txz_x2,nxz);
   zero1float(phi_Txz_z2,nxz);
   zero1float(phi_Tzz_z2,nxz);

   float *phi_vx_x1,*phi_vx_z1,*phi_vz_x1,*phi_vz_z1;
   float *phi_vx_x2,*phi_vx_z2,*phi_vz_x2,*phi_vz_z2;
   phi_vx_x1 = alloc1float(nxz);
   phi_vx_z1 = alloc1float(nxz);
   phi_vz_x1 = alloc1float(nxz);
   phi_vz_z1 = alloc1float(nxz);
   phi_vx_x2 = alloc1float(nxz);
   phi_vx_z2 = alloc1float(nxz);
   phi_vz_x2 = alloc1float(nxz);
   phi_vz_z2 = alloc1float(nxz);
   zero1float(phi_vx_x1,nxz);
   zero1float(phi_vx_z1,nxz);
   zero1float(phi_vz_x1,nxz);
   zero1float(phi_vz_z1,nxz);
   zero1float(phi_vx_x2,nxz);
   zero1float(phi_vx_z2,nxz);
   zero1float(phi_vz_x2,nxz);
   zero1float(phi_vz_z2,nxz);

   for(k=0;k<nt;k++)
   {
	   RstaggerPS1order2Ddx1(vx2,d1,nx,nz,dx,dz,kx,kz,0,0); //dvx/dx 
	   RstaggerPS1order2Ddx1(vx2,d2,nx,nz,dx,dz,kx,kz,1,0); //dvx/dz 
	   RstaggerPS1order2Ddx1(vz2,d3,nx,nz,dx,dz,kx,kz,1,0); //dvz/dz 
	   RstaggerPS1order2Ddx1(vz2,d4,nx,nz,dx,dz,kx,kz,0,0); //dvz/dx 

	   i = 0;
	   for(ix=0;ix<nx;ix++)
		   for(iz=0;iz<nz;iz++)
		   {
			   Txx2[i] = dt*(c11[i]*(d1[i]+phi_vx_x2[i]) + c13[i]*(d3[i]+phi_vz_z2[i]) + c15[i]*(d2[i]+d4[i]+phi_vx_z2[i]+phi_vz_x2[i])) + Txx1[i];
			   Tzz2[i] = dt*(c13[i]*(d1[i]+phi_vx_x2[i]) + c33[i]*(d3[i]+phi_vz_z2[i]) + c35[i]*(d2[i]+d4[i]+phi_vx_z2[i]+phi_vz_x2[i])) + Tzz1[i];
			   Txz2[i] = dt*(c15[i]*(d1[i]+phi_vx_x2[i]) + c35[i]*(d3[i]+phi_vz_z2[i]) + c55[i]*(d2[i]+d4[i]+phi_vx_z2[i]+phi_vz_x2[i])) + Txz1[i];
               if(iz<nzpad || ix<nxpad || iz>nz-nzpad || ix>nx-nxpad)
			   {
                   phi_vx_x2[i] = bx[i]*phi_vx_x1[i] + ax1[i]*d1[i];
                   phi_vx_z2[i] = bz[i]*phi_vx_z1[i] + az1[i]*d2[i];
                   phi_vz_z2[i] = bz[i]*phi_vz_z1[i] + az1[i]*d3[i];
                   phi_vz_x2[i] = bx[i]*phi_vz_x1[i] + ax1[i]*d4[i];
               }

			   i++;
		   }
		
	   Txx2[nx*nz/2+nz/2] += 2*Ricker(k*dt,30,0.04,10);
	   Tzz2[nx*nz/2+nz/2] += 2*Ricker(k*dt,30,0.04,10); //source term

	   if(k%100==0)
		   sf_warning("k=%d",k);

	   RstaggerPS1order2Ddx1(Txx2,d1,nx,nz,dx,dz,kx,kz,0,1); //dTxx/dx 
	   RstaggerPS1order2Ddx1(Tzz2,d2,nx,nz,dx,dz,kx,kz,1,1); //dTzz/dz 
	   RstaggerPS1order2Ddx1(Txz2,d3,nx,nz,dx,dz,kx,kz,1,1); //dTxz/dz 
	   RstaggerPS1order2Ddx1(Txz2,d4,nx,nz,dx,dz,kx,kz,0,1); //dTxz/dx 

	   i = 0;
	   for(ix=0;ix<nx;ix++)
		   for(iz=0;iz<nz;iz++)
		   {
			   vx2[i] = dt*(d1[i]+d3[i]+phi_Txx_x2[i]+phi_Txz_z2[i]) + vx1[i];
			   vz2[i] = dt*(d2[i]+d4[i]+phi_Tzz_z2[i]+phi_Txz_x2[i]) + vz1[i];
               if(iz<nzpad || ix<nxpad || iz>nz-nzpad || ix>nx-nxpad)
			   {
			       phi_Txx_x2[i] = bx[i]*phi_Txx_x1[i] + ax1[i]*d1[i];		
			       phi_Txz_z2[i] = bz[i]*phi_Txz_z1[i] + az1[i]*d3[i];		
			       phi_Tzz_z2[i] = bz[i]*phi_Tzz_z1[i] + az1[i]*d2[i];		
			       phi_Txz_x2[i] = bx[i]*phi_Txz_x1[i] + ax1[i]*d4[i];		
			   }
			   i++;
		   }

	   for(ix=nxpad;ix<nx-nxpad;ix++)
	   {
		   i = ix - nxpad;
		   recordvx[i*nt+k] = vx2[ix*nz+nzpad];
		   recordvz[i*nt+k] = vz2[ix*nz+nzpad];
	   }

	   for(i=0;i<nxz;i++)
	   {
		   vx1[i] = vx2[i];
		   vz1[i] = vz2[i];
		   Txx1[i] = Txx2[i];
		   Txz1[i] = Txz2[i];
		   Tzz1[i] = Tzz2[i];
		   phi_vx_x1[i] = phi_vx_x2[i];
		   phi_vx_z1[i] = phi_vx_z2[i];
		   phi_vz_x1[i] = phi_vz_x2[i];
		   phi_vz_z1[i] = phi_vz_z2[i];
		   phi_Txx_x1[i] = phi_Txx_x2[i];
		   phi_Txz_z1[i] = phi_Txz_z2[i];
		   phi_Tzz_z1[i] = phi_Tzz_z2[i];
		   phi_Txz_x1[i] = phi_Txz_x2[i];
	   }
   }

   sf_floatwrite(vx2,nxz,Fo1);
   sf_floatwrite(vz2,nxz,Fo2);
   sf_floatwrite(recordvx,nt*(nx-2*nxpad),Fo3);
   sf_floatwrite(recordvz,nt*(nx-2*nxpad),Fo4);

   t2 = clock();
   timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
   sf_warning("costime %fs",timespent);

   return 0;
}

