/* 2-D two-components arbitrary isotropic elastic wavefield
   extrapolation using  pseudo-spectral or finite difference method base on 
   Lebedev grid configuration with velocity-stress wave 
   equation in heterogeneous media

   Copyright (C) 2016-1-20 Tongji University, Shanghai, China 
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

   clock_t t1, t2, t3;
   float   timespent;

   t1=clock();

   int i,k;
   
   int nt,flag;
   float dt;
   if (!sf_getint("nt",&nt)) nt = 500;
   if (!sf_getint("flag",&flag)) flag = 1;
   if (!sf_getfloat("dt",&dt)) dt = 0.001;

   if(flag==0)
	   sf_warning("Using pseudo-spectral method");
   else
	   sf_warning("Using finite-difference method");
   sf_warning("nt=%d dt=%f",nt,dt);

   int nxpad,nzpad;
   if (!sf_getint("nxpad",&nxpad)) nxpad = 20;
   if (!sf_getint("nzpad",&nzpad)) nzpad = 20;
   sf_warning("nxpad=%d  nzpad=%d",nxpad,nzpad);

   sf_warning("read anisotropic elastic parameters"); 

   /* setup I/O files */
	sf_file Fvp0,Fvs0,Fepsi,Fdel,Fthe;
	sf_file Fo1,Fo2;

	Fvp0 = sf_input("vp0"); //Vp0
	Fvs0 = sf_input("vs0"); //Vs0
	Fepsi = sf_input("epsi"); //epsi
	Fdel = sf_input("del"); //del
	Fthe = sf_input("the"); //the 

	Fo1 = sf_output("Elasticvx");  //particle velocities x-component 
	Fo2 = sf_output("Elasticvz");  //particle velocities z-component 

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

   //From Lisitsa(2010, Geophysical Prospecting and 2011,Numerical Analysis and 
   //Applications), I use vx_1 denote one set of grid and vx_2 denote another set grid.

   float *vx1_1,*vx2_1,*vx1_2,*vx2_2;
   float *vz1_1,*vz2_1,*vz1_2,*vz2_2;
   vx1_1 = alloc1float(nxz);
   vx2_1 = alloc1float(nxz);
   vx1_2 = alloc1float(nxz);
   vx2_2 = alloc1float(nxz);
   vz1_1 = alloc1float(nxz);
   vz2_1 = alloc1float(nxz);
   vz1_2 = alloc1float(nxz);
   vz2_2 = alloc1float(nxz);
   zero1float(vx1_1,nxz);
   zero1float(vx2_1,nxz);
   zero1float(vx1_2,nxz);
   zero1float(vx2_2,nxz);
   zero1float(vz1_1,nxz);
   zero1float(vz2_1,nxz);
   zero1float(vz1_2,nxz);
   zero1float(vz2_2,nxz);

   float *Txx1_1,*Txx2_1,*Txx1_2,*Txx2_2;
   float *Tzz1_1,*Tzz2_1,*Tzz1_2,*Tzz2_2;
   float *Txz1_1,*Txz2_1,*Txz1_2,*Txz2_2;
   Txx1_1 = alloc1float(nxz);
   Txx2_1 = alloc1float(nxz);
   Txx1_2 = alloc1float(nxz);
   Txx2_2 = alloc1float(nxz);
   Tzz1_1 = alloc1float(nxz);
   Tzz2_1 = alloc1float(nxz);
   Tzz1_2 = alloc1float(nxz);
   Tzz2_2 = alloc1float(nxz);
   Txz1_1 = alloc1float(nxz);
   Txz2_1 = alloc1float(nxz);
   Txz1_2 = alloc1float(nxz);
   Txz2_2 = alloc1float(nxz);

   zero1float(Txx1_1,nxz);
   zero1float(Txx2_1,nxz);
   zero1float(Txx1_2,nxz);
   zero1float(Txx2_2,nxz);
   zero1float(Tzz1_1,nxz);
   zero1float(Tzz2_1,nxz);
   zero1float(Tzz1_2,nxz);
   zero1float(Tzz2_2,nxz);
   zero1float(Txz1_1,nxz);
   zero1float(Txz2_1,nxz);
   zero1float(Txz1_2,nxz);
   zero1float(Txz2_2,nxz);

   float *d1,*d2,*d3,*d4,*d5,*d6,*d7,*d8; //derivation variable
   d1 = alloc1float(nxz);
   d2 = alloc1float(nxz);
   d3 = alloc1float(nxz);
   d4 = alloc1float(nxz);
   d5 = alloc1float(nxz);
   d6 = alloc1float(nxz);
   d7 = alloc1float(nxz);
   d8 = alloc1float(nxz);

   float *vp,*vs,*ep,*de,*th; //input model parameter
   vp = alloc1float(nxz);
   vs = alloc1float(nxz);
   ep = alloc1float(nxz);
   de = alloc1float(nxz);
   th = alloc1float(nxz);
	   
   /*read model parameter*/
   sf_floatread(vp,nxz,Fvp0);
   sf_floatread(vs,nxz,Fvs0);
   sf_floatread(ep,nxz,Fepsi);
   sf_floatread(de,nxz,Fdel);
   sf_floatread(th,nxz,Fthe);

   float *alpha_x,*alpha_z;
   float alpha_max,Vmax;
   alpha_x = alloc1float(nxpad);
   alpha_z = alloc1float(nzpad);
   alpha_max = 4.0;
   Vmax = 6000.0;

   for(i=0;i<nxpad;i++)
	   alpha_x[i] = alpha_max*Vmax/dx*pow(i*1.0/nxpad,4);
   for(i=0;i<nzpad;i++)
       alpha_z[i] = alpha_max*Vmax/dz*pow(i*1.0/nzpad,4); //attenuation boundary 

   float*c11,*c13,*c15,*c33,*c55,*c35;
   c11 = alloc1float(nxz);
   c13 = alloc1float(nxz);
   c15 = alloc1float(nxz);
   c33 = alloc1float(nxz);
   c35 = alloc1float(nxz);
   c55 = alloc1float(nxz);

   for(i=0;i<nxz;i++) th[i] *= SF_PI/180.;

   Thomson2stiffness_2d(vp,vs,ep,de,th,c11,c13,c15,c33,c35,c55,nx,nz);

   free(vp);
   free(vs);
   free(ep);
   free(th);
   free(de);

   float dkz,dkx,kz0,kx0;

   dkx=2*SF_PI/dx/nx;
   dkz=2*SF_PI/dz/nz;

   kx0=-SF_PI/dx;
   kz0=-SF_PI/dz;

   sf_warning("dkx=%f dkz=%f",dkx,dkz);
   float *kx = sf_floatalloc(nxz);
   float *kz = sf_floatalloc(nxz);
   int   ix, iz;
   
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

        sf_complex *f_sta_x,*b_sta_x,*f_sta_z,*b_sta_z;
        f_sta_x = sf_complexalloc(nxz);
        b_sta_x = sf_complexalloc(nxz);
        f_sta_z = sf_complexalloc(nxz);
        b_sta_z = sf_complexalloc(nxz);

        for(i=0;i<nxz;i++)
		{
			f_sta_x[i] = I*kx[i]*cexpf(I*(kx[i]*dx)/2.0);
			b_sta_x[i] = I*kx[i]*cexpf(-I*(kx[i]*dx)/2.0);
			f_sta_z[i] = I*kz[i]*cexpf(I*(kz[i]*dz)/2.0);
			b_sta_z[i] = I*kz[i]*cexpf(-I*(kz[i]*dz)/2.0);
        }

   for(k=0;k<nt;k++)
   {

	   staggerPS1order2Ddx3(vx2_1,d1,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,1); //dvx/dx backward_stagger
	   staggerPS1order2Ddx3(vx2_1,d2,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,0); //dvx/dz forward_stagger
	   staggerPS1order2Ddx3(vz2_1,d3,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,0); //dvz/dz backward_stagger
	   staggerPS1order2Ddx3(vz2_1,d4,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,1); //dvz/dx forward_stagger
	   staggerPS1order2Ddx3(vx2_2,d5,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,0); //dvx/dx backward_stagger
	   staggerPS1order2Ddx3(vx2_2,d6,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,1); //dvx/dz forward_stagger
	   staggerPS1order2Ddx3(vz2_2,d7,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,1); //dvz/dz backward_stagger
	   staggerPS1order2Ddx3(vz2_2,d8,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,0); //dvz/dx forward_stagger

	   for(i=0;i<nxz;i++)
	   {
		     Txx2_1[i] = dt*(c11[i]*d1[i]+c13[i]*d7[i]+c15[i]*(d6[i]+d4[i])) + Txx1_1[i];
		     Txx2_2[i] = dt*(c11[i]*d5[i]+c13[i]*d3[i]+c15[i]*(d2[i]+d8[i])) + Txx1_2[i];

		     Tzz2_1[i] = dt*(c13[i]*d1[i]+c33[i]*d7[i]+c35[i]*(d6[i]+d4[i])) + Tzz1_1[i];
		     Tzz2_2[i] = dt*(c13[i]*d5[i]+c33[i]*d3[i]+c35[i]*(d2[i]+d8[i])) + Tzz1_2[i];

		     Txz2_1[i] = dt*(c15[i]*d1[i]+c35[i]*d7[i]+c55[i]*(d6[i]+d4[i])) + Txz1_1[i];
		     Txz2_2[i] = dt*(c15[i]*d5[i]+c35[i]*d3[i]+c55[i]*(d2[i]+d8[i])) + Txz1_2[i];
	   }
		
	   Txx2_1[(nz/2)*nx+nx/2] += Ricker(k*dt,30,0.04,10);
	   Tzz2_1[(nz/2)*nx+nx/2] += Ricker(k*dt,30,0.04,10); //source term
	   Txx2_2[(nz/2-1)*nx+nx/2] += 0.25*Ricker(k*dt,30,0.04,10);
	   Tzz2_2[(nz/2-1)*nx+nx/2] += 0.25*Ricker(k*dt,30,0.04,10); //source term
	   Txx2_2[(nz/2)*nx+nx/2-1] += 0.25*Ricker(k*dt,30,0.04,10);
	   Tzz2_2[(nz/2)*nx+nx/2-1] += 0.25*Ricker(k*dt,30,0.04,10); //source term
	   Txx2_2[(nz/2-1)*nx+nx/2-1] += 0.25*Ricker(k*dt,30,0.04,10);
	   Tzz2_2[(nz/2-1)*nx+nx/2-1] += 0.25*Ricker(k*dt,30,0.04,10); //source term
	  // Txx2_2[(nz/2)*nx+nx/2] += 0.25*Ricker(k*dt,20,0.04,10);
	  // Tzz2_2[(nz/2)*nx+nx/2] += 0.25*Ricker(k*dt,20,0.04,10); //source term
	   
	   Txx2_2[(nz/2)*nx+nx/2] += Ricker(k*dt,30,0.04,10);
	   Tzz2_2[(nz/2)*nx+nx/2] += Ricker(k*dt,30,0.04,10); //source term
	   Txx2_1[(nz/2+1)*nx+nx/2] += 0.25*Ricker(k*dt,30,0.04,10);
	   Tzz2_1[(nz/2+1)*nx+nx/2] += 0.25*Ricker(k*dt,30,0.04,10); //source term
	   Txx2_1[(nz/2)*nx+nx/2+1] += 0.25*Ricker(k*dt,30,0.04,10);
	   Tzz2_1[(nz/2)*nx+nx/2+1] += 0.25*Ricker(k*dt,30,0.04,10); //source term
	   Txx2_1[(nz/2+1)*nx+nx/2+1] += 0.25*Ricker(k*dt,30,0.04,10);
	   Tzz2_1[(nz/2+1)*nx+nx/2+1] += 0.25*Ricker(k*dt,30,0.04,10); //source term
	  // Txx2_1[(nz/2)*nx+nx/2] += 0.25*Ricker(k*dt,20,0.04,10);
	  // Tzz2_1[(nz/2)*nx+nx/2] += 0.25*Ricker(k*dt,20,0.04,10); //source term

	   if(k%100==0)
		   sf_warning("k=%d",k);

	   staggerPS1order2Ddx3(Txx2_1,d1,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,0); //dTxx/dx 
	   staggerPS1order2Ddx3(Tzz2_1,d2,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,0); //dTzz/dz 
	   staggerPS1order2Ddx3(Txz2_1,d3,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,0); //dTxz/dz 
	   staggerPS1order2Ddx3(Txz2_1,d4,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,0); //dTxz/dx 
	   staggerPS1order2Ddx3(Txx2_2,d5,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,1); //dTxx/dx 
	   staggerPS1order2Ddx3(Tzz2_2,d6,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,1); //dTzz/dz 
	   staggerPS1order2Ddx3(Txz2_2,d7,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,1); //dTxz/dz 
	   staggerPS1order2Ddx3(Txz2_2,d8,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,1); //dTxz/dx 

	   for(i=0;i<nxz;i++)
	   {
			vx2_1[i] = dt*(d1[i]+d7[i]) + vx1_1[i];
			vx2_2[i] = dt*(d5[i]+d3[i]) + vx1_2[i];
					
			vz2_1[i] = dt*(d4[i]+d6[i]) + vz1_1[i];
			vz2_2[i] = dt*(d8[i]+d2[i]) + vz1_2[i];
	   }

	   for(i=0;i<nxz;i++)
	   {
		   vx1_1[i] = vx2_1[i];
		   vx1_2[i] = vx2_2[i];
		   vz1_1[i] = vz2_1[i];
		   vz1_2[i] = vz2_2[i];

		   Txx1_1[i] = Txx2_1[i];
		   Txx1_2[i] = Txx2_2[i];
		   Tzz1_1[i] = Tzz2_1[i];
		   Tzz1_2[i] = Tzz2_2[i];
		   Txz1_1[i] = Txz2_1[i];
		   Txz1_2[i] = Txz2_2[i];
	   }
	   t2 = clock();
   }

/*   sf_complex *xin, *xout;
   xin=sf_complexalloc(nxz);
   xout=sf_complexalloc(nxz);
   fftwf_plan xp;
   fftwf_plan xpi;
   xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,FFTW_FORWARD,FFTW_ESTIMATE);
   xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,FFTW_BACKWARD,FFTW_ESTIMATE);
   for(i=0;i<nxz;i++)
	   xin[i] = sf_cmplx(vx2_2[i], 0.0);
   fftwf_execute(xp);
   for(i=0;i<nxz;i++)
	   xin[i] = I*kx[i]*xout[i]*cexpf(I*(kx[i]-kz[i])*dx/2.0);
   fftwf_execute(xpi);
   for(i=0;i<nxz;i++)
	   vx2_2[i]=creal(xout[i])/nxz;

   for(i=0;i<nxz;i++)
	   xin[i] = sf_cmplx(vz2_2[i], 0.0);
   fftwf_execute(xp);
   for(i=0;i<nxz;i++)
	   xin[i] = I*kx[i]*xout[i]*cexpf(I*(kx[i]+kz[i])*dx/2.0);
   fftwf_execute(xpi);
   for(i=0;i<nxz;i++)
	   vz2_2[i]=creal(xout[i])/nxz;

   fftwf_destroy_plan(xp);
   fftwf_destroy_plan(xpi);
   free(xin);
   free(xout);*/

   i = 0;
   for(ix=0;ix<nx;ix++)
	   for(iz=0;iz<nz;iz++)
	   {
		     vx2_1[i] = (vx2_1[i] + vx2_2[i])/2.;
		     vz2_1[i] = (vz2_1[i] + vz2_2[i])/2.;
		   i++;
	   }
   sf_floatwrite(vx2_1,nxz,Fo1);
   sf_floatwrite(vz2_1,nxz,Fo2);

   timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
   sf_warning("costime %fs",timespent);
   timespent = (float)(t2 - t3)/CLOCKS_PER_SEC/nt*1000;
   sf_warning("each step costime %fms",timespent);

   return 0;
}

