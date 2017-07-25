/* 3D three-components TTI  elastic wavefield extrapolation 
   using  rotated staggered pseudo-spectral method with velocity-stress wave 
   equation in heterogeneous media. In order to adapt large time step, we use
   k-space adjustment.

   Copyright (C) 2016 Tongji University, Shanghai, China 
   Authors: Peng Zou and Jiubing Cheng
     
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
#include "vti2tti.h"

/* prepared head files by myself */
#include "_cjb.h"
#include "ricker.h"
#include "kykxkztaper.h"

/*****************************************************************************************/

int main(int argc, char* argv[])
{
   sf_init(argc,argv);
   fftwf_init_threads();
   omp_set_num_threads(30); //Openmp initialize
   omp_set_nested(1);

   clock_t t1, t2;
   float   timespent;

   t1=clock();

   int i,k;
   
   int nt;
   float dt;
   if (!sf_getint("nt",&nt)) nt = 500;
   if (!sf_getfloat("dt",&dt)) dt = 0.001;
   int nxpad,nypad,nzpad;
   if (!sf_getint("nxpad",&nxpad)) nxpad = 20;
   if (!sf_getint("nypad",&nypad)) nypad = 20;
   if (!sf_getint("nzpad",&nzpad)) nzpad = 20;
   sf_warning("nxpad=%d nypad=%d nzpad=%d",nxpad,nypad,nzpad);


   sf_warning("nt=%d dt=%f",nt,dt);


   float vref; //k-space adjustment refrence velocity
   if (!sf_getfloat("vref",&vref)) vref = 1200.0;

   /* setup I/O files */
    sf_file Fvp0,Fvs0,Fepsi,Fdel,Fgam,Fthe,Fphi;
	sf_file Fo1,Fo2,Fo3,Fo4,Fo5,Fo6;

	Fvp0 = sf_input("vp0");
	Fvs0 = sf_input("vs0");
	Fepsi = sf_input("epsi");
	Fdel = sf_input("del");
	Fgam = sf_input("gam");
	Fthe = sf_input("the");
	Fphi = sf_input("phi"); // Thomosen parameters for anisotropic media
	Fo1 = sf_output("Elasticvx");  //particle velocities x-component 
	Fo2 = sf_output("Elasticvy");  //particle velocities y-component 
	Fo3 = sf_output("Elasticvz");  //particle velocities z-component 
	Fo4 = sf_output("recordvx"); // surface record of x-component
	Fo5 = sf_output("recordvy"); // surface record of y-component
	Fo6 = sf_output("recordvz"); // surface record of z-component

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

   // write header file
   puthead3x(Fo1,nz,nx,ny,dz/1000.0,dx/1000.0,dy/1000.,0,0,0);
   puthead3x(Fo2,nz,nx,ny,dz/1000.0,dx/1000.0,dy/1000.,0,0,0);
   puthead3x(Fo3,nz,nx,ny,dz/1000.0,dx/1000.0,dy/1000.,0,0,0);
   puthead3t(Fo4,nt,nx,ny,dt,dx/1000.0,dy/1000.,0,0,0);
   puthead3t(Fo5,nt,nx,ny,dt,dx/1000.0,dy/1000.,0,0,0);
   puthead3t(Fo6,nt,nx,ny,dt,dx/1000.0,dy/1000.,0,0,0);

   // surface seismic record
   float *recordvx,*recordvy,*recordvz;
   recordvx = alloc1float(nt*nx*ny);
   recordvy = alloc1float(nt*nx*ny);
   recordvz = alloc1float(nt*nx*ny);

   // wavefield variable
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

   // elastic parameter
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

   float *vp0,*vs0,*epsi,*del,*gam,*the,*phi;
   vp0 = alloc1float(nxyz);
   vs0 = alloc1float(nxyz);
   epsi = alloc1float(nxyz);
   del = alloc1float(nxyz);
   gam = alloc1float(nxyz);
   the = alloc1float(nxyz);
   phi = alloc1float(nxyz);

   for(i=0;i<nxyz;i++)
   {
	   rho[i] = 1.0;
	   sf_floatread(&vp0[i],1,Fvp0);
	   sf_floatread(&vs0[i],1,Fvs0);
	   sf_floatread(&epsi[i],1,Fepsi);
	   sf_floatread(&del[i],1,Fdel);
	   sf_floatread(&gam[i],1,Fgam);
	   sf_floatread(&the[i],1,Fthe);
	   sf_floatread(&phi[i],1,Fphi);
	   phi[i] *= SF_PI/180.;
	   the[i] *= SF_PI/180.;

	   // This is triclinic stiffness from (Igel,1995) 
	/* c11[i] = 10e6;
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
	   c66[i] = 3e6;*/ 
   }
   // invert Thomson parameter to stiffness
   Thomson2stiffness_3d(vp0,vs0,epsi,del,gam,the,phi,c11,c12,c13,c14,c15,c16,c22,
		                   c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66,nx,ny,nz);
   free(vp0);
   free(vs0);
   free(epsi);
   free(del);
   free(gam);
   free(the);
   free(phi);

   sf_warning("c11=%g",c11[1]);
   sf_warning("c12=%g",c12[1]);
   sf_warning("c13=%g",c13[1]);
   sf_warning("c22=%g",c22[1]);
   sf_warning("c23=%g",c23[1]);
   sf_warning("c33=%g",c33[1]);
   sf_warning("c44=%g",c44[1]);
   sf_warning("c55=%g",c55[1]);
   sf_warning("c66=%g",c66[1]);
   sf_warning("c14=%g",c14[1]);
   sf_warning("c15=%g",c15[1]);
   sf_warning("c16=%g",c16[1]);
   sf_warning("c24=%g",c24[1]);
   sf_warning("c25=%g",c25[1]);
   sf_warning("c26=%g",c26[1]);
   sf_warning("c34=%g",c34[1]);
   sf_warning("c35=%g",c35[1]);
   sf_warning("c36=%g",c36[1]);
   sf_warning("c45=%g",c45[1]);
   sf_warning("c46=%g",c46[1]);
   sf_warning("c56=%g",c56[1]);

   int ix,iy,iz;

   // wavenumber domain variable
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

   float temp;
   sf_complex *f_Rksta_x,*f_Rksta_y,*f_Rksta_z,*b_Rksta_x,*b_Rksta_y,*b_Rksta_z;
   f_Rksta_x = sf_complexalloc(nxyz);
   f_Rksta_y = sf_complexalloc(nxyz);
   f_Rksta_z = sf_complexalloc(nxyz);
   b_Rksta_x = sf_complexalloc(nxyz);
   b_Rksta_y = sf_complexalloc(nxyz);
   b_Rksta_z = sf_complexalloc(nxyz); // pseudo-spectral differential operator 

   for(iy=0; iy < ny; iy++) 
	   for(ix=0; ix < nx; ix++)
		   for (iz=0; iz < nz; iz++)
		   {
			   i = iy*nx*nz + ix*nz + iz;
			   temp = vref*dt/2.0*sqrt(kx[i]*kx[i]+ky[i]*ky[i]+kz[i]*kz[i]) + 1.e-10; //avoid dividing 0
			   if(ix>nx/2-1&&ix<nx/2+1)
			   {
				   f_Rksta_x[i] = I*kx[i]*cexpf(I*(kx[i]*dx+ky[i]*dy)/2.0)*sin(temp)/temp;
				   b_Rksta_x[i] = I*kx[i]*cexpf(-I*(kx[i]*dx+ky[i]*dy)/2.0)*sin(temp)/temp;
			   }
			   else
			   {
				   f_Rksta_x[i] = I*kx[i]*cexpf(I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
				   b_Rksta_x[i] = I*kx[i]*cexpf(-I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
			   }
			   if(iy>ny/2-1&&iy<ny/2+1)
			   {
				   f_Rksta_y[i] = I*ky[i]*cexpf(I*(ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
				   b_Rksta_y[i] = I*ky[i]*cexpf(-I*(ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
			   }
			   else
			   {
				   f_Rksta_y[i] = I*ky[i]*cexpf(I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
				   b_Rksta_y[i] = I*ky[i]*cexpf(-I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
			   }
			   if(iz>nz/2-1&&iz<nz/2+1)
			   {
				   f_Rksta_z[i] = I*kz[i]*cexpf(I*(-kz[i]*dz+kx[i]*dx)/2.0)*sin(temp)/temp;
				   b_Rksta_z[i] = I*kz[i]*cexpf(-I*(-kz[i]*dz+kx[i]*dx)/2.0)*sin(temp)/temp;
			   }
			   else
			   {
				   f_Rksta_z[i] = I*kz[i]*cexpf(I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
				   b_Rksta_z[i] = I*kz[i]*cexpf(-I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*sin(temp)/temp;
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

   // C-PML boundary conditions from (Komatitsch and Martin 2007)
   float *dx1,*dy1,*dz1,*ax1,*ay1,*az1,*bx,*by,*bz,*alphax,*alphay,*alphaz;
   dx1 = alloc1float(nxyz);
   dy1 = alloc1float(nxyz);
   dz1 = alloc1float(nxyz);
   ax1 = alloc1float(nxyz);
   ay1 = alloc1float(nxyz);
   az1 = alloc1float(nxyz);
   bx = alloc1float(nxyz);
   by = alloc1float(nxyz);
   bz = alloc1float(nxyz);
   alphax = alloc1float(nxyz);
   alphay = alloc1float(nxyz);
   alphaz = alloc1float(nxyz);
   zero1float(dx1,nxyz);
   zero1float(dy1,nxyz);
   zero1float(dz1,nxyz);
   zero1float(ax1,nxyz);
   zero1float(ay1,nxyz);
   zero1float(az1,nxyz);
   zero1float(bx,nxyz);
   zero1float(by,nxyz);
   zero1float(bz,nxyz);
   zero1float(alphax,nxyz);
   zero1float(alphay,nxyz);
   zero1float(alphaz,nxyz);

   float kax,kay,kaz,alphax_max,alphay_max,alphaz_max,d0x,d0y,d0z;
   kax = 1.;
   kay = 1.;
   kaz = 1.;
   alphax_max = SF_PI*20; // 20 is the dominant frequency of the source
   alphay_max = SF_PI*20;
   alphaz_max = SF_PI*20;
   d0x = -3.0*4500*log(0.001)/2./nxpad/dx;
   d0y = -3.0*4500*log(0.001)/2./nypad/dy;
   d0z = -3.0*4500*log(0.001)/2./nzpad/dz;

   i = 0;
   for(iy=0;iy<ny;iy++)
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
           if(iy<nypad)
           {
               alphay[i] = alphay_max*iy/nypad;
               dy1[i] = d0y*pow((nypad-iy)*1.0/nypad,2);
               by[i] = exp(-(dy1[i]/kay+alphay[i])*dt);
               ay1[i] = dy1[i]/kay/(dy1[i]+kay*alphay[i]+0.001)*(by[i]-1.);
           }
           if(iy>=ny-nypad)
           {
               alphay[i] = alphay_max*(ny-iy)/nypad;
               dy1[i] = d0y*pow((iy+nypad-ny)*1.0/nypad,2);
               by[i] = exp(-(dy1[i]/kay+alphay[i])*dt);
               ay1[i] = dy1[i]/kay/(dy1[i]+kay*alphay[i]+0.001)*(by[i]-1.);
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
           i++;
      }

   float *phi_vx_x1,*phi_vx_y1,*phi_vx_z1,*phi_vy_x1,*phi_vy_y1,*phi_vy_z1,*phi_vz_x1,*phi_vz_y1,*phi_vz_z1;
   float *phi_vx_x2,*phi_vx_y2,*phi_vx_z2,*phi_vy_x2,*phi_vy_y2,*phi_vy_z2,*phi_vz_x2,*phi_vz_y2,*phi_vz_z2;
   phi_vx_x1 = alloc1float(nxyz);
   phi_vx_y1 = alloc1float(nxyz);
   phi_vx_z1 = alloc1float(nxyz);
   phi_vy_x1 = alloc1float(nxyz);
   phi_vy_y1 = alloc1float(nxyz);
   phi_vy_z1 = alloc1float(nxyz);
   phi_vz_x1 = alloc1float(nxyz);
   phi_vz_y1 = alloc1float(nxyz);
   phi_vz_z1 = alloc1float(nxyz);
   phi_vx_x2 = alloc1float(nxyz);
   phi_vx_y2 = alloc1float(nxyz);
   phi_vx_z2 = alloc1float(nxyz);
   phi_vy_x2 = alloc1float(nxyz);
   phi_vy_y2 = alloc1float(nxyz);
   phi_vy_z2 = alloc1float(nxyz);
   phi_vz_x2 = alloc1float(nxyz);
   phi_vz_y2 = alloc1float(nxyz);
   phi_vz_z2 = alloc1float(nxyz);
   zero1float(phi_vx_x1,nxyz);
   zero1float(phi_vx_y1,nxyz);
   zero1float(phi_vx_z1,nxyz);
   zero1float(phi_vy_x1,nxyz);
   zero1float(phi_vy_y1,nxyz);
   zero1float(phi_vy_z1,nxyz);
   zero1float(phi_vz_x1,nxyz);
   zero1float(phi_vz_y1,nxyz);
   zero1float(phi_vz_z1,nxyz);
   zero1float(phi_vx_x2,nxyz);
   zero1float(phi_vx_y2,nxyz);
   zero1float(phi_vx_z2,nxyz);
   zero1float(phi_vy_x2,nxyz);
   zero1float(phi_vy_y2,nxyz);
   zero1float(phi_vy_z2,nxyz);
   zero1float(phi_vz_x2,nxyz);
   zero1float(phi_vz_y2,nxyz);
   zero1float(phi_vz_z2,nxyz);

   float *phi_Txx_x1,*phi_Tyy_y1,*phi_Tzz_z1,*phi_Txy_x1,*phi_Txy_y1,*phi_Txz_x1,*phi_Txz_z1,*phi_Tyz_y1,*phi_Tyz_z1;
   float *phi_Txx_x2,*phi_Tyy_y2,*phi_Tzz_z2,*phi_Txy_x2,*phi_Txy_y2,*phi_Txz_x2,*phi_Txz_z2,*phi_Tyz_y2,*phi_Tyz_z2;
   phi_Txx_x1 = alloc1float(nxyz);
   phi_Tyy_y1 = alloc1float(nxyz);
   phi_Tzz_z1 = alloc1float(nxyz);
   phi_Txy_x1 = alloc1float(nxyz);
   phi_Txy_y1 = alloc1float(nxyz);
   phi_Txz_x1 = alloc1float(nxyz);
   phi_Txz_z1 = alloc1float(nxyz);
   phi_Tyz_y1 = alloc1float(nxyz);
   phi_Tyz_z1 = alloc1float(nxyz);
   phi_Txx_x2 = alloc1float(nxyz);
   phi_Tyy_y2 = alloc1float(nxyz);
   phi_Tzz_z2 = alloc1float(nxyz);
   phi_Txy_x2 = alloc1float(nxyz);
   phi_Txy_y2 = alloc1float(nxyz);
   phi_Txz_x2 = alloc1float(nxyz);
   phi_Txz_z2 = alloc1float(nxyz);
   phi_Tyz_y2 = alloc1float(nxyz);
   phi_Tyz_z2 = alloc1float(nxyz);
   zero1float(phi_Txx_x1,nxyz);
   zero1float(phi_Tyy_y1,nxyz);
   zero1float(phi_Tzz_z1,nxyz);
   zero1float(phi_Txy_x1,nxyz);
   zero1float(phi_Txy_y1,nxyz);
   zero1float(phi_Txz_x1,nxyz);
   zero1float(phi_Txz_z1,nxyz);
   zero1float(phi_Tyz_y1,nxyz);
   zero1float(phi_Tyz_z1,nxyz);
   zero1float(phi_Txx_x2,nxyz);
   zero1float(phi_Tyy_y2,nxyz);
   zero1float(phi_Tzz_z2,nxyz);
   zero1float(phi_Txy_x2,nxyz);
   zero1float(phi_Txy_y2,nxyz);
   zero1float(phi_Txz_x2,nxyz);
   zero1float(phi_Txz_z2,nxyz);
   zero1float(phi_Tyz_y2,nxyz);
   zero1float(phi_Tyz_z2,nxyz);
   // end of C-PML parameters definition

   for(k=0;k<nt;k++)
   {
	   RkstaggerPS1order3Ddx(vx2,d1,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,0,1); //dvx/dx
	   RkstaggerPS1order3Ddx(vx2,d2,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,1,1); //dvx/dy
	   RkstaggerPS1order3Ddx(vx2,d3,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,2,1); //dvx/dz
	   RkstaggerPS1order3Ddx(vy2,d4,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,0,1); //dvy/dx
	   RkstaggerPS1order3Ddx(vy2,d5,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,1,1); //dvy/dy
	   RkstaggerPS1order3Ddx(vy2,d6,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,2,1); //dvy/dz
	   RkstaggerPS1order3Ddx(vz2,d7,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,0,1); //dvz/dx
	   RkstaggerPS1order3Ddx(vz2,d8,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,1,1); //dvz/dy
	   RkstaggerPS1order3Ddx(vz2,d9,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,2,1);//dvz/dz

	  #pragma omp parallel for private(iy,ix,iz,i) schedule(dynamic)\
		   shared(Txx1,Txx2,dt,c11,c12,c13,c14,c15,c16,c22,c23,c24,c25,c26,c33,c34,c35,c36,c44,c45,c46,c55,c56,c66,d1,d2,\
				   d3,d4,d5,d6,d7,d8,d9,Tyy1,Tyy2,Tzz1,Tzz2,Txy1,Txy2,Txz1,Txz2,phi_vx_x1,phi_vx_x2,phi_vy_y1,phi_vy_y2,\
				   phi_vz_z2,phi_vz_z1,phi_vy_z2,phi_vy_z1,phi_vz_y2,phi_vz_y1,phi_vx_z2,phi_vx_z1,phi_vz_x2,phi_vz_x1,\
				   phi_vx_y2,phi_vx_y1,phi_vy_x2,phi_vy_x1,nx,ny,nz,nxpad,nypad,nzpad,bx,by,bz,ax1,ay1,az1)
	  for(iy=0;iy<ny;iy++)
	    for(ix=0;ix<nx;ix++)
		   for(iz=0;iz<nz;iz++)	 
	   {
		   i = iy*nx*nz+ix*nz+iz;
           Txx2[i] = Txx1[i] + dt*(c11[i]*d1[i]+c12[i]*d5[i]+c13[i]*d9[i]+c14[i]*(d6[i]+d8[i])
                     +c15[i]*(d3[i]+d7[i])+c16[i]*(d2[i]+d4[i]));

           Tyy2[i] = Tyy1[i] + dt*(c12[i]*d1[i]+c22[i]*d5[i]+c23[i]*d9[i]+c24[i]*(d6[i]+d8[i])
                     +c25[i]*(d3[i]+d7[i])+c26[i]*(d2[i]+d4[i]));

           Tzz2[i] = Tzz1[i] + dt*(c13[i]*d1[i]+c23[i]*d5[i]+c33[i]*d9[i]+c34[i]*(d6[i]+d8[i])
                     +c35[i]*(d3[i]+d7[i])+c36[i]*(d2[i]+d4[i]));

           Tyz2[i] = Tyz1[i] + dt*(c14[i]*d1[i]+c24[i]*d5[i]+c34[i]*d9[i]+c44[i]*(d6[i]+d8[i])
                     +c45[i]*(d3[i]+d7[i])+c46[i]*(d2[i]+d4[i]));

           Txz2[i] = Txz1[i] + dt*(c15[i]*d1[i]+c25[i]*d5[i]+c35[i]*d9[i]+c45[i]*(d6[i]+d8[i])
                     +c55[i]*(d3[i]+d7[i])+c56[i]*(d2[i]+d4[i]));

           Txy2[i] = Txy1[i] + dt*(c16[i]*d1[i]+c26[i]*d5[i]+c36[i]*d9[i]+c46[i]*(d6[i]+d8[i])
                     +c56[i]*(d3[i]+d7[i])+c66[i]*(d2[i]+d4[i]));

		   if(ix<nxpad || ix>nx-nxpad-1)
		   {
			   Txx2[i] += dt*(c11[i]*phi_vx_x2[i] + c15[i]*phi_vz_x2[i] + c16[i]*phi_vy_x2[i]); 
			   Tyy2[i] += dt*(c12[i]*phi_vx_x2[i] + c25[i]*phi_vz_x2[i] + c26[i]*phi_vy_x2[i]);
			   Tzz2[i] += dt*(c13[i]*phi_vx_x2[i] + c35[i]*phi_vz_x2[i] + c36[i]*phi_vy_x2[i]);
			   Tyz2[i] += dt*(c14[i]*phi_vx_x2[i] + c45[i]*phi_vz_x2[i] + c46[i]*phi_vy_x2[i]);
			   Txz2[i] += dt*(c15[i]*phi_vx_x2[i] + c55[i]*phi_vz_x2[i] + c56[i]*phi_vy_x2[i]);
			   Txy2[i] += dt*(c16[i]*phi_vx_x2[i] + c56[i]*phi_vz_x2[i] + c66[i]*phi_vy_x2[i]);
			   phi_vx_x2[i] = bx[i]*phi_vx_x1[i] + ax1[i]*d1[i];
			   phi_vy_x2[i] = bx[i]*phi_vy_x1[i] + ax1[i]*d4[i];
			   phi_vz_x2[i] = bx[i]*phi_vz_x1[i] + ax1[i]*d7[i];
		   }
		   if(iy<nypad || iy>ny-nypad-1)
		   {
			   Txx2[i] += dt*(c16[i]*phi_vx_y2[i] + c14[i]*phi_vz_y2[i] + c12[i]*phi_vy_y2[i]); 
			   Tyy2[i] += dt*(c26[i]*phi_vx_y2[i] + c24[i]*phi_vz_y2[i] + c22[i]*phi_vy_y2[i]);
			   Tzz2[i] += dt*(c36[i]*phi_vx_y2[i] + c34[i]*phi_vz_y2[i] + c23[i]*phi_vy_y2[i]);
			   Tyz2[i] += dt*(c46[i]*phi_vx_y2[i] + c44[i]*phi_vz_y2[i] + c24[i]*phi_vy_y2[i]);
			   Txz2[i] += dt*(c56[i]*phi_vx_y2[i] + c45[i]*phi_vz_y2[i] + c25[i]*phi_vy_y2[i]);
			   Txy2[i] += dt*(c66[i]*phi_vx_y2[i] + c46[i]*phi_vz_y2[i] + c26[i]*phi_vy_y2[i]);
			   phi_vx_y2[i] = by[i]*phi_vx_y1[i] + ay1[i]*d2[i];
			   phi_vy_y2[i] = by[i]*phi_vy_y1[i] + ay1[i]*d5[i];
			   phi_vz_y2[i] = by[i]*phi_vz_y1[i] + ay1[i]*d8[i];
		   }
		   if(iz<nzpad || iz>nz-nzpad-1)
		   {
			   Txx2[i] += dt*(c15[i]*phi_vx_z2[i] + c13[i]*phi_vz_z2[i] + c14[i]*phi_vy_z2[i]); 
			   Tyy2[i] += dt*(c25[i]*phi_vx_z2[i] + c23[i]*phi_vz_z2[i] + c24[i]*phi_vy_z2[i]);
			   Tzz2[i] += dt*(c35[i]*phi_vx_z2[i] + c33[i]*phi_vz_z2[i] + c34[i]*phi_vy_z2[i]);
			   Tyz2[i] += dt*(c45[i]*phi_vx_z2[i] + c34[i]*phi_vz_z2[i] + c44[i]*phi_vy_z2[i]);
			   Txz2[i] += dt*(c55[i]*phi_vx_z2[i] + c35[i]*phi_vz_z2[i] + c45[i]*phi_vy_z2[i]);
			   Txy2[i] += dt*(c56[i]*phi_vx_z2[i] + c36[i]*phi_vz_z2[i] + c46[i]*phi_vy_z2[i]); 
			   phi_vx_z2[i] = bz[i]*phi_vx_z1[i] + az1[i]*d3[i];
			   phi_vy_z2[i] = bz[i]*phi_vy_z1[i] + az1[i]*d6[i];
			   phi_vz_z2[i] = bz[i]*phi_vz_z1[i] + az1[i]*d9[i];
		   }
	   }

	   Txx2[nz*nx*(ny/2)+nx/2*nz+nzpad] += Ricker(k*dt,20,0.04,10);
	   Tyy2[nz*nx*(ny/2)+nx/2*nz+nzpad] += Ricker(k*dt,20,0.04,10);
	   Tzz2[nz*nx*(ny/2)+nx/2*nz+nzpad] += Ricker(k*dt,20,0.04,10); //source term

	   if(k%10==0)
	   sf_warning("k=%d",k);

	  RkstaggerPS1order3Ddx(Txx2,d1,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,0,0); //dTxx/dx
	  RkstaggerPS1order3Ddx(Txy2,d2,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,1,0);//dTxy/dy
	  RkstaggerPS1order3Ddx(Txz2,d3,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,2,0); //dTxz/dz
	  RkstaggerPS1order3Ddx(Txy2,d4,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,0,0); //dTxy/dx
	  RkstaggerPS1order3Ddx(Tyy2,d5,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,1,0); //dTyy/dy
	  RkstaggerPS1order3Ddx(Tyz2,d6,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,2,0); //dTyz/dz
	  RkstaggerPS1order3Ddx(Txz2,d7,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,0,0); //dTxz/dx
	  RkstaggerPS1order3Ddx(Tyz2,d8,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,1,0); //dTyz/dy
	  RkstaggerPS1order3Ddx(Tzz2,d9,nx,ny,nz,f_Rksta_x,f_Rksta_y,f_Rksta_z,b_Rksta_x,b_Rksta_y,b_Rksta_z,2,0); //dTzz/dz

	  #pragma omp parallel for private(iy,ix,iz,i) schedule(dynamic)\
		   shared(nx,ny,nz,nxpad,nzpad,nypad,vx2,vx1,vz1,vz2,vy1,vy2,d1,d2,d3,d4,d5,d6,d7,d8,d9,rho,bx,by,bz,\
				   ax1,ay1,az1,phi_Txx_x2,phi_Txx_x1,phi_Txy_x2,phi_Txy_x1,phi_Txz_x2,phi_Txz_x1,phi_Txy_y2,\
				   phi_Txy_y1,phi_Tyy_y2,phi_Tyy_y1,phi_Tyz_y2,phi_Tyz_y1,phi_Txz_z2,phi_Txz_z1,phi_Tyz_z2,\
				   phi_Tyz_z1,phi_Tzz_z2,phi_Tzz_z1)
	 for(iy=0;iy<ny;iy++)
	   for(ix=0;ix<nx;ix++)
		 for(iz=0;iz<nz;iz++)	
	   {
		   i = iy*nx*nz+ix*nz+iz;
		   vx2[i] = vx1[i] + dt*(d1[i]+d2[i]+d3[i])/rho[i];
		   vy2[i] = vy1[i] + dt*(d4[i]+d5[i]+d6[i])/rho[i];
		   vz2[i] = vz1[i] + dt*(d7[i]+d8[i]+d9[i])/rho[i];
		   if(ix<nxpad || ix>nx-nxpad-1)
		   {
			   vx2[i] += dt*phi_Txx_x2[i];
			   vy2[i] += dt*phi_Txy_x2[i];
			   vz2[i] += dt*phi_Txz_x2[i];
			   phi_Txx_x2[i] = bx[i]*phi_Txx_x1[i] + ax1[i]*d1[i];
			   phi_Txy_x2[i] = bx[i]*phi_Txy_x1[i] + ax1[i]*d4[i];
			   phi_Txz_x2[i] = bx[i]*phi_Txz_x1[i] + ax1[i]*d7[i];
		   }
		   if(iy<nypad || iy>ny-nypad-1)
		   {
			   vx2[i] += dt*phi_Txy_y2[i];
			   vy2[i] += dt*phi_Tyy_y2[i];
			   vz2[i] += dt*phi_Tyz_y2[i];
			   phi_Txy_y2[i] = by[i]*phi_Txy_y1[i] + ay1[i]*d2[i];
			   phi_Tyy_y2[i] = by[i]*phi_Tyy_y1[i] + ay1[i]*d5[i];
			   phi_Tyz_y2[i] = by[i]*phi_Tyz_y1[i] + ay1[i]*d8[i];
		   }
		   if(iz<nzpad || iz>nz-nzpad-1)
		   {
			   vx2[i] += dt*phi_Txz_z2[i];
			   vy2[i] += dt*phi_Tyz_z2[i];
			   vz2[i] += dt*phi_Tzz_z2[i];
			   phi_Txz_z2[i] = bz[i]*phi_Txz_z1[i] + az1[i]*d3[i];
			   phi_Tyz_z2[i] = bz[i]*phi_Tyz_z1[i] + az1[i]*d6[i];
			   phi_Tzz_z2[i] = bz[i]*phi_Tzz_z1[i] + az1[i]*d9[i];
		   }
	   }

	 #pragma omp parallel for private(iy,ix,i) schedule(dynamic)\
	 shared(recordvx,recordvy,recordvz,vx2,vy2,vz2,nx,ny)
	 for(iy=0;iy<ny;iy++)
	   for(ix=0;ix<nx;ix++)
	   {
		   i = k*nx*ny + iy*nx + ix;
		   recordvx[i] = vx2[(iy*nx+ix)*nz+nzpad]; 
		   recordvy[i] = vy2[(iy*nx+ix)*nz+nzpad]; 
		   recordvz[i] = vz2[(iy*nx+ix)*nz+nzpad]; 
	   }

	 #pragma omp parallel for private(i) schedule(dynamic)\
	 shared(vx1,vx2,vy1,vy2,vz1,vz2,Txx1,Txx2,Tyy1,Tyy2,Tzz1,Tzz2,Txy1,Txy2,Txz1,Txz2,Tyz1,Tyz2,phi_vx_x1,phi_vx_x2,\
			 phi_vx_y1,phi_vx_y2,phi_vx_z1,phi_vx_z2,phi_vy_x1,phi_vy_x2,phi_vy_y1,phi_vy_y2,phi_vy_z1,phi_vy_z2,\
			 phi_vz_x1,phi_vz_x2,phi_vz_y1,phi_vz_y2,phi_vz_z1,phi_vz_z2,phi_Txx_x1,phi_Txx_x2,phi_Txy_x1,phi_Txy_x2,\
			 phi_Txz_x1,phi_Txz_x2,phi_Txy_y1,phi_Txy_y2,phi_Tyy_y1,phi_Tyy_y2,phi_Tyz_y1,phi_Tyz_y2,phi_Txz_z1,phi_Txz_z2,\
			 phi_Tyz_z1,phi_Tyz_z2,phi_Tzz_z1,phi_Tzz_z2,nxyz)
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
		   phi_vx_x1[i] = phi_vx_x2[i];
		   phi_vx_y1[i] = phi_vx_y2[i];
		   phi_vx_z1[i] = phi_vx_z2[i];
		   phi_vy_x1[i] = phi_vy_x2[i];
		   phi_vy_y1[i] = phi_vy_y2[i];
		   phi_vy_z1[i] = phi_vy_z2[i];
		   phi_vz_x1[i] = phi_vz_x2[i];
		   phi_vz_y1[i] = phi_vz_y2[i];
		   phi_vz_z1[i] = phi_vz_z2[i];
		   phi_Txx_x1[i] = phi_Txx_x2[i];
		   phi_Txy_x1[i] = phi_Txy_x2[i];
		   phi_Txz_x1[i] = phi_Txz_x2[i];
		   phi_Txy_y1[i] = phi_Txy_y2[i];
		   phi_Tyy_y1[i] = phi_Tyy_y2[i];
		   phi_Tyz_y1[i] = phi_Tyz_y2[i];
		   phi_Txz_z1[i] = phi_Txz_z2[i];
		   phi_Tyz_z1[i] = phi_Tyz_z2[i];
		   phi_Tzz_z1[i] = phi_Tzz_z2[i];
	   }
   } // end of nt loop

   for(i=0;i<nxyz;i++)
   {
	   sf_floatwrite(&vx2[i],1,Fo1);
	   sf_floatwrite(&vy2[i],1,Fo2);
	   sf_floatwrite(&vz2[i],1,Fo3);
   }

   for(iy=0;iy<ny;iy++)
   for(ix=0;ix<nx;ix++)
   for(k=0;k<nt;k++)	 
   {
	   i = k*nx*ny+iy*nx+ix;
	   sf_floatwrite(&recordvx[i],1,Fo4);
	   sf_floatwrite(&recordvy[i],1,Fo5);
	   sf_floatwrite(&recordvz[i],1,Fo6);
   }

   t2 = clock();
   timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
   sf_warning("costime %fs",timespent);

   free(kx);
   free(ky);
   free(kz);

   free(f_Rksta_x);
   free(f_Rksta_y);
   free(f_Rksta_z);
   free(b_Rksta_x);
   free(b_Rksta_y);
   free(b_Rksta_z);

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
   free(recordvx);
   free(recordvy);
   free(recordvz);

   free(phi_Txx_x1);
   free(phi_Txy_x1);
   free(phi_Txz_x1);
   free(phi_Txy_y1);
   free(phi_Tyy_y1);
   free(phi_Tyz_y1);
   free(phi_Txz_z1);
   free(phi_Tyz_z1);
   free(phi_Tzz_z1);
   free(phi_Txx_x2);
   free(phi_Txy_x2);
   free(phi_Txz_x2);
   free(phi_Txy_y2);
   free(phi_Tyy_y2);
   free(phi_Tyz_y2);
   free(phi_Txz_z2);
   free(phi_Tyz_z2);
   free(phi_Tzz_z2);

   free(phi_vx_x1);
   free(phi_vx_y1);
   free(phi_vx_z1);
   free(phi_vy_x1);
   free(phi_vy_y1);
   free(phi_vy_z1);
   free(phi_vz_x1);
   free(phi_vz_y1);
   free(phi_vz_z1);
   free(phi_vx_x2);
   free(phi_vx_y2);
   free(phi_vx_z2);
   free(phi_vy_x2);
   free(phi_vy_y2);
   free(phi_vy_z2);
   free(phi_vz_x2);
   free(phi_vz_y2);
   free(phi_vz_z2);

   free(c11);
   free(c12);
   free(c13);
   free(c14);
   free(c15);
   free(c16);
   free(c22);
   free(c23);
   free(c24);
   free(c25);
   free(c26);
   free(c33);
   free(c34);
   free(c35);
   free(c36);
   free(c44);
   free(c45);
   free(c46);
   free(c55);
   free(c56);
   free(c66);

   free(dx1);
   free(dy1);
   free(dz1);
   free(ax1);
   free(ay1);
   free(az1);
   free(bx);
   free(by);
   free(bz);
   free(alphax);
   free(alphay);
   free(alphaz);

   return 0;
}

