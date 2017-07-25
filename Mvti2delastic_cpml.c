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

   int i,j,k;
   
   int nt,flag;
   float dt;
   if (!sf_getint("nt",&nt)) nt = 500;
   if (!sf_getint("flag",&flag)) flag = 0;
   if (!sf_getfloat("dt",&dt)) dt = 0.001;

   if(flag==0)
	   sf_warning("Using staggered pseudospectral method");
    else if(flag==1)
	   sf_warning("Using rotated staggered pseudospectral method");
   else
	   sf_warning("Using staggered finite-difference method");
   sf_warning("nt=%d dt=%f",nt,dt);

   int nxpad,nzpad;
   if (!sf_getint("nxpad",&nxpad)) nxpad = 20;
   if (!sf_getint("nzpad",&nzpad)) nzpad = 20;
   sf_warning("nxpad=%d  nzpad=%d",nxpad,nzpad);

   sf_warning("read anisotropic elastic parameters"); 

   /* setup I/O files */
	sf_file Fvp0,Fvs0,Fepsi,Fdel;
	sf_file Fo1,Fo2;

	Fvp0 = sf_input("vp0"); //Vp0
	Fvs0 = sf_input("vs0"); //Vs0
	Fepsi = sf_input("epsi"); //epsi
	Fdel = sf_input("del"); //del

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

   float **vx1,**vx2;
   float **vz1,**vz2;
   vx1 = alloc2float(nx,nz);
   vx2 = alloc2float(nx,nz);
   vz1 = alloc2float(nx,nz);
   vz2 = alloc2float(nx,nz);
   zero2float(vx1,nx,nz);
   zero2float(vx2,nx,nz);
   zero2float(vz1,nx,nz);
   zero2float(vz2,nx,nz);

   float **Txx1,**Txx2;
   float **Tzz1,**Tzz2;
   float **Txz1,**Txz2;
   Txx1 = alloc2float(nx,nz);
   Txx2 = alloc2float(nx,nz);
   Tzz1 = alloc2float(nx,nz);
   Tzz2 = alloc2float(nx,nz);
   Txz1 = alloc2float(nx,nz);
   Txz2 = alloc2float(nx,nz);

   zero2float(Txx1,nx,nz);
   zero2float(Txx2,nx,nz);
   zero2float(Tzz1,nx,nz);
   zero2float(Tzz2,nx,nz);
   zero2float(Txz1,nx,nz);
   zero2float(Txz2,nx,nz);

   float **d1,**d2,**d3,**d4; //derivation variable
   d1 = alloc2float(nx,nz);
   d2 = alloc2float(nx,nz);
   d3 = alloc2float(nx,nz);
   d4 = alloc2float(nx,nz);

   float **Vp0,**Vs0,**epsi,**del; //input model parameter
   Vp0 = alloc2float(nx,nz);
   Vs0 = alloc2float(nx,nz);
   epsi = alloc2float(nx,nz);
   del = alloc2float(nx,nz);
	   
   /*read model parameter*/
   for(j=0;j<nx;j++)
	   for(i=0;i<nz;i++)
	   {
		   sf_floatread(&Vp0[i][j],1,Fvp0);
           sf_floatread(&Vs0[i][j],1,Fvs0);
           sf_floatread(&epsi[i][j],1,Fepsi);
           sf_floatread(&del[i][j],1,Fdel);
	   }

  /* for(j=0;j<nx;j++)
	   for(i=0;i<nz;i++)
	   { 
		   if(j<nxpad)
		   {
			   Vp0[i][j] = Vp0[i][nxpad-1];
			   Vs0[i][j] = Vs0[i][nxpad-1];
			   epsi[i][j] = epsi[i][nxpad-1];
			   del[i][j] = del[i][nxpad-1];
		   }
		   if(j>nxpad)
		   {
			   Vp0[i][j] = Vp0[i][nx-nxpad];
			   Vs0[i][j] = Vs0[i][nx-nxpad];
			   epsi[i][j] = epsi[i][nx-nxpad];
			   del[i][j] = del[i][nx-nxpad];
		   }
		   if(i<nzpad)
		   {
			   Vp0[i][j] = Vp0[nzpad-1][j];
			   Vs0[i][j] = Vs0[nzpad-1][j];
			   epsi[i][j] = epsi[nzpad-1][j];
			   del[i][j] = del[nzpad-1][j];
		   }
		   if(i<nzpad)
		   {
			   Vp0[i][j] = Vp0[nz-nzpad][j];
			   Vs0[i][j] = Vs0[nz-nzpad][j];
			   epsi[i][j] = epsi[nz-nzpad][j];
			   del[i][j] = del[nz-nzpad][j];
		   }
	   }*/


   float **c11,**c13,**c33,**c55;
   c11 = alloc2float(nx,nz);
   c13 = alloc2float(nx,nz);
   c33 = alloc2float(nx,nz);
   c55 = alloc2float(nx,nz);

   for(i=0;i<nz;i++)
	   for(j=0;j<nx;j++)
	   {
		   c33[i][j] = Vp0[i][j]*Vp0[i][j];
		   c55[i][j] = Vs0[i][j]*Vs0[i][j];
		   c11[i][j] = c33[i][j]*(1.0 + 2*epsi[i][j]);
		   c13[i][j] = sqrt(((1.0+2.0*del[i][j])*c33[i][j]-c55[i][j])*(c33[i][j]-c55[i][j])) - c55[i][j];
	   } //attenuation and elastic parameter

   float **dx1,**dz1,**ax1,**az1,**bx,**bz,**alphax,**alphaz;
   dx1 = alloc2float(nx,nz);
   ax1 = alloc2float(nx,nz);
   bx = alloc2float(nx,nz);
   alphax = alloc2float(nx,nz);
   dz1 = alloc2float(nx,nz);
   az1 = alloc2float(nx,nz);
   bz = alloc2float(nx,nz);
   alphaz = alloc2float(nx,nz);
   zero2float(dx1,nx,nz);
   zero2float(ax1,nx,nz);
   zero2float(bx,nx,nz);
   zero2float(alphax,nx,nz);
   zero2float(dz1,nx,nz);
   zero2float(az1,nx,nz);
   zero2float(bz,nx,nz);
   zero2float(alphaz,nx,nz);

   float kax,kaz,alphax_max,alphaz_max,d0x,d0z;
   kax = 1.;
   kaz = 1.;
   alphax_max = SF_PI*20.;
   alphaz_max = SF_PI*20.;
   d0x = -3.0*4000*log(0.001)/2./nxpad/dx;
   d0z = -3.0*4000*log(0.001)/2./nzpad/dz;

   for(i=0;i<nz;i++)
	   for(j=0;j<nx;j++)
	   {
		   if(j<nxpad)
		   {
			   alphax[i][j] = alphax_max*(j)/nxpad;
			   dx1[i][j] = d0x*pow((nxpad-j)*1.0/nxpad,2);
			   bx[i][j] = exp(-(dx1[i][j]/kax+alphax[i][j])*dt);
			   ax1[i][j] = dx1[i][j]/kax/(dx1[i][j]+kax*alphax[i][j]+0.001)*(bx[i][j]-1.);
		   }
		   if(j>=nx-nxpad)
		   {
			   alphax[i][j] = alphax_max*(nx-j)/nxpad;
			   dx1[i][j] = d0x*pow((j+nxpad-nx)*1.0/nxpad,2);
			   bx[i][j] = exp(-(dx1[i][j]/kax+alphax[i][j])*dt);
			   ax1[i][j] = dx1[i][j]/kax/(dx1[i][j]+kax*alphax[i][j]+0.001)*(bx[i][j]-1.);
		   }
		   if(i<nzpad)
		   {
			   alphaz[i][j] = alphaz_max*(i)/nzpad;
			   dz1[i][j] = d0z*pow((nzpad-i)*1.0/nzpad,2);
	           bz[i][j] = exp(-(dz1[i][j]/kaz+alphaz[i][j])*dt);
	           az1[i][j] = dz1[i][j]/kaz/(dz1[i][j]+kaz*alphaz[i][j]+0.001)*(bz[i][j]-1.);
		   }
		   if(i>=nz-nzpad)
		   {
			   alphaz[i][j] = alphaz_max*(nz-i)/nzpad;
			   dz1[i][j] = d0z*pow((i+nzpad-nz)*1.0/nzpad,2);
	           bz[i][j] = exp(-(dz1[i][j]/kaz+alphaz[i][j])*dt);
	           az1[i][j] = dz1[i][j]/kaz/(dz1[i][j]+kaz*alphaz[i][j]+0.001)*(bz[i][j]-1.);
		   }
		   bx[i][j] = exp(-(dx1[i][j]/kax+alphax[i][j])*dt);
	       bz[i][j] = exp(-(dz1[i][j]/kaz+alphaz[i][j])*dt);
	  }

   float **phi_Txx_x1,**phi_Txz_x1,**phi_Txz_z1,**phi_Tzz_z1;
   float **phi_Txx_x2,**phi_Txz_x2,**phi_Txz_z2,**phi_Tzz_z2;
   phi_Txx_x1 = alloc2float(nx,nz);
   phi_Txz_x1 = alloc2float(nx,nz);
   phi_Txz_z1 = alloc2float(nx,nz);
   phi_Tzz_z1 = alloc2float(nx,nz);
   phi_Txx_x2 = alloc2float(nx,nz);
   phi_Txz_x2 = alloc2float(nx,nz);
   phi_Txz_z2 = alloc2float(nx,nz);
   phi_Tzz_z2 = alloc2float(nx,nz);
   zero2float(phi_Txx_x1,nx,nz);
   zero2float(phi_Txz_x1,nx,nz);
   zero2float(phi_Txz_z1,nx,nz);
   zero2float(phi_Tzz_z1,nx,nz);
   zero2float(phi_Txx_x2,nx,nz);
   zero2float(phi_Txz_x2,nx,nz);
   zero2float(phi_Txz_z2,nx,nz);
   zero2float(phi_Tzz_z2,nx,nz);

   float **phi_vx_x1,**phi_vx_z1,**phi_vz_x1,**phi_vz_z1;
   float **phi_vx_x2,**phi_vx_z2,**phi_vz_x2,**phi_vz_z2;
   phi_vx_x1 = alloc2float(nx,nz);
   phi_vx_z1 = alloc2float(nx,nz);
   phi_vz_x1 = alloc2float(nx,nz);
   phi_vz_z1 = alloc2float(nx,nz);
   phi_vx_x2 = alloc2float(nx,nz);
   phi_vx_z2 = alloc2float(nx,nz);
   phi_vz_x2 = alloc2float(nx,nz);
   phi_vz_z2 = alloc2float(nx,nz);
   zero2float(phi_vx_x1,nx,nz);
   zero2float(phi_vx_z1,nx,nz);
   zero2float(phi_vz_x1,nx,nz);
   zero2float(phi_vz_z1,nx,nz);
   zero2float(phi_vx_x2,nx,nz);
   zero2float(phi_vx_z2,nx,nz);
   zero2float(phi_vz_x2,nx,nz);
   zero2float(phi_vz_z2,nx,nz);
   //FILE *fp=fopen("recordvx.dat","wb");

        for(k=0;k<nt;k++)
        {
			if(flag==0)
			{
				staggerPS1order2Ddx(vx2,d1,nx,nz,dx,dz,0,1); //dvx/dx backward_stagger
				staggerPS1order2Ddx(vx2,d2,nx,nz,dx,dz,1,0); //dvx/dz forward_stagger
				staggerPS1order2Ddx(vz2,d3,nx,nz,dx,dz,1,1); //dvz/dz backward_stagger
				staggerPS1order2Ddx(vz2,d4,nx,nz,dx,dz,0,0); //dvz/dx forward_stagger
			}
			else if(flag==1)
			{
				RstaggerPS1order2Ddx(vx2,d1,nx,nz,dx,dz,0,1); //dvx/dx 
				RstaggerPS1order2Ddx(vx2,d2,nx,nz,dx,dz,1,1); //dvx/dz 
				RstaggerPS1order2Ddx(vz2,d3,nx,nz,dx,dz,1,1); //dvz/dz 
				RstaggerPS1order2Ddx(vz2,d4,nx,nz,dx,dz,0,1); //dvz/dx 
			}
			else 
			{
				staggerFD1order2Ddx(vx2,d1,nx,nz,dx,dz,flag,0,1); //dvx/dx backward_stagger
				staggerFD1order2Ddx(vx2,d2,nx,nz,dx,dz,flag,1,0); //dvx/dz forward_stagger
				staggerFD1order2Ddx(vz2,d3,nx,nz,dx,dz,flag,1,1); //dvz/dz backward_stagger
				staggerFD1order2Ddx(vz2,d4,nx,nz,dx,dz,flag,0,0); //dvz/dx forward_stagger
			}

				for(i=0;i<nz;i++)
					for(j=0;j<nx;j++)
					{
							Txx2[i][j] = dt*(c13[i][j]*(d3[i][j]+phi_vz_z2[i][j])+c11[i][j]*(d1[i][j]+phi_vx_x2[i][j])) + Txx1[i][j];
							Tzz2[i][j] = dt*(c13[i][j]*(d1[i][j]+phi_vx_x2[i][j])+c33[i][j]*(d3[i][j]+phi_vz_z2[i][j])) + Tzz1[i][j];
							Txz2[i][j] = dt*c55[i][j]*(d2[i][j]+d4[i][j]+phi_vx_z2[i][j]+phi_vz_x2[i][j]) + Txz1[i][j];
							if(j<nxpad || i>nz-nzpad || j>nx-nxpad)
							{
								phi_vx_x2[i][j] = bx[i][j]*phi_vx_x1[i][j] + ax1[i][j]*d1[i][j];
								phi_vx_z2[i][j] = bz[i][j]*phi_vx_z1[i][j] + az1[i][j]*d2[i][j];
								phi_vz_z2[i][j] = bz[i][j]*phi_vz_z1[i][j] + az1[i][j]*d3[i][j];
								phi_vz_x2[i][j] = bx[i][j]*phi_vz_x1[i][j] + ax1[i][j]*d4[i][j];
							}
					}
		
				if(k%100==0)
				sf_warning("k=%d",k);

   /*     int N1 = flag;
        for(j=0;j<nx;j++)
        {
            Txz2[N1][j] = 0;
			for(i=1;i<=N1;i++)
				{
					Txz2[N1-i][j] = -Txz2[N1+i][j];
					Tzz2[N1-i+1][j] = -Tzz2[N1+i][j];
				}
        } */

			if(flag==0)
			{
				staggerPS1order2Ddx(Txx2,d1,nx,nz,dx,dz,0,0); //dTxx/dx backward_stagger
				staggerPS1order2Ddx(Tzz2,d2,nx,nz,dx,dz,1,0); //dTzz/dz backward_stagger
				staggerPS1order2Ddx(Txz2,d3,nx,nz,dx,dz,1,1); //dTxz/dz forward_stagger
				staggerPS1order2Ddx(Txz2,d4,nx,nz,dx,dz,0,1); //dTxz/dx forward_stagger
			}
			else if(flag==1)
			{
				RstaggerPS1order2Ddx(Txx2,d1,nx,nz,dx,dz,0,0); //dTxx/dx 
				RstaggerPS1order2Ddx(Tzz2,d2,nx,nz,dx,dz,1,0); //dTzz/dz 
				RstaggerPS1order2Ddx(Txz2,d3,nx,nz,dx,dz,1,0); //dTxz/dz 
				RstaggerPS1order2Ddx(Txz2,d4,nx,nz,dx,dz,0,0); //dTxz/dx 
			}
			else
			{
				staggerFD1order2Ddx(Txx2,d1,nx,nz,dx,dz,flag,0,0); //dTxx/dx backward_stagger
				staggerFD1order2Ddx(Tzz2,d2,nx,nz,dx,dz,flag,1,0); //dTzz/dz backward_stagger
				staggerFD1order2Ddx(Txz2,d3,nx,nz,dx,dz,flag,1,1); //dTxz/dz forward_stagger
				staggerFD1order2Ddx(Txz2,d4,nx,nz,dx,dz,flag,0,1); //dTxz/dx forward_stagger
			}

				for(i=0;i<nz;i++)
					for(j=0;j<nx;j++)
					{
						    vx2[i][j] = dt*(d1[i][j]+d3[i][j]+phi_Txx_x2[i][j]+phi_Txz_z2[i][j]) + vx1[i][j];
	                        vz2[i][j] = dt*(d2[i][j]+d4[i][j]+phi_Tzz_z2[i][j]+phi_Txz_x2[i][j]) + vz1[i][j];
							if(j<nxpad || i>nz-nzpad || j>nx-nxpad)
							{
								phi_Txx_x2[i][j] = bx[i][j]*phi_Txx_x1[i][j] + ax1[i][j]*d1[i][j];
								phi_Txz_z2[i][j] = bz[i][j]*phi_Txz_z1[i][j] + az1[i][j]*d3[i][j];
								phi_Tzz_z2[i][j] = bz[i][j]*phi_Tzz_z1[i][j] + az1[i][j]*d2[i][j];
								phi_Txz_x2[i][j] = bx[i][j]*phi_Txz_x1[i][j] + ax1[i][j]*d4[i][j];
							}
					}

				Txx2[flag][nx/2] += Ricker(k*dt,20,0.04,10);
				Txz2[flag][nx/2] += Ricker(k*dt,20,0.04,10);
				Tzz2[flag][nx/2] += Ricker(k*dt,20,0.04,10); //source term

				/*for(j=0;j<nx;j++)
					fwrite(&vx2[nzpad][j],sizeof(float),1,fp);*/

                for(i=0;i<nz;i++)
                        for(j=0;j<nx;j++)
                        {
                                vx1[i][j] = vx2[i][j];
                                vz1[i][j] = vz2[i][j];
                                Txx1[i][j] = Txx2[i][j];
                                Txz1[i][j] = Txz2[i][j];
                                Tzz1[i][j] = Tzz2[i][j];
								phi_Txx_x1[i][j] = phi_Txx_x2[i][j];
								phi_Txz_x1[i][j] = phi_Txz_x2[i][j];
								phi_Txz_z1[i][j] = phi_Txz_z2[i][j];
								phi_Tzz_z1[i][j] = phi_Tzz_z2[i][j];
								phi_vx_x1[i][j] = phi_vx_x2[i][j];
								phi_vx_z1[i][j] = phi_vx_z2[i][j];
								phi_vz_x1[i][j] = phi_vz_x2[i][j];
								phi_vz_z1[i][j] = phi_vz_z2[i][j];
                        }
		}

		for(j=0;j<nx;j++)
			for(i=0;i<nz;i++)
			{
				sf_floatwrite(&vx1[i][j],1,Fo1);
				sf_floatwrite(&vz1[i][j],1,Fo2);
			}

		t2 = clock();
		timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
		sf_warning("costime %fs",timespent);

   return 0;
}

