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
   if (!sf_getint("flag",&flag)) flag = 1;
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

   float **xvx1,**xvx2,**zvx1,**zvx2,**vx2;
   float **xvz1,**xvz2,**zvz1,**zvz2,**vz2;
   xvx1 = alloc2float(nx,nz);
   xvx2 = alloc2float(nx,nz);
   zvx1 = alloc2float(nx,nz);
   zvx2 = alloc2float(nx,nz);
   vx2 = alloc2float(nx,nz);
   xvz1 = alloc2float(nx,nz);
   xvz2 = alloc2float(nx,nz);
   zvz1 = alloc2float(nx,nz);
   zvz2 = alloc2float(nx,nz);
   vz2 = alloc2float(nx,nz);
   zero2float(xvx1,nx,nz);
   zero2float(xvx2,nx,nz);
   zero2float(zvx1,nx,nz);
   zero2float(zvx2,nx,nz);
   zero2float(vx2,nx,nz);
   zero2float(xvz1,nx,nz);
   zero2float(xvz2,nx,nz);
   zero2float(zvz1,nx,nz);
   zero2float(zvz2,nx,nz);
   zero2float(vz2,nx,nz);

   float **xTxx1,**xTxx2,**zTxx1,**zTxx2,**Txx2;
   float **xTzz1,**xTzz2,**zTzz1,**zTzz2,**Tzz2;
   float **xTxz1,**xTxz2,**zTxz1,**zTxz2,**Txz2;
   xTxx1 = alloc2float(nx,nz);
   xTxx2 = alloc2float(nx,nz);
   zTxx1 = alloc2float(nx,nz);
   zTxx2 = alloc2float(nx,nz);
   Txx2 = alloc2float(nx,nz);
   xTzz1 = alloc2float(nx,nz);
   xTzz2 = alloc2float(nx,nz);
   zTzz1 = alloc2float(nx,nz);
   zTzz2 = alloc2float(nx,nz);
   Tzz2 = alloc2float(nx,nz);
   xTxz1 = alloc2float(nx,nz);
   xTxz2 = alloc2float(nx,nz);
   zTxz1 = alloc2float(nx,nz);
   zTxz2 = alloc2float(nx,nz);
   Txz2 = alloc2float(nx,nz);

   zero2float(xTxx1,nx,nz);
   zero2float(xTxx2,nx,nz);
   zero2float(zTxx1,nx,nz);
   zero2float(zTxx2,nx,nz);
   zero2float(Txx2,nx,nz);
   zero2float(xTzz1,nx,nz);
   zero2float(xTzz2,nx,nz);
   zero2float(zTzz1,nx,nz);
   zero2float(zTzz2,nx,nz);
   zero2float(Tzz2,nx,nz);
   zero2float(xTxz1,nx,nz);
   zero2float(xTxz2,nx,nz);
   zero2float(zTxz1,nx,nz);
   zero2float(zTxz2,nx,nz);
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
				staggerFD1order2Ddx(vx2,d1,nx,nz,dx,dz,5,0,1); //dvx/dx backward_stagger
				staggerFD1order2Ddx(vx2,d2,nx,nz,dx,dz,5,1,0); //dvx/dz forward_stagger
				staggerFD1order2Ddx(vz2,d3,nx,nz,dx,dz,5,1,1); //dvz/dz backward_stagger
				staggerFD1order2Ddx(vz2,d4,nx,nz,dx,dz,5,0,0); //dvz/dx forward_stagger
			}

				for(i=0;i<nz;i++)
					for(j=0;j<nx;j++)
					{
						if(i<nzpad)
							zTxx2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(dt*c13[i][j]*d3[i][j] + exp(-alpha_z[nzpad-i-1]*dt/2)*zTxx1[i][j]);
						else if (i>=nz-nzpad)
							zTxx2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(dt*c13[i][j]*d3[i][j] +  exp(-alpha_z[nzpad+i-nz]*dt/2)*zTxx1[i][j]);
						else
							zTxx2[i][j] = dt*c13[i][j]*d3[i][j] + zTxx1[i][j];

						if(j<nxpad)
							xTxx2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(dt*c11[i][j]*d1[i][j] + exp(-alpha_x[nxpad-j-1]*dt/2)*xTxx1[i][j]);
						else if(j>=nx-nxpad)
							xTxx2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(dt*c11[i][j]*d1[i][j] + exp(- alpha_x[nxpad+j-nx]*dt/2)*xTxx1[i][j]);
						else
							xTxx2[i][j] = dt*c11[i][j]*d1[i][j] + xTxx1[i][j];
						Txx2[i][j] = xTxx2[i][j] + zTxx2[i][j];

						if(i<nzpad)
							zTzz2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(dt*c33[i][j]*d3[i][j] + exp(-alpha_z[nzpad-i-1]*dt/2)*zTzz1[i][j]);
						else if (i>=nz-nzpad)
							zTzz2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(dt*c33[i][j]*d3[i][j] +  exp(-alpha_z[nzpad+i-nz]*dt/2)*zTzz1[i][j]);
						else
							zTzz2[i][j] = dt*c33[i][j]*d3[i][j] + zTzz1[i][j];

						if(j<nxpad)
							xTzz2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(dt*c13[i][j]*d1[i][j] + exp(-alpha_x[nxpad-j-1]*dt/2)*xTzz1[i][j]);
						else if(j>=nx-nxpad)
							xTzz2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(dt*c13[i][j]*d1[i][j] + exp(- alpha_x[nxpad+j-nx]*dt/2)*xTzz1[i][j]);
						else
							xTzz2[i][j] = dt*c13[i][j]*d1[i][j] + xTzz1[i][j];
						Tzz2[i][j] = xTzz2[i][j] + zTzz2[i][j];

						if(i<nzpad)
							zTxz2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(dt*c55[i][j]*d2[i][j] + exp(-alpha_z[nzpad-i-1]*dt/2)*zTxz1[i][j]);
						else if (i>=nz-nzpad)
							zTxz2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(dt*c55[i][j]*d2[i][j] +  exp(-alpha_z[nzpad+i-nz]*dt/2)*zTxz1[i][j]);
						else
							zTxz2[i][j] = dt*c55[i][j]*d2[i][j] + zTxz1[i][j];

						if(j<nxpad)
							xTxz2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(dt*c55[i][j]*d4[i][j] + exp(-alpha_x[nxpad-j-1]*dt/2)*xTxz1[i][j]);
						else if(j>=nx-nxpad)
							xTxz2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(dt*c55[i][j]*d4[i][j] + exp(- alpha_x[nxpad+j-nx]*dt/2)*xTxz1[i][j]);
						else
							xTxz2[i][j] = dt*c55[i][j]*d4[i][j] + xTxz1[i][j];
						Txz2[i][j] = xTxz2[i][j] + zTxz2[i][j];
					}
		
				xTxx2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10);
				xTzz2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10); //source term
				//Txz2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10); //source term
				//Txz2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10); //source term

				if(k%100==0)
				sf_warning("k=%d",k);

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
				staggerFD1order2Ddx(Txx2,d1,nx,nz,dx,dz,5,0,0); //dTxx/dx backward_stagger
				staggerFD1order2Ddx(Tzz2,d2,nx,nz,dx,dz,5,1,0); //dTzz/dz backward_stagger
				staggerFD1order2Ddx(Txz2,d3,nx,nz,dx,dz,5,1,1); //dTxz/dz forward_stagger
				staggerFD1order2Ddx(Txz2,d4,nx,nz,dx,dz,5,0,1); //dTxz/dx forward_stagger
			}

				for(i=0;i<nz;i++)
					for(j=0;j<nx;j++)
					{
						if(j<nxpad)
							xvx2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(dt*d1[i][j] + exp(-alpha_x[nxpad-j-1]*dt/2)*xvx1[i][j]);
						else if(j>=nx-nxpad)
							xvx2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(dt*d1[i][j] + exp(-alpha_x[nxpad+j-nx]*dt/2)*xvx1[i][j]);
						else
		                    xvx2[i][j] = dt*d1[i][j] + xvx1[i][j];
						
						if(i<nzpad)
							zvx2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(dt*d3[i][j] + exp(-alpha_z[nzpad-i-1]*dt/2)*zvx1[i][j]);
						else if (i>=nz-nzpad)
							zvx2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(dt*d3[i][j] + exp(-alpha_z[nzpad+i-nz]*dt/2)*zvx1[i][j]);
						else
						    zvx2[i][j] = dt*d3[i][j] + zvx1[i][j];
                        vx2[i][j] = xvx2[i][j] + zvx2[i][j];

						if(j<nxpad)
							xvz2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(dt*d4[i][j] + exp(-alpha_x[nxpad-j-1]*dt/2)*xvz1[i][j]);
						else if(j>=nx-nxpad)
							xvz2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(dt*d4[i][j] + exp(-alpha_x[nxpad+j-nx]*dt/2)*xvz1[i][j]);
						else
	                        xvz2[i][j] = dt*d4[i][j] + xvz1[i][j];

						if(i<nzpad)
							zvz2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(dt*d2[i][j] + exp(-alpha_z[nzpad-i-1]*dt/2)*zvz1[i][j]);
						else if (i>=nz-nzpad)
							zvz2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(dt*d2[i][j] + exp(-alpha_z[nzpad+i-nz]*dt/2)*zvz1[i][j]);
						else
	                        zvz2[i][j] = dt*d2[i][j] + zvz1[i][j];
                        vz2[i][j] = xvz2[i][j] + zvz2[i][j];
					}

                for(i=0;i<nz;i++)
                        for(j=0;j<nx;j++)
                        {
                                xvx1[i][j] = xvx2[i][j];
                                zvx1[i][j] = zvx2[i][j];
                                xvz1[i][j] = xvz2[i][j];
                                zvz1[i][j] = zvz2[i][j];
                                xTxx1[i][j] = xTxx2[i][j];
                                zTxx1[i][j] = zTxx2[i][j];
                                xTxz1[i][j] = xTxz2[i][j];
                                zTxz1[i][j] = zTxz2[i][j];
                                xTzz1[i][j] = xTzz2[i][j];
                                zTzz1[i][j] = zTzz2[i][j];
                        }
		}

		for(j=0;j<nx;j++)
			for(i=0;i<nz;i++)
			{
				sf_floatwrite(&vx2[i][j],1,Fo1);
				sf_floatwrite(&vz2[i][j],1,Fo2);
			}

		t2 = clock();
		timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
		sf_warning("costime %fs",timespent);

   return 0;
}

