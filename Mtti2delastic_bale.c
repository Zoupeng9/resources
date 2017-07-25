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

   clock_t t1, t2, t3;
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
   float **d11,**d22,**d33,**d44; //derivation variable
   d1 = alloc2float(nx,nz);
   d2 = alloc2float(nx,nz);
   d3 = alloc2float(nx,nz);
   d4 = alloc2float(nx,nz);
   d11 = alloc2float(nx,nz);
   d22 = alloc2float(nx,nz);
   d33 = alloc2float(nx,nz);
   d44 = alloc2float(nx,nz);

   float **Vp0,**Vs0,**epsi,**del,**the; //input model parameter
   Vp0 = alloc2float(nx,nz);
   Vs0 = alloc2float(nx,nz);
   epsi = alloc2float(nx,nz);
   del = alloc2float(nx,nz);
   the = alloc2float(nx,nz);
	   
   /*read model parameter*/
   for(j=0;j<nx;j++)
	   for(i=0;i<nz;i++)
	   {
		   sf_floatread(&Vp0[i][j],1,Fvp0);
           sf_floatread(&Vs0[i][j],1,Fvs0);
           sf_floatread(&epsi[i][j],1,Fepsi);
           sf_floatread(&del[i][j],1,Fdel);
           sf_floatread(&the[i][j],1,Fthe);
	   }

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

   float **c11,**c13,**c15,**c33,**c55,**c35;
   c11 = alloc2float(nx,nz);
   c13 = alloc2float(nx,nz);
   c15 = alloc2float(nx,nz);
   c33 = alloc2float(nx,nz);
   c35 = alloc2float(nx,nz);
   c55 = alloc2float(nx,nz);

   float *c1_11,*c1_13,*c1_15,*c1_33,*c1_35,*c1_55;
   c1_11 = alloc1float(nx*nz);
   c1_13 = alloc1float(nx*nz);
   c1_15 = alloc1float(nx*nz);
   c1_33 = alloc1float(nx*nz);
   c1_35 = alloc1float(nx*nz);
   c1_55 = alloc1float(nx*nz);
   float *vp,*vs,*de,*th,*ep;
   vp = alloc1float(nx*nz);
   vs = alloc1float(nx*nz);
   de = alloc1float(nx*nz);
   th = alloc1float(nx*nz);
   ep = alloc1float(nx*nz);
   for(i=0;i<nz;i++)
	   for(j=0;j<nx;j++)
	   {
		   vp[i*nx+j] = Vp0[i][j];
		   vs[i*nx+j] = Vs0[i][j];
		   de[i*nx+j] = del[i][j];
		   ep[i*nx+j] = epsi[i][j];
		   th[i*nx+j] = the[i][j]*SF_PI/180.0;
	   }

   Thomson2stiffness_2d(vp,vs,ep,de,th,c1_11,c1_13,c1_15,c1_33,c1_35,c1_55,nx,nz);

   for(i=0;i<nz;i++)
       for(j=0;j<nx;j++)
	   {

		   c11[i][j] = c1_11[i*nx+j];
		   c13[i][j] = c1_13[i*nx+j];
		   c15[i][j] = c1_15[i*nx+j];
		   c33[i][j] = c1_33[i*nx+j];
		   c35[i][j] = c1_35[i*nx+j];
		   c55[i][j] = c1_55[i*nx+j];

	//	   c11[i][j] = 10e6;
	//	   c13[i][j] = 2.5e6;
	//	   c15[i][j] = 0.1e6;
	//	   c33[i][j] = 6e6;
	//	   c35[i][j] = 0.4e6;
	//	   c55[i][j] = 4e6;
	   }
   sf_warning("c11=%f",c11[nz-1][nx-1]);
   sf_warning("c13=%f",c13[nz-1][nx-1]);
   sf_warning("c15=%f",c15[nz-1][nx-1]);
   sf_warning("c33=%f",c33[nz-1][nx-1]);
   sf_warning("c35=%f",c35[nz-1][nx-1]);
   sf_warning("c55=%f",c55[nz-1][nx-1]);
   free(c1_11);
   free(c1_13);
   free(c1_15);
   free(c1_33);
   free(c1_35);
   free(c1_55);
   free(vp);
   free(vs);
   free(ep);
   free(th);
   free(de);

       float *kx,*kz;
       kx = sf_floatalloc(nx);
       kz = sf_floatalloc(nz);
       if(nx%2==0)
       {
            for(j=0;j<nx;j++)
                if(j<nx/2)
                    kx[j] = 2.0*SF_PI*j/dx/nx;
                else
                    kx[j] = 2.0*SF_PI*(j-nx)/dx/nx;
        }
        else
        {
           for(j=0;j<nx;j++)
               if(j<=nx/2)
                   kx[j] = 2.0*SF_PI*j/dx/nx;
               else
                   kx[j] = 2.0*SF_PI*(j-nx)/dx/nx;
        }
       if(nz%2==0)
       {
            for(i=0;i<nz;i++)
                if(i<nz/2)
                    kz[i] = 2.0*SF_PI*i/dz/nz;
                else
                    kz[i] = 2.0*SF_PI*(i-nz)/dz/nz;
        }
        else
        {
           for(i=0;i<nz;i++)
               if(i<=nz/2)
                   kz[i] = 2.0*SF_PI*i/dz/nz;
             else
                   kz[i] = 2.0*SF_PI*(i-nz)/dz/nz;
        }

        sf_complex *f_sta_x,*b_sta_x,*f_sta_z,*b_sta_z,*fx_bale,*fz_bale,*fxz_bale;
        f_sta_x = sf_complexalloc(nxz);
        b_sta_x = sf_complexalloc(nxz);
        f_sta_z = sf_complexalloc(nxz);
        b_sta_z = sf_complexalloc(nxz);
        fx_bale = sf_complexalloc(nxz);
        fz_bale = sf_complexalloc(nxz);
        fxz_bale = sf_complexalloc(nxz);

        for(i=0;i<nz;i++)
            for(j=0;j<nx;j++)
            {
                f_sta_x[i*nx+j] = I*kx[j]*cexpf(I*(kx[j]*dx)/2.0);
                b_sta_x[i*nx+j] = I*kx[j]*cexpf(-I*(kx[j]*dx)/2.0);
                f_sta_z[i*nx+j] = I*kz[i]*cexpf(I*(kz[i]*dz)/2.0);
                b_sta_z[i*nx+j] = I*kz[i]*cexpf(-I*(kz[i]*dz)/2.0);
				fx_bale[i*nx+j] = I*kx[j]*cexpf(-I*kz[i]*dz/2.0);
				fz_bale[i*nx+j] = I*kz[i]*cexpf(-I*kx[j]*dx/2.0);
				fxz_bale[i*nx+j] = cexpf(I*(kx[j]*dx+kz[i]*dz)/2.0);
            }

	t3 = clock();
   for(k=0;k<nt;k++)
   {
	   staggerPS1order2Ddx4(zvx2,d1,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,1); //dvx/dx 
	   staggerPS1order2Ddx4(zvx2,d2,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,0); //dvx/dz 
	   staggerPS1order2Ddx4(zvz2,d3,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,1); //dvz/dz 
	   staggerPS1order2Ddx4(zvz2,d4,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,0); //dvz/dx 

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
		   {
			   d11[i][j]=c15[i][j]*d1[i][j];
			   d33[i][j]=c35[i][j]*d3[i][j];
		   }

	   staggerPS1order2Ddx_bale(d11,d11,nx,nz,fx_bale,fz_bale,fxz_bale,0,1); //dvx/dx 
	   staggerPS1order2Ddx_bale(zvx2,d22,nx,nz,fx_bale,fz_bale,fxz_bale,1,0); //dvx/dz 
	   staggerPS1order2Ddx_bale(d33,d33,nx,nz,fx_bale,fz_bale,fxz_bale,1,1); //dvz/dz 
	   staggerPS1order2Ddx_bale(zvz2,d44,nx,nz,fx_bale,fz_bale,fxz_bale,0,0); //dvz/dx 

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
		   {
			/*	   zTxx2[i][j] = dt*(c13[i][j]*d3[i][j]+c15[i][j]*d22[i][j]) + zTxx1[i][j];
				   xTxx2[i][j] = dt*(c11[i][j]*d1[i][j]+c15[i][j]*d44[i][j]) + xTxx1[i][j];
			   Txx2[i][j] = xTxx2[i][j] + zTxx2[i][j];

				   zTzz2[i][j] = dt*(c33[i][j]*d3[i][j]+c35[i][j]*d22[i][j]) + zTzz1[i][j];
				   xTzz2[i][j] = dt*(c13[i][j]*d1[i][j]+c35[i][j]*d44[i][j]) + xTzz1[i][j];
			   Tzz2[i][j] = xTzz2[i][j] + zTzz2[i][j];

				   zTxz2[i][j] = dt*(c55[i][j]*d2[i][j]+d33[i][j]) + zTxz1[i][j];
				   xTxz2[i][j] = dt*(c55[i][j]*d4[i][j]+d11[i][j]) + xTxz1[i][j];
			   Txz2[i][j] = xTxz2[i][j] + zTxz2[i][j];*/
			   zTxx2[i][j] = dt*(c13[i][j]*d3[i][j]+c15[i][j]*d22[i][j]+c11[i][j]*d1[i][j]+c15[i][j]*d44[i][j]) + zTxx1[i][j];
			   zTzz2[i][j] = dt*(c33[i][j]*d3[i][j]+c35[i][j]*d22[i][j]+c13[i][j]*d1[i][j]+c35[i][j]*d44[i][j]) + zTzz1[i][j];
			   zTxz2[i][j] = dt*(c55[i][j]*d2[i][j]+d33[i][j]+c55[i][j]*d4[i][j]+d11[i][j]) + zTxz1[i][j];
		   }
		
	  // xTxx2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10);
	  // xTzz2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10); //source term
	   zTxx2[nz/2][nx/2] += Ricker(k*dt,30,0.04,10);
	   zTzz2[nz/2][nx/2] += Ricker(k*dt,30,0.04,10); //source term
	   //Txz2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10); //source term
	   ////Txz2[nz/2][nx/2] += Ricker(k*dt,20,0.04,10); //source term

	   if(k%100==0)
		   sf_warning("k=%d",k);

	   staggerPS1order2Ddx4(zTxx2,d1,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,0); //dTxx/dx 
	   staggerPS1order2Ddx4(zTzz2,d2,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,0); //dTzz/dz 
	   staggerPS1order2Ddx4(zTxz2,d3,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,1,1); //dTxz/dz 
	   staggerPS1order2Ddx4(zTxz2,d4,nx,nz,f_sta_x,b_sta_x,f_sta_z,b_sta_z,0,1); //dTxz/dx 


	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
		   {
		    /*       xvx2[i][j] = dt*d1[i][j] + xvx1[i][j];
				   zvx2[i][j] = dt*d3[i][j] + zvx1[i][j];
			   vx2[i][j] = xvx2[i][j] + zvx2[i][j];

				   xvz2[i][j] = dt*d4[i][j] + xvz1[i][j];
				   zvz2[i][j] = dt*d2[i][j] + zvz1[i][j];
			   vz2[i][j] = xvz2[i][j] + zvz2[i][j];*/
			   zvx2[i][j] = dt*(d3[i][j]+d1[i][j]) + zvx1[i][j];
			   zvz2[i][j] = dt*(d2[i][j]+d4[i][j]) + zvz1[i][j];
		   }

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
		   {
			//   xvx1[i][j] = xvx2[i][j];
			   zvx1[i][j] = zvx2[i][j];
			//   xvz1[i][j] = xvz2[i][j];
			   zvz1[i][j] = zvz2[i][j];
			//   xTxx1[i][j] = xTxx2[i][j];
			   zTxx1[i][j] = zTxx2[i][j];
			//   xTxz1[i][j] = xTxz2[i][j];
			   zTxz1[i][j] = zTxz2[i][j];
			//   xTzz1[i][j] = xTzz2[i][j];
			   zTzz1[i][j] = zTzz2[i][j];
		   }

		t2 = clock();
   }

   for(j=0;j<nx;j++)
	   for(i=0;i<nz;i++)
	   {
		   sf_floatwrite(&zvx2[i][j],1,Fo1);
		   sf_floatwrite(&zvz2[i][j],1,Fo2);
	   }

   timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
   sf_warning("costime %fs",timespent);
   timespent = (float)(t2 - t3)/CLOCKS_PER_SEC/nt*1000;
   sf_warning("each step costime %fms",timespent);

   return 0;
}

