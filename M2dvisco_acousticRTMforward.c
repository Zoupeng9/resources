/* 2-D two-components visco-acoustic wavefield extrapolation using 
   pseudo-spectral method with velocity-stress wave equation in 
   heterogeneous media

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

   clock_t t1, t2;
   float   timespent;

   t1=clock();

   int i,j,k;
   
   int nt,flag;
   float dt,fw,f0;
   if (!sf_getint("nt",&nt)) nt = 500;
   if (!sf_getint("flag",&flag)) flag = 1;
   if (!sf_getfloat("dt",&dt)) dt = 0.001;
   if (!sf_getfloat("fw",&fw)) fw = 80.0;
   if (!sf_getfloat("f0",&f0)) f0 = 30.0;
   if(flag==0)
	   sf_warning("Using staggered pseudospectral method");
   else
	   sf_warning("Using staggered finite-difference method");
   sf_warning("nt=%d dt=%f fw=%f f0=%f",nt,dt,fw,f0);

   int nxpad,nzpad;
   if (!sf_getint("nxpad",&nxpad)) nxpad = 20;
   if (!sf_getint("nzpad",&nzpad)) nzpad = 20;
   sf_warning("nxpad=%d  nzpad=%d",nxpad,nzpad);

   sf_warning("read Vp/quality-factor model parameters"); 

   /* setup I/O files */
	sf_file Fi1,Fi2;
	sf_file Fo;

	Fi1 = sf_input("Qp"); /*qualiy factor*/
	Fi2 = sf_input("Vp"); /*Vp*/

	Fo = sf_output("out");   /*pressure field */

   /* Read/Write axes */
   sf_axis ax,az;
   int nxv, nzv;
   float dx,dz,fx,fz;
   az = sf_iaxa(Fi1,1); nzv = sf_n(az); dz = sf_d(az);
   ax = sf_iaxa(Fi1,2); nxv = sf_n(ax); dx = sf_d(ax);
   fz = sf_o(az)*1000.0;
   fx = sf_o(ax)*1000.0;
   dx=dz=10;

   sf_warning("nx=%d nz=%d",nxv,nzv);
   sf_warning("dx=%f dz=%f",dx,dz);

   /* wave modeling space */
   int nx, nz, nxz;
   nx=nxv;
   nz=nzv;
   nxz=nx*nz;

   puthead3(Fo,nz,nx,nt,dz,dx,dt,0,0,0);

   float **vx1,**vx2,**vz1,**vz2;
   vx1 = alloc2float(nx,nz);
   vx2 = alloc2float(nx,nz);
   vz1 = alloc2float(nx,nz);
   vz2 = alloc2float(nx,nz);
   zero2float(vx1,nx,nz);
   zero2float(vx2,nx,nz);
   zero2float(vz1,nx,nz);
   zero2float(vz2,nx,nz);

   float **xp1,**xp2,**zp1,**zp2,**p2;
   float **xrp1,**xrp2,**zrp1,**zrp2;
   p2 = alloc2float(nx,nz);
   xp1 = alloc2float(nx,nz);
   xp2 = alloc2float(nx,nz);
   zp1 = alloc2float(nx,nz);
   zp2 = alloc2float(nx,nz);
   xrp1 = alloc2float(nx,nz);
   xrp2 = alloc2float(nx,nz);
   zrp1 = alloc2float(nx,nz);
   zrp2 = alloc2float(nx,nz);
   zero2float(p2,nx,nz);
   zero2float(xp1,nx,nz);
   zero2float(xp2,nx,nz);
   zero2float(zp1,nx,nz);
   zero2float(zp2,nx,nz);
   zero2float(xrp1,nx,nz);
   zero2float(xrp2,nx,nz);
   zero2float(zrp1,nx,nz);
   zero2float(zrp2,nx,nz);

   float **record;
   record = alloc2float(nx,nt);

   float **dpx,**dpz,**dvx,**dvz; //derivation
   dpx = alloc2float(nx,nz);
   dpz = alloc2float(nx,nz);
   dvx = alloc2float(nx,nz);
   dvz = alloc2float(nx,nz);

   float **tc,**te,**bulk,**Qp;
   tc = alloc2float(nx,nz);
   te = alloc2float(nx,nz);
   bulk = alloc2float(nx,nz);
   Qp = alloc2float(nx,nz);
	   
   /*read model parameter*/
   for(j=0;j<nx;j++)
	   for(i=0;i<nz;i++)
	   {
		   sf_floatread(&Qp[i][j],1,Fi1);
           sf_floatread(&bulk[i][j],1,Fi2);//vp
		   bulk[i][j] = bulk[i][j]*bulk[i][j];
	   }

   float *alpha_x,*alpha_z;
   float alpha_max,Vmax;
   alpha_x = alloc1float(nxpad);
   alpha_z = alloc1float(nzpad);
   alpha_max = 4.0;
   Vmax = 5000.0;

   for(i=0;i<nxpad;i++)
	   alpha_x[i] = alpha_max*Vmax/dx*pow(i*1.0/nxpad,4);
   for(i=0;i<nzpad;i++)
       alpha_z[i] = alpha_max*Vmax/dz*pow(i*1.0/nzpad,4);

   for(i=0;i<nz;i++)
	   for(j=0;j<nx;j++)
	   {
		   tc[i][j] = ( sqrt(1.0+1.0/Qp[i][j]/Qp[i][j])-1.0/Qp[i][j])/fw;
		   te[i][j] = 1.0/fw/fw/tc[i][j];
	   }

        for(k=0;k<nt;k++)
        {
			if(flag==0)
			{
				staggerPS1order2Ddx(vx2,dvx,nx,nz,dx,dz,0,1); //dvx/dx forward_stagger
				staggerPS1order2Ddx(vz2,dvz,nx,nz,dx,dz,1,1); //dvz/dz forward_stagger
			}
			else
			{
				staggerFD1order2Ddx(vx2,dvx,nx,nz,dx,dz,5,0,1);//dvx/dx forward_stagger
				staggerFD1order2Ddx(vz2,dvz,nx,nz,dx,dz,5,1,1); //dvz/dz forward_stagger
			}
                for(i=0;i<nz;i++)
                        for(j=0;j<nx;j++)
                        {
								if(i<=nzpad)
									zrp2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(-dt/tc[i][j]*(zrp1[i][j] + (te[i][j]/tc[i][j]-1.0)*dvz[i][j]*bulk[i][j]) + exp(-alpha_z[nzpad-i-1]*dt/2)*zrp1[i][j]);
								else if (i>=nz-nzpad)
									zrp2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(-dt/tc[i][j]*(zrp1[i][j] + (te[i][j]/tc[i][j]-1.0)*dvz[i][j]*bulk[i][j]) + exp(-alpha_z[nzpad+i-nz]*dt/2)*zrp1[i][j]);
								else
									zrp2[i][j] = -dt/tc[i][j]*(zrp1[i][j] + (te[i][j]/tc[i][j]-1.0)*dvz[i][j]*bulk[i][j]) + zrp1[i][j];

								if(j<=nxpad)
									xrp2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(-dt/tc[i][j]*(xrp1[i][j] + (te[i][j]/tc[i][j]-1.0)*dvx[i][j]*bulk[i][j]) + exp(-alpha_x[nxpad-j-1]*dt/2)*xrp1[i][j]);
								else if(j>=nx-nxpad)
									xrp2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(-dt/tc[i][j]*(xrp1[i][j] + (te[i][j]/tc[i][j]-1.0)*dvx[i][j]*bulk[i][j]) + exp(- alpha_x[nxpad+j-nx]*dt/2)*xrp1[i][j]);
								else
									xrp2[i][j] = -dt/tc[i][j]*(xrp1[i][j] + (te[i][j]/tc[i][j]-1.0)*dvx[i][j]*bulk[i][j]) + xrp1[i][j];


								if(i<=nzpad)
									zp2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(dt*(te[i][j]*bulk[i][j]/tc[i][j]*dvz[i][j] + zrp2[i][j]) + exp(-alpha_z[nzpad-i-1]*dt/2)*zp1[i][j]);
								else if (i>=nz-nzpad)
									zp2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(dt*(te[i][j]*bulk[i][j]/tc[i][j]*dvz[i][j] + zrp2[i][j]) + exp(-alpha_z[nzpad+i-nz]*dt/2)*zp1[i][j]);
								else
									zp2[i][j] = dt*(te[i][j]*bulk[i][j]/tc[i][j]*dvz[i][j] + zrp2[i][j]) + zp1[i][j];

								if(j<=nxpad)
									xp2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(dt*(te[i][j]*bulk[i][j]/tc[i][j]*dvx[i][j] + xrp2[i][j]) + exp(-alpha_x[nxpad-j-1]*dt/2)*xp1[i][j]);
								else if(j>=nx-nxpad)
									xp2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(dt*(te[i][j]*bulk[i][j]/tc[i][j]*dvx[i][j] + xrp2[i][j]) + exp(- alpha_x[nxpad+j-nx]*dt/2)*xp1[i][j]);
								else
                                xp2[i][j] = dt*(te[i][j]*bulk[i][j]/tc[i][j]*dvx[i][j] + xrp2[i][j]) + xp1[i][j];
								p2[i][j] = xp2[i][j] + zp2[i][j];
                        }

                p2[nzpad+1][nx*2/4] += Ricker(k*dt,f0,0.04,10);

				if(flag==0)
				{
					staggerPS1order2Ddx(p2,dpx,nx,nz,dx,dz,0,0); //dp/dx backward_stagger
					staggerPS1order2Ddx(p2,dpz,nx,nz,dx,dz,1,0); //dp/dz backward_stagger
				}
				else
				{
					staggerFD1order2Ddx(p2,dpx,nx,nz,dx,dz,5,0,0); //dp/dx backward_stagger
					staggerFD1order2Ddx(p2,dpz,nx,nz,dx,dz,5,1,0); //dp/dz backward_stagger
				}
                for(i=0;i<nz;i++)
                        for(j=0;j<nx;j++)
                        {
							if(i<=nzpad)
                                vz2[i][j] = exp(-alpha_z[nzpad-i-1]*dt/2)*(dt*dpz[i][j] + exp(-alpha_z[nzpad-i-1]*dt/2)*vz1[i][j]);
							else if(i>=nz-nzpad)
								vz2[i][j] = exp(-alpha_z[nzpad+i-nz]*dt/2)*(dt*dpz[i][j] + exp(-alpha_z[nzpad+i-nz]*dt/2)*vz1[i][j]);
							else
								vz2[i][j] = dt*dpz[i][j] + vz1[i][j];

							if(j<=nxpad)
								vx2[i][j] = exp(-alpha_x[nxpad-j-1]*dt/2)*(dt*dpx[i][j] + exp(-alpha_x[nxpad-j-1]*dt/2)*vx1[i][j]);
							else if(j>=nx-nxpad)
								vx2[i][j] = exp(-alpha_x[nxpad+j-nx]*dt/2)*(dt*dpx[i][j] + exp(- alpha_x[nxpad+j-nx]*dt/2)*vx1[i][j]);
							else
                                vx2[i][j] = dt*dpx[i][j] + vx1[i][j];
                        }

                for(i=0;i<nz;i++)
                        for(j=0;j<nx;j++)
                        {
                                vx1[i][j] = vx2[i][j];
                                vz1[i][j] = vz2[i][j];
                                xrp1[i][j] = xrp2[i][j];
                                zrp1[i][j] = zrp2[i][j];
                                xp1[i][j] = xp2[i][j];
                                zp1[i][j] = zp2[i][j];
                        }

				for(j=0;j<nx;j++)
					record[k][j] = p2[nzpad+1][j]; //surface record

		for(j=0;j<nx;j++)
			for(i=0;i<nz;i++)
				sf_floatwrite(&p2[i][j],1,Fo);

		}

		t2 = clock();
		timespent = (float)(t2 - t1)/CLOCKS_PER_SEC;
		sf_warning("costime %fs",timespent);

   return 0;
}

