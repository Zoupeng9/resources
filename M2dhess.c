#include <rsf.h>
#include <assert.h>
#include <complex.h>
#include <fftw3.h>
#include <time.h>
#include <alloc.h>
#include <stdio.h>
#include "puthead.h"

/* prepared head files by myself */
#include "_cjb.h"

/*****************************************************************************************/

int main(int argc, char* argv[])
{
   sf_init(argc,argv);

   int i,j,k;
   
   /* setup I/O files */
	sf_file Fvp0,Fvs0;
	sf_file FQp,FQs,FVp0,FVs0,Fdensity;

	Fvp0 = sf_input("in");  
	Fvs0 = sf_input("vs0"); 

	FQp = sf_output("Qp"); //Qp
	FQs = sf_output("Qs"); //Qs
	FVp0 = sf_output("Vp0"); //Vp0
	FVs0 = sf_output("Vs0"); //Vs0
	Fdensity = sf_output("density"); //density


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

   dx = dx/1000.0;
   dz = dz/1000.0;
   puthead2(FQp,nz,nx,dz,fx/1000.0,dx,fx/1000.0);
   puthead2(FQs,nz,nx,dz,fx/1000.0,dx,fx/1000.0);
   puthead2(FVp0,nz,nx,dz,fx/1000.0,dx,fx/1000.0);
   puthead2(FVs0,nz,nx,dz,fx/1000.0,dx,fx/1000.0);
   puthead2(Fdensity,nz,nx,dz,fx/1000.0,dx,fx/1000.0);

   float **Qp,**Qs,**Vp0,**Vs0,**density; //input model parameter
   Qp = alloc2float(nx,nz);
   Qs = alloc2float(nx,nz);
   Vp0 = alloc2float(nx,nz);
   Vs0 = alloc2float(nx,nz);
   density = alloc2float(nx,nz);
	   
   /*read model parameter*/
   for(j=0;j<nx;j++)
	   for(i=0;i<nz;i++)
	   {
		   sf_floatread(&Vp0[i][j],1,Fvp0);
           sf_floatread(&Vs0[i][j],1,Fvs0);
		   Qp[i][j] = Vp0[i][j]/40.0;
		   Qs[i][j] = Vs0[i][j]/40.0;
		   //Qp[i][j] = 10000000000.0;
		   //Qs[i][j] = 10000000000.0;
		   density[i][j] = 2300.0;
		   sf_floatwrite(&Qp[i][j],1,FQp);
           sf_floatwrite(&Qs[i][j],1,FQs);
		   sf_floatwrite(&Vp0[i][j],1,FVp0);
           sf_floatwrite(&Vs0[i][j],1,FVs0);
           sf_floatwrite(&density[i][j],1,Fdensity);
	   }
   return 0;
}

