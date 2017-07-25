/* using staggered pseudo-spectral method to caculate derivation */
/*
  Copyright (C) 2015 Tongji University (Peng Zou) 
 
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
#include "_cjb.h"

#include <fftw3.h>
#include <omp.h>

//staggered coefficent
float  c[5][5]={{1.0},
                {1.125,-0.04166667},
                {1.171875,-0.06510416,0.0046875},
                {1.196289,-0.0797526,0.009570313,-0.0006975447},
                {1.211243,-0.08972168,0.013842773,-0.00176566,0.0001186795}};

/*****************************************************************************************/
void staggerPS1order2Ddx(float **p,float **dp,int nx,int nz,float dx,float dz,int flag1,int flag2)
/*< staggerPS1order2Ddx caculate 1order derivation of p using staggered PS method 
             if flag1==0,caculate dp/dx else caculate dp/dz  
               if flag2==0,forward stagger else backward >*/
{
	
#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;
	   
	   int i,j,nxz;
	   nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   xin[i*nx+j] = sf_cmplx(p[i][j], 0.0);

       fftwf_execute(xp);

	   if(flag1 == 0) //dp/dx
	   {
		   sf_complex *f_sta_x,*b_sta_x;
		   f_sta_x = sf_complexalloc(nx);
		   b_sta_x = sf_complexalloc(nx);

		   float *kx;
		   kx = sf_floatalloc(nx);
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

		   for(j=0;j<nx;j++)
		   {
			   f_sta_x[j] = I*kx[j]*cexpf(I*kx[j]*dx/2.0);
			   b_sta_x[j] = I*kx[j]*cexpf(-I*kx[j]*dx/2.0);
		   }

		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = f_sta_x[j]*xout[i*nx+j];
			   free(f_sta_x);
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = b_sta_x[j]*xout[i*nx+j];
			   free(b_sta_x);
		   }
	   }
	   else // dp/dz
	   {
		   sf_complex *f_sta_z,*b_sta_z;
		   f_sta_z = sf_complexalloc(nz);
		   b_sta_z = sf_complexalloc(nz);

		   float *kz;
		   kz = sf_floatalloc(nz);
		   if(nz%2==0)
		   {
				for(j=0;j<nz;j++)     
					if(j<nz/2)
						kz[j] = 2.0*SF_PI*j/dz/nz;
					else
						kz[j] = 2.0*SF_PI*(j-nz)/dz/nz;
		   }
		   else
		   {
			   for(j=0;j<nz;j++)
				   if(j<=nz/2)
					   kz[j] = 2.0*SF_PI*j/dz/nz;
				   else
					   kz[j] = 2.0*SF_PI*(j-nz)/dz/nz;
		   }

		   for(j=0;j<nz;j++)
		   {
			   f_sta_z[j] = I*kz[j]*cexpf(I*kz[j]*dz/2.0);
			   b_sta_z[j] = I*kz[j]*cexpf(-I*kz[j]*dz/2.0);
		   }

		   if(flag2 == 0) // forward stagger
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = f_sta_z[i]*xout[i*nx+j];
			   free(f_sta_z);
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = b_sta_z[i]*xout[i*nx+j];
			   free(b_sta_z);
		   }
	   }

       fftwf_execute(xpi);

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   dp[i][j] = creal(xout[i*nx+j])/nxz;
	   
       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif 
}

void staggerFD1order2Ddx(float **p,float **dp,int nx,int nz,float dx,float dz,int N,int flag1,int flag2)
/*< staggerFD1order2Ddx caculate 1order derivation of p using FD method 
             if flag1==0,caculate dp/dx else caculate dp/dz 
               if flag2==0,forward stagger else backward 
				 N is the order of the difference accurate >*/
{
	int i,j,n;
	float temp;
	for(i=N;i<nz-N;i++)
		for(j=N;j<nx-N;j++)
		{
			temp = 0.0;
			if(flag1==0)
			{
				if(flag2==0)
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i][j+n+1] - p[i][j-n])/dx;
				else
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i][j+n] - p[i][j-n-1])/dx;
			}
			else
			{
				if(flag2==0)
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i+n+1][j] - p[i-n][j])/dz;
				else
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i+n][j] - p[i-n-1][j])/dz;
			}

			dp[i][j] = temp;
		}
}

void staggerFD1order2Ddx1(float **p,float **dp,int nx,int nz,float dx,float dz,int N,int flag1,int flag2)
/*< staggerFD1order2Ddx caculate 1order derivation of p using FD method 
             if flag1==0,caculate dp/dx else caculate dp/dz 
               if flag2==0,forward stagger else backward 
				 N is the order of the difference accurate >*/
{
	int i,j,n;
	float temp;
	for(i=N;i<nx-N;i++)
		for(j=N;j<nz-N;j++)
		{
			temp = 0.0;
			if(flag1==0)
			{
				if(flag2==0)
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i+n+1][j] - p[i-n][j])/dx;
				else
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i+n][j] - p[i-n-1][j])/dx;
			}
			else
			{
				if(flag2==0)
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i][j+n+1] - p[i][j-n])/dz;
				else
					for(n=0;n<N;n++)
						temp += c[N-1][n]*(p[i][j+n] - p[i][j-n-1])/dz;
			}

			dp[i][j] = temp;
		}
}

/*****************************************************************************************/
void RstaggerPS1order2Ddx(float **p,float **dp,int nx,int nz,float dx,float dz,int flag1,int flag2)
/*< RstaggerPS1order2Ddx caculate 1order derivation of p using rotated-staggered PS method 
            if flag1==0,caculate dp/dx else caculate dp/dz
             if flag2==0,forward stagger else backward >*/
{
	
#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;
	   
	   int i,j,nxz;
	   nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

//       fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

//     fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   xin[i*nx+j] = sf_cmplx(p[i][j], 0.0);

       fftwf_execute(xp);

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

	   sf_complex f_sta_x,b_sta_x,f_sta_z,b_sta_z;


	   if(flag1 == 0) //dp/dx
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
				   {
					   f_sta_x = I*kx[j]*cexpf(I*(kx[j]*dx-kz[i]*dz)/2.0);
					   xin[i*nx+j] = f_sta_x*xout[i*nx+j];
				   }
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
				   {
					   b_sta_x = I*kx[j]*cexpf(-I*(kx[j]*dx-kz[i]*dz)/2.0);
					   xin[i*nx+j] = b_sta_x*xout[i*nx+j];
				   }
		   }
	   }
	   else // dp/dz
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
				   {
					   f_sta_z = I*kz[i]*cexpf(I*(kx[j]*dx-kz[i]*dz)/2.0);
					   xin[i*nx+j] = f_sta_z*xout[i*nx+j];
				   }
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
				   {
					   b_sta_z = I*kz[i]*cexpf(-I*(kx[j]*dx-kz[i]*dz)/2.0);
					   xin[i*nx+j] = b_sta_z*xout[i*nx+j];
				   }
		   }
	   }

       fftwf_execute(xpi);

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   dp[i][j] = creal(xout[i*nx+j])/nxz;
	   
       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif 
}

void RstaggerPS1order3Ddx(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1,int flag2)
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

       fftwf_plan_with_nthreads(30);
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       fftwf_plan_with_nthreads(30);
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

	   i = 0;
       if(flag1 == 0) //dp/dx
       {
           if(flag2 == 0) //forward stagger
               for(iy=0;iy<ny;iy++)
				   for(ix=0;ix<nx;ix++)
					   for(iz=0;iz<nz;iz++)
					   {
						 //  if(ix>nx/2-1&&ix<nx/2+1)
						//	   xin[i] = I*kx[i]*cexpf(I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*xout[i];
						//   else
							   xin[i] = I*kx[i]/*cexpf(I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*/*xout[i];
							   i++;
					   }
           else
               for(iy=0;iy<ny;iy++)
				   for(ix=0;ix<nx;ix++)
					   for(iz=0;iz<nz;iz++)
					   {
							   xin[i] = I*kx[i]/*cexpf(-I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*/*xout[i];
							   i++;
					   }
       }
       else if(flag1==1) //dp/dy
       {
           if(flag2 == 0) //forward stagger
               for(iy=0;iy<ny;iy++)
				   for(ix=0;ix<nx;ix++)
					   for(iz=0;iz<nz;iz++)
					   {
							   xin[i] = I*ky[i]/*cexpf(I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*/*xout[i];
							   i++;
					   }
		   else
               for(iy=0;iy<ny;iy++)
				   for(ix=0;ix<nx;ix++)
					   for(iz=0;iz<nz;iz++)
					   {
							   xin[i] = I*ky[i]/*cexpf(-I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*/*xout[i];
							   i++;
					   }
       }
       else // dp/dz
       {
           if(flag2 == 0) //forward stagger
               for(iy=0;iy<ny;iy++)
				   for(ix=0;ix<nx;ix++)
					   for(iz=0;iz<nz;iz++)
					   {
							   xin[i] = I*kz[i]/*cexpf(I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*/*xout[i];
							   i++;
					   }
           else
               for(iy=0;iy<ny;iy++)
				   for(ix=0;ix<nx;ix++)
					   for(iz=0;iz<nz;iz++)
					   {
							   xin[i] = I*kz[i]/*cexpf(-I*(kx[i]*dx+ky[i]*dy-kz[i]*dz)/2.0)*/*xout[i];
							   i++;
					   }
       }

       fftwf_execute(xpi);

       for(i=0;i<nxyz;i++)
              dp[i] = creal(xout[i])/nxyz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
//	   fftwf_cleanup_threads();
       free(xin);
       free(xout);
#endif
}

void staggerPS1order3Ddx(float *p,float *dp,int nx,int ny,int nz,float dx,float dy,float dz,float*kx,
                          float*ky,float*kz,int flag1,int flag2)
/*< RstaggerPS1order3Ddx caculate 1order derivation of p using rotated-staggered PS method 
             if flag1==0,caculate dp/dx flag==1 dp/dy,else caculate dp/dz 
              if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,nxyz;
       nxyz = nx*ny*nz;

       xin=sf_complexalloc(nxyz);
       xout=sf_complexalloc(nxyz);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       if(flag1 == 0) //dp/dx
       {
           if(flag2 == 0) //forward stagger
               for(i=0;i<nxyz;i++)
						   xin[i] = I*kx[i]*xout[i]*cexpf(I*kx[i]*dx/2.0);
           else
               for(i=0;i<nxyz;i++)
						   xin[i] = I*kx[i]*xout[i]*cexpf(-I*kx[i]*dx/2.0);
       }
       else if(flag1==1) //dp/dy
       {
           if(flag2 == 0) //forward stagger
               for(i=0;i<nxyz;i++)
						   xin[i] = I*ky[i]*xout[i]*cexpf(I*ky[i]*dy/2.0);
           else
               for(i=0;i<nxyz;i++)
						   xin[i] = I*ky[i]*xout[i]*cexpf(-I*ky[i]*dy/2.0);
       }
       else // dp/dz
       {
           if(flag2 == 0) //forward stagger
               for(i=0;i<nxyz;i++)
						   xin[i] = I*kz[i]*xout[i]*cexpf(I*kz[i]*dz/2.0);
           else
               for(i=0;i<nxyz;i++)
						   xin[i] = I*kz[i]*xout[i]*cexpf(-I*kz[i]*dz/2.0);
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

void staggerPS1order2Ddx1(float *p,float *dp,int nx,int nz,float dx,float dz,int flag1,int flag2)

/*< staggerPS1order2Ddx caculate 1order derivation of p using staggered PS method 
              if flag1==0,caculate dp/dx else caculate dp/dz 
               if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,j,nxz;
       nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<nxz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       if(flag1 == 0) //dp/dx
       {
           sf_complex *f_sta_x,*b_sta_x;
           f_sta_x = sf_complexalloc(nx);
           b_sta_x = sf_complexalloc(nx);

           float *kx;
           kx = sf_floatalloc(nx);
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

           for(j=0;j<nx;j++)
           {
               f_sta_x[j] = I*kx[j]*cexpf(I*kx[j]*dx/2.0);
               b_sta_x[j] = I*kx[j]*cexpf(-I*kx[j]*dx/2.0);
           }

           if(flag2 == 0) //forward stagger
           {
               for(i=0;i<nz;i++)
                   for(j=0;j<nx;j++)
                       xin[i*nx+j] = f_sta_x[j]*xout[i*nx+j];
               free(f_sta_x);
           }
           else
           {
               for(i=0;i<nz;i++)
                   for(j=0;j<nx;j++)
                       xin[i*nx+j] = b_sta_x[j]*xout[i*nx+j];
               free(b_sta_x);
           }
       }
       else // dp/dz
       {
           sf_complex *f_sta_z,*b_sta_z;
           f_sta_z = sf_complexalloc(nz);
           b_sta_z = sf_complexalloc(nz);

           float *kz;
           kz = sf_floatalloc(nz);
           if(nz%2==0)
           {
                for(j=0;j<nz;j++)
                    if(j<nz/2)
                        kz[j] = 2.0*SF_PI*j/dz/nz;
                    else
                        kz[j] = 2.0*SF_PI*(j-nz)/dz/nz;
           }
           else
           {
               for(j=0;j<nz;j++)
                   if(j<=nz/2)
                       kz[j] = 2.0*SF_PI*j/dz/nz;
                   else
                       kz[j] = 2.0*SF_PI*(j-nz)/dz/nz;
           }

           for(j=0;j<nz;j++)
           {
               f_sta_z[j] = I*kz[j]*cexpf(I*kz[j]*dz/2.0);
               b_sta_z[j] = I*kz[j]*cexpf(-I*kz[j]*dz/2.0);
           }

           if(flag2 == 0) // forward stagger
           {
               for(i=0;i<nz;i++)
                   for(j=0;j<nx;j++)
                       xin[i*nx+j] = f_sta_z[i]*xout[i*nx+j];
               free(f_sta_z);
           }
           else
           {
               for(i=0;i<nz;i++)
                   for(j=0;j<nx;j++)
                       xin[i*nx+j] = b_sta_z[i]*xout[i*nx+j];
               free(b_sta_z);
           }
       }

       fftwf_execute(xpi);

       for(i=0;i<nxz;i++)
               dp[i] = creal(xout[i])/nxz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif
}

void staggerPS1order2Ddx2(float *p,float *dp,int nx,int nz,float dx,float dz,float*kx,float*kz,int flag1,int flag2)
/*< staggerPS1order2Ddx caculate 1order derivation of p using staggered PS method 
             if flag1==0,caculate dp/dx else caculate dp/dz 
               if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,nxz;
       nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<nxz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       if(flag1 == 0) //dp/dx
       {
           if(flag2 == 0) //forward stagger
               for(i=0;i<nxz;i++)
                       xin[i] = I*kx[i]*xout[i]*cexpf(I*kx[i]*dx/2.0);
           else
               for(i=0;i<nxz;i++)
                       xin[i] = I*kx[i]*xout[i]*cexpf(-I*kx[i]*dx/2.0);
       }
       else // dp/dz
       {
           if(flag2 == 0) // forward stagger
               for(i=0;i<nxz;i++)
                       xin[i] = I*kz[i]*xout[i]*cexpf(I*kz[i]*dz/2.0);
           else
               for(i=0;i<nxz;i++)
                       xin[i] = I*kz[i]*xout[i]*cexpf(-I*kz[i]*dz/2.0);
       }

       fftwf_execute(xpi);

       for(i=0;i<nxz;i++)
               dp[i] = creal(xout[i])/nxz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif
}

/*****************************************************************************************/
void RstaggerPS1order2Ddx1(float *p,float *dp,int nx,int nz,float dx,float dz,float *kx,float *kz,int flag1,int flag2)
/*< RstaggerPS1order2Ddx caculate 1order derivation of p using rotated-staggered PS method 
            if flag1==0,caculate dp/dx else caculate dp/dz
             if flag2==0,forward stagger else backward >*/
{
	
#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;
	   
	   int i,j,nxz;
	   nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

       //fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

     //  fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			  xin[i*nx+j] = sf_cmplx(p[i*nx+j], 0.0);
			   

       fftwf_execute(xp);

	   sf_complex f_sta_x,b_sta_x,f_sta_z,b_sta_z;


	   if(flag1 == 0) //dp/dx
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nxz;i++)
			   {
				   f_sta_x = I*kx[i]*cexpf(I*(kx[i]*dx-kz[i]*dz)/2.0);
				   xin[i] = f_sta_x*xout[i];
			   }
		   }
		   else
		   {
			   for(i=0;i<nxz;i++)
			   {
				   b_sta_x = I*kx[i]*cexpf(-I*(kx[i]*dx-kz[i]*dz)/2.0);
				   xin[i] = b_sta_x*xout[i];
			   }
		   }
	   }
	   else // dp/dz
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nxz;i++)
			   {
				   f_sta_z = I*kz[i]*cexpf(I*(kx[i]*dx-kz[i]*dz)/2.0);
				   xin[i] = f_sta_z*xout[i];
			   }
		   }
		   else
		   {
			   for(i=0;i<nxz;i++)
			   {
				   b_sta_z = I*kz[i]*cexpf(-I*(kx[i]*dx-kz[i]*dz)/2.0);
				   xin[i] = b_sta_z*xout[i];
			   }
		   }
	   }

       fftwf_execute(xpi);

	   for(i=0;i<nxz;i++)
			   dp[i] = creal(xout[i])/nxz;
	   
       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif 
}


void staggerPS1order2Ddx_bale(float **p,float **dp,int nx,int nz,sf_complex *fx_bale,sf_complex *fz_bale,sf_complex *fxz_bale,int flag1,int flag2)
/*< RstaggerPS1order2Ddx caculate 1order derivation of p using rotated-staggered PS method 
            if flag1==0,caculate dp/dx else caculate dp/dz
             if flag2==0,forward stagger else backward >*/
{
	
#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;
	   
	   int i,j,nxz;
	   nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

     //  fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

     //  fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   xin[i*nx+j] = sf_cmplx(p[i][j], 0.0);

       fftwf_execute(xp);

	   if(flag1 == 0) //dp/dx
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = fx_bale[i*nx+j]*xout[i*nx+j];
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = fxz_bale[i*nx+j]*xout[i*nx+j];
		   }
	   }
	   else // dp/dz
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = fz_bale[i*nx+j]*xout[i*nx+j];
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = fxz_bale[i*nx+j]*xout[i*nx+j];
		   }
	   }

       fftwf_execute(xpi);

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   dp[i][j] = creal(xout[i*nx+j])/nxz;
	   
       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif 
}

/*****************************************************************************************/
void RstaggerPS1order2Ddx3(float **p,float **dp,int nx,int nz,sf_complex *f_sta_x,sf_complex *b_sta_x,
							sf_complex *f_sta_z,sf_complex *b_sta_z,int flag1,int flag2)
/*< RstaggerPS1order2Ddx caculate 1order derivation of p using rotated-staggered PS method 
            if flag1==0,caculate dp/dx else caculate dp/dz
             if flag2==0,forward stagger else backward >*/
{
	
#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;
	   
	   int i,j,nxz;
	   nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

     //  fftwf_plan_with_nthreads(omp_get_max_threads());
       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       //fftwf_plan_with_nthreads(omp_get_max_threads());
       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   xin[i*nx+j] = sf_cmplx(p[i][j], 0.0);

       fftwf_execute(xp);


	   if(flag1 == 0) //dp/dx
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = xout[i*nx+j]*f_sta_x[i*nx+j];
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = xout[i*nx+j]*b_sta_x[i*nx+j];
		   }
	   }
	   else // dp/dz
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = xout[i*nx+j]*f_sta_z[i*nx+j];
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = xout[i*nx+j]*b_sta_z[i*nx+j];
		   }
	   }

       fftwf_execute(xpi);

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   dp[i][j] = creal(xout[i*nx+j])/nxz;
	   
       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif 
}

void staggerPS1order2Ddx3(float *p,float *dp,int nx,int nz,sf_complex *f_sta_x,sf_complex *b_sta_x,
						sf_complex *f_sta_z,sf_complex *b_sta_z,int flag1,int flag2)
/*< staggerPS1order2Ddx caculate 1order derivation of p using staggered PS method 
             if flag1==0,caculate dp/dx else caculate dp/dz 
               if flag2==0,forward stagger else backward >*/
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;

       int i,nxz;
       nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<nxz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       if(flag1 == 0) //dp/dx
       {
           if(flag2 == 0) //forward stagger
               for(i=0;i<nxz;i++)
                       xin[i] = xout[i]*f_sta_x[i];
           else
               for(i=0;i<nxz;i++)
                       xin[i] = xout[i]*b_sta_x[i];
       }
       else // dp/dz
       {
           if(flag2 == 0) // forward stagger
               for(i=0;i<nxz;i++)
                       xin[i] = xout[i]*f_sta_z[i];
           else
               for(i=0;i<nxz;i++)
                       xin[i] = xout[i]*b_sta_z[i];
       }

       fftwf_execute(xpi);

       for(i=0;i<nxz;i++)
               dp[i] = creal(xout[i])/nxz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif
}

/*****************************************************************************************/
void staggerPS1order2Ddx4(float **p,float **dp,int nx,int nz,sf_complex *f_sta_x,sf_complex *b_sta_x,
						sf_complex *f_sta_z,sf_complex *b_sta_z,int flag1,int flag2)
/*< staggerPS1order2Ddx caculate 1order derivation of p using staggered PS method 
             if flag1==0,caculate dp/dx else caculate dp/dz  
               if flag2==0,forward stagger else backward >*/
{
	
#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_complex *xin, *xout;

       fftwf_plan xp;
       fftwf_plan xpi;
	   
	   int i,j,nxz;
	   nxz = nx*nz;

       xin=sf_complexalloc(nxz);
       xout=sf_complexalloc(nxz);

       xp=fftwf_plan_dft_2d(nz,nx, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nz,nx,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   xin[i*nx+j] = sf_cmplx(p[i][j], 0.0);

       fftwf_execute(xp);

	   if(flag1 == 0) //dp/dx
	   {
		   if(flag2 == 0) //forward stagger 
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = f_sta_x[i*nx+j]*xout[i*nx+j];
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = b_sta_x[i*nx+j]*xout[i*nx+j];
		   }
	   }
	   else // dp/dz
	   {
		   if(flag2 == 0) // forward stagger
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = f_sta_z[i*nx+j]*xout[i*nx+j];
		   }
		   else
		   {
			   for(i=0;i<nz;i++)
				   for(j=0;j<nx;j++)
					   xin[i*nx+j] = b_sta_z[i*nx+j]*xout[i*nx+j];
		   }
	   }

       fftwf_execute(xpi);

	   for(i=0;i<nz;i++)
		   for(j=0;j<nx;j++)
			   dp[i][j] = creal(xout[i*nx+j])/nxz;
	   
       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xin);
       free(xout);
#endif 
}

void RkstaggerPS1order3Ddx(float *p,float *dp,int nx,int ny,int nz, complex float *f_Rksta_x,complex float *f_Rksta_y,
		complex float *f_Rksta_z,complex float *b_Rksta_x,complex float *b_Rksta_y,complex float *b_Rksta_z,int flag1,int flag2)
/*< RstaggerPS1order3Ddx caculate 1order derivation of p using rotated-staggered PS method with k-space adjustment 
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

       fftwf_plan_with_nthreads(30);
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       fftwf_plan_with_nthreads(30);
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

       // FFT: from (x,z) to (kx, kz) domain /
       #pragma omp parallel for
       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(p[i], 0.0);

       fftwf_execute(xp);

       if(flag1 == 0) //dp/dx
       {
           if(flag2 == 0) //forward stagger
			   #pragma omp parallel for private(i) schedule(dynamic) shared(xin,f_Rksta_x,xout)
			   for(i=0;i<nxyz;i++)
				   xin[i] = f_Rksta_x[i]*xout[i];
           else
			   #pragma omp parallel for private(i) schedule(dynamic) shared(xin,b_Rksta_x,xout)
			   for(i=0;i<nxyz;i++)
				   xin[i] = b_Rksta_x[i]*xout[i];
       }
       else if(flag1==1) //dp/dy
       {
           if(flag2 == 0) //forward stagger
			   #pragma omp parallel for private(i) schedule(dynamic) shared(xin,f_Rksta_y,xout)
			   for(i=0;i<nxyz;i++)
				   xin[i] = f_Rksta_y[i]*xout[i];
           else
			   #pragma omp parallel for private(i) schedule(dynamic) shared(xin,b_Rksta_y,xout)
			   for(i=0;i<nxyz;i++)
				   xin[i] = b_Rksta_y[i]*xout[i];
       }
       else // dp/dz
       {
           if(flag2 == 0) //forward stagger
			   #pragma omp parallel for private(i) schedule(dynamic) shared(xin,f_Rksta_z,xout)
			   for(i=0;i<nxyz;i++)
				   xin[i] = f_Rksta_z[i]*xout[i];
           else
			   #pragma omp parallel for private(i) schedule(dynamic) shared(xin,b_Rksta_z,xout)
			   for(i=0;i<nxyz;i++)
				   xin[i] = b_Rksta_z[i]*xout[i];
       }

       fftwf_execute(xpi);

	   #pragma omp parallel for private(i) schedule(dynamic) shared(nxyz,dp,xout)
       for(i=0;i<nxyz;i++)
              dp[i] = creal(xout[i])/nxyz;

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
//	   fftwf_cleanup_threads();
       free(xin);
       free(xout);
#endif
}

void FFT3d(float *in, complex float *xout, int nx,int ny,int nz)
/*< 3D Fourier transform using FFTW3 package >*/ 
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin;

       fftwf_plan xp;

       int i,nxyz;
       nxyz = nx*ny*nz;

       xin=sf_complexalloc(nxyz);

       fftwf_plan_with_nthreads(30);
       xp=fftwf_plan_dft_3d(ny,nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_FORWARD,FFTW_ESTIMATE);

       #pragma omp parallel for private(i) schedule(dynamic) shared(xin,in)
       for(i=0;i<nxyz;i++)
               xin[i] = sf_cmplx(in[i], 0.0);

       fftwf_execute(xp);

       fftwf_destroy_plan(xp);
	 //  fftwf_cleanup_threads();
       free(xin);
#endif
}

void FFTderivative3d(complex float *p,float *dp, int nx,int ny,int nz, complex float *ikx)
/*< derivative with FFT, ikx is derivative operator >*/ 
{

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar

       sf_complex *xin,*xout;

       fftwf_plan xpi;

       int i,nxyz;
       nxyz = nx*ny*nz;

       xin=sf_complexalloc(nxyz);
       xout=sf_complexalloc(nxyz);

       fftwf_plan_with_nthreads(30);
       xpi=fftwf_plan_dft_3d(ny,nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
                FFTW_BACKWARD,FFTW_ESTIMATE);

	   #pragma omp parallel for private(i) schedule(dynamic) shared(xin,ikx,p)
	   for(i=0;i<nxyz;i++)
		   xin[i] = ikx[i]*p[i];

       fftwf_execute(xpi);

	   #pragma omp parallel for private(i) schedule(dynamic) shared(nxyz,dp,xout)
       for(i=0;i<nxyz;i++)
              dp[i] = creal(xout[i])/nxyz;

       fftwf_destroy_plan(xpi);
//	   fftwf_cleanup_threads();
       free(xin);
       free(xout);
#endif
}

