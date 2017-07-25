/* apply low-rank propagators to wavefield component */
/*
  Copyright (C) 2014 Tongji University (Jiubing Cheng) 
  and King Abdulah University of Science and Technology (Zedong Wu and Tariq Alkhalifah)
 
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

/*****************************************************************************************/
void fwpvti2de1ststagger( sf_complex *ldata, sf_complex *rdata, sf_complex *fmid,
		float *y, float *x, int *ijkx, int *ijkz, int nx,int nz,int m,int n,int m2,int n2)
/*< fwpvti2de1ststegger: apply low-rank decomposed propagator using velocity-stress equation considering stability to the wavefield component >*/
{
       int i, im, im2, jn2, ikx, ikz;
       float sum1, sum2;
       sf_complex *xx, *xin, *xout;

       sf_complex *wp = sf_complexalloc(m*n2);

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       xx=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++) xin[i]=sf_dcmplx(x[i], 0.0);

       fftwf_execute(xp);
           
       for(i=0;i<n;i++) xx[i] = xout[i];

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
	   int index;
       for(jn2=0;jn2<n2;jn2++)
       {
           i=0;
           int jn2n=jn2*n;
           for(ikx=0;ikx<nx;ikx++)
           {
              // Note: Spectrum of the operator is differently orderred as the spectrum after FFT
              int ixnz=ijkx[ikx]*nz;
              int ii=jn2n+ixnz;

              for(ikz=0;ikz<nz;ikz++)
              {
				index = ixnz + ijkz[ikz];
#ifdef SF_HAS_COMPLEX_H
				xin[i]=rdata[ii+ijkz[ikz]]*xx[i];          
#else
				xin[i]=sf_cmul(rdata[ii+ijkz[ikz]], xx[i]);          
#endif
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
#ifdef SF_HAS_COMPLEX_H
                wp[jn2*m+im] = xout[im]*(1.0/n);
#else
                wp[jn2*m+im] = sf_crmul(xout[im], 1.0/n);
#endif
       }

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xx);
       free(xin);
       free(xout);

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=sf_cmplx(0.,0.);
         for(im2=0;im2<m2;im2++)
         {
           sum2=sf_cmplx(0.,0.);
           for(jn2=0;jn2<n2;jn2++)
#ifdef SF_HAS_COMPLEX_H
              sum2 += fmid[im2*n2+jn2]*wp[jn2*m+im];
#else
              sum2 += sf_cmul(fmid[im2*n2+jn2], wp[jn2*m+im]);
#endif

#ifdef SF_HAS_COMPLEX_H
           sum1 += ldata[im*m2+im2]*sum2;
#else
           sum1 += sf_cmul(ldata[im*m2+im2], sum2);
#endif
         }//im2 loop
         y[im] = creal(sum1);
       } 
	   free(wp);
}

/*****************************************************************************************/
void fwpvti2de1ststaggersvd(sf_complex *ldata, sf_complex *rdata, sf_complex *fmid,
		float *y, float *x, int *ijkx, int *ijkz, int nx,int nz,int m,int n,int m2,int n2,float kxm, float kzm, float kxzm, float *akx,    float *akz)
/*< fwpvti2de1ststegger: apply low-rank decomposed propagator using velocity-stress equation considering stability to the wavefield component >*/
{
       int i, im, im2, jn2, ikx, ikz;
       float sum1, sum2;
       sf_complex *xx, *xin, *xout;
	   float kx, kz, kxz;

       sf_complex *wp = sf_complexalloc(m*n2);

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       xx=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++) xin[i]=sf_dcmplx(x[i], 0.0);

       fftwf_execute(xp);
           
       for(i=0;i<n;i++) xx[i] = xout[i];

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
	   int index;
       for(jn2=0;jn2<n2;jn2++)
       {
           i=0;
           int jn2n=jn2*n;
           for(ikx=0;ikx<nx;ikx++)
           {
              // Note: Spectrum of the operator is differently orderred as the spectrum after FFT
              int ixnz=ijkx[ikx]*nz;
              int ii=jn2n+ixnz;

              for(ikz=0;ikz<nz;ikz++)
              {
				index = ixnz + ijkz[ikz];
				kx = akx[index]; 
                kz = akz[index];
                kxz = sqrt(kx*kx+kz*kz);

				if (kx>kxm || kz>kzm || kxz>kxzm)
                   xin[i] = 0.0;
                else if (kx<-kxm || kz<-kzm)
                   xin[i] = 0.0;
                else{
#ifdef SF_HAS_COMPLEX_H
				xin[i]=rdata[ii+ijkz[ikz]]*xx[i];          
#else
				xin[i]=sf_cmul(rdata[ii+ijkz[ikz]], xx[i]);          
#endif
				}
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
#ifdef SF_HAS_COMPLEX_H
                wp[jn2*m+im] = xout[im]*(1.0/n);
#else
                wp[jn2*m+im] = sf_crmul(xout[im], 1.0/n);
#endif
       }

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xx);
       free(xin);
       free(xout);

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=sf_cmplx(0.,0.);
         for(im2=0;im2<m2;im2++)
         {
           sum2=sf_cmplx(0.,0.);
           for(jn2=0;jn2<n2;jn2++)
#ifdef SF_HAS_COMPLEX_H
              sum2 += fmid[im2*n2+jn2]*wp[jn2*m+im];
#else
              sum2 += sf_cmul(fmid[im2*n2+jn2], wp[jn2*m+im]);
#endif

#ifdef SF_HAS_COMPLEX_H
           sum1 += ldata[im*m2+im2]*sum2;
#else
           sum1 += sf_cmul(ldata[im*m2+im2], sum2);
#endif
         }//im2 loop
         y[im] = creal(sum1);
       } 
	   free(wp);
}

/*****************************************************************************************/
void fwpvti2destaggersvd(sf_complex *ldata, sf_complex *rdata, sf_complex *fmid,
		float *y, float *x, int *ijkx, int *ijkz, int nx,int nz,int m,int n,int m2,int n2,float kxm, float kzm, float kxzm, float *akx,    float *akz)
/*< fwpvti2destegger: apply low-rank decomposed propagator using displacement-stress equation considering stability to the wavefield component >*/
{
       int i, im, im2, jn2, ikx, ikz;
       float sum1, sum2;
       sf_complex *xx, *xin, *xout;
	   float kx, kz, kxz;

       sf_complex *wp = sf_complexalloc(m*n2);

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       xx=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++) xin[i]=sf_dcmplx(x[i], 0.0);

       fftwf_execute(xp);
           
       for(i=0;i<n;i++) xx[i] = xout[i];

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
	   int index;
       for(jn2=0;jn2<n2;jn2++)
       {
           i=0;
           int jn2n=jn2*n;
           for(ikx=0;ikx<nx;ikx++)
           {
              // Note: Spectrum of the operator is differently orderred as the spectrum after FFT
              int ixnz=ijkx[ikx]*nz;
              int ii=jn2n+ixnz;

              for(ikz=0;ikz<nz;ikz++)
              {
				index = ixnz + ijkz[ikz];
				kx = akx[index]; 
                kz = akz[index];
                kxz = sqrt(kx*kx+kz*kz);

				if (kx>kxm || kz>kzm || kxz>kxzm)
                   xin[i] = 0.0;
                else if (kx<-kxm || kz<-kzm)
                   xin[i] = 0.0;
                else{
#ifdef SF_HAS_COMPLEX_H
				xin[i]=rdata[ii+ijkz[ikz]]*xx[i];          
#else
				xin[i]=sf_cmul(rdata[ii+ijkz[ikz]], xx[i]);          
#endif
				}
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
#ifdef SF_HAS_COMPLEX_H
                wp[jn2*m+im] = xout[im]*(1.0/n);
#else
                wp[jn2*m+im] = sf_crmul(xout[im], 1.0/n);
#endif
       }

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xx);
       free(xin);
       free(xout);

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=sf_cmplx(0.,0.);
         for(im2=0;im2<m2;im2++)
         {
           sum2=sf_cmplx(0.,0.);
           for(jn2=0;jn2<n2;jn2++)
#ifdef SF_HAS_COMPLEX_H
              sum2 += fmid[im2*n2+jn2]*wp[jn2*m+im];
#else
              sum2 += sf_cmul(fmid[im2*n2+jn2], wp[jn2*m+im]);
#endif

#ifdef SF_HAS_COMPLEX_H
           sum1 += ldata[im*m2+im2]*sum2;
#else
           sum1 += sf_cmul(ldata[im*m2+im2], sum2);
#endif
         }//im2 loop
         y[im] = creal(sum1);
       } 
	   free(wp);
}

/*****************************************************************************************/
void fwpvti2de1stlr(float *ldata,float *rdata,float *fmid, float *y, float *x, int *ijkx, int *ijkz,
                      int nx,int nz,int m,int n,int m2,int n2)
/*< fwpvti2de1stlr: apply low-rank decomposed propagator using velocity-stress equation considering stability to the wavefield component >*/
{
       int i, im, im2, jn2, ikx, ikz;
       float sum1, sum2, *wp;
       sf_complex *xx, *xin, *xout;

       wp = sf_floatalloc(m*n2);

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       xx=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++) xin[i]=sf_cmplx(0.0, x[i]);

       fftwf_execute(xp);
           
       for(i=0;i<n;i++) xx[i] = xout[i];

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
	   int index;
       for(jn2=0;jn2<n2;jn2++)
       {
           i=0;
           int jn2n=jn2*n;
           for(ikx=0;ikx<nx;ikx++)
           {
              // Note: Spectrum of the operator is differently orderred as the spectrum after FFT
              int ixnz=ijkx[ikx]*nz;
              int ii=jn2n+ixnz;

              for(ikz=0;ikz<nz;ikz++)
              {
				index = ixnz + ijkz[ikz];
				xin[i]=rdata[ii+ijkz[ikz]]*xx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = creal(xout[im])/n;
       }

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xx);
       free(xin);
       free(xout);

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2;jn2++)
              sum2 += fmid[im2*n2+jn2]*wp[jn2*m+im];

           sum1 += ldata[im*m2+im2]*sum2;
         }//im2 loop
         y[im] = sum1;
       } 
	   free(wp);
}

/*****************************************************************************************/
void fwpvti2de1stlrsvd(float *ldata,float *rdata,float *fmid, float *y, float *x, int *ijkx, int *ijkz,
                      int nx,int nz,int m,int n,int m2,int n2,float kxm, float kzm, float kxzm, float *akx, float *akz)
/*< fwpvti2de1stlrsvd: apply low-rank decomposed propagator using velocity-stress equation considering stability to the wavefield component with svd filtering >*/
{
       int i, im, im2, jn2, ikx, ikz;
       float sum1, sum2, *wp;
       sf_complex *xx, *xin, *xout;
	   float kx, kz, kxz;

       wp = sf_floatalloc(m*n2);

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       xx=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++) xin[i]=sf_cmplx(0.0, x[i]);

       fftwf_execute(xp);
           
       for(i=0;i<n;i++) xx[i] = xout[i];

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
	   int index;
       for(jn2=0;jn2<n2;jn2++)
       {
           i=0;
           int jn2n=jn2*n;
           for(ikx=0;ikx<nx;ikx++)
           {
              // Note: Spectrum of the operator is differently orderred as the spectrum after FFT
              int ixnz=ijkx[ikx]*nz;
              int ii=jn2n+ixnz;

              for(ikz=0;ikz<nz;ikz++)
              {
				index = ixnz + ijkz[ikz];
				// filtering wavenumber beyond maximum wavenumber determined by Fmax
				kx = akx[index];
				kz = akz[index];
				kxz = sqrt(kx*kx+kz*kz);
				if (kx>kxm || kz>kzm || kxz>kxzm)
	                xin[i] = 0.0;
	            else if (kx<-kxm || kz<-kzm)
	                 xin[i] = 0.0;
	            else
				  xin[i]=rdata[ii+ijkz[ikz]]*xx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = creal(xout[im])/n;
       }

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xx);
       free(xin);
       free(xout);

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2;jn2++)
              sum2 += fmid[im2*n2+jn2]*wp[jn2*m+im];

           sum1 += ldata[im*m2+im2]*sum2;
         }//im2 loop
         y[im] = sum1;
       } 
	   free(wp);
}


/*****************************************************************************************/
void fwpvti2de1stlr_rsg(float *ldata,float *rdata,float *fmid, float *y, float *x, int *ijkx, int *ijkz,
                     int nx,int nz,int m,int n,int m2,int n2,
					 float dxxh, float dzzh, float*kx, float*kz, int flag)
/*< fwpvti2de1stlr: apply low-rank decomposed propagator using velocity-stress equation considering stability to the wavefield component >*/
{
       int i, im, im2, jn2, ikx, ikz;
       float sum1, sum2, *wp;
       sf_complex *xx, *xin, *xout;

       wp = sf_floatalloc(m*n2);

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       xx=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);


       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++) xin[i]=sf_cmplx(0.0, x[i]);

       fftwf_execute(xp);
 
	  i = 0;
	  int index;
	  for(ikx=0;ikx<nx;ikx++)
	  {
		  for(ikz=0;ikz<nz;ikz++)
		  {
			  index = ijkx[ikx]*nz + ijkz[ikz];
			  if(flag==0) //forward shift
				xx[i] = xout[i]*cexpf(I*(kx[index]*dxxh-kz[index]*dzzh));
			  else
				xx[i] = xout[i]*cexpf(-I*(kx[index]*dxxh-kz[index]*dzzh));
			  i++;
		  }
	  }

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       for(jn2=0;jn2<n2;jn2++)
       {
           i=0;
           int jn2n=jn2*n;
           for(ikx=0;ikx<nx;ikx++)
           {
              // Note: Spectrum of the operator is differently orderred as the spectrum after FFT
              int ixnz=ijkx[ikx]*nz;
              int ii=jn2n+ixnz;

              for(ikz=0;ikz<nz;ikz++)
              {
				index = ii+ijkz[ikz];
			    xin[i]=rdata[index]*xx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = creal(xout[im])/n;
       }

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xx);
       free(xin);
       free(xout);

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2;jn2++)
              sum2 += fmid[im2*n2+jn2]*wp[jn2*m+im];

           sum1 += ldata[im*m2+im2]*sum2;
         }//im2 loop
         y[im] = sum1;
       } 
	   free(wp);
}

/*****************************************************************************************/
void fwpvti2de1stlr_rsg_double(double *ldata,double *rdata,double *fmid, double *y, double *x, int *ijkx, int *ijkz,
                     int nx,int nz,int m,int n,int m2,int n2,
					 float dxxh, float dzzh, float*kx, float*kz, int flag)
/*< fwpvti2de1stlr: apply low-rank decomposed propagator using velocity-stress equation considering stability to the wavefield component >*/
{
       int i, im, im2, jn2, ikx, ikz;
       long double sum1, sum2, *wp;
       fftw_complex *xx, *xin, *xout;

       wp = (long double*)malloc(sizeof(long double)*m*n2);

       fftwf_plan xp;
       fftwf_plan xpi;

	   xin=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*m);
	   xout=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n);
	   xx=(fftw_complex*) fftw_malloc(sizeof(fftw_complex)*n);

       xp=fftw_plan_dft_2d(nx,nz, (fftw_complex *) xin, (fftw_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftw_plan_dft_2d(nx,nz,(fftw_complex *) xin, (fftw_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);


       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++) xin[i]=0.0+ I*x[i];

       fftwf_execute(xp);
 
	  i = 0;
	  int index;
	  for(ikx=0;ikx<nx;ikx++)
	  {
		  for(ikz=0;ikz<nz;ikz++)
		  {
			  index = ijkx[ikx]*nz + ijkz[ikz];
			  if(flag==0) //forward shift
				xx[i] = xout[i]*cexpf(I*(kx[index]*dxxh-kz[index]*dzzh));
			  else
				xx[i] = xout[i]*cexpf(-I*(kx[index]*dxxh-kz[index]*dzzh));
			  i++;
		  }
	  }

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       for(jn2=0;jn2<n2;jn2++)
       {
           i=0;
           int jn2n=jn2*n;
           for(ikx=0;ikx<nx;ikx++)
           {
              // Note: Spectrum of the operator is differently orderred as the spectrum after FFT
              int ixnz=ijkx[ikx]*nz;
              int ii=jn2n+ixnz;

              for(ikz=0;ikz<nz;ikz++)
              {
				index = ii+ijkz[ikz];
			    xin[i]=rdata[index]*xx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = creal(xout[im])/n;
       }

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);
       free(xx);
       free(xin);
       free(xout);

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2;jn2++)
              sum2 += fmid[im2*n2+jn2]*wp[jn2*m+im];

           sum1 += ldata[im*m2+im2]*sum2;
         }//im2 loop
         y[im] = sum1;
       } 
	   free(wp);
}

