/* wave-modes vector decomposition based on low-rank decomposition */
/*
  Copyright (C) 2012 Tongji University (Jiubing Cheng) 
  and The University of Texas at Austin (Sergey Fomel)
 
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
void decomplowrank2d_stagger(sf_complex *ldataxx,sf_complex *rdataxx,sf_complex *fmidxx,
                      sf_complex *ldataxz,sf_complex *rdataxz,sf_complex *fmidxz,
                      sf_complex *ldatazx,sf_complex *rdatazx,sf_complex *fmidzx,
                      sf_complex *ldatazz,sf_complex *rdatazz,sf_complex *fmidzz,
                      float *px, float *pz, int *ijkx, int *ijkz,
                      int nx, int nz, int m, int n, int MM,
                      int m2xx, int n2xx, int m2xz, int n2xz,int m2zx,int n2zx,int m2zz, int n2zz)
/*< decomplowrank2d_stagger: vector decomposition based on low-rank decomposition >*/
{
       int   i, im, im2, jn2, ikx, ikz;
       sf_complex sum1, sum2, *wp;

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_warning("============= using SF_HAS_FFTW ====");

       sf_complex *pxx, *xin, *xout;
       sf_complex *pzz;

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       pxx=sf_complexalloc(n);
       pzz=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++){
           xin[i]=sf_cmplx(px[i], 0.);
           px[i] = 0.0;
       }
       fftwf_execute(xp);
       for(i=0;i<n;i++) pxx[i] = xout[i];

       for(i=0;i<m;i++){
           xin[i]=sf_cmplx(pz[i], 0.);
           pz[i] = 0.0;
       }
       fftwf_execute(xp);
       for(i=0;i<n;i++) pzz[i] = xout[i];

       ///////////////////////////////////////////////////////// P-wave's x-component
       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2xx);
       for(jn2=0;jn2<n2xx;jn2++)
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
                xin[i]=rdataxx[ii+ijkz[ikz]]*pxx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = xout[im]/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2xx;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2xx;jn2++)
              sum2 += fmidxx[im2*n2xx+jn2]*wp[jn2*m+im];

           sum1 += ldataxx[im*m2xx+im2]*sum2;
         }//im2 loop
         px[im] = creal(sum1);
       } 
       free(wp);

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2xz);
       for(jn2=0;jn2<n2xz;jn2++)
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
                xin[i]=rdataxz[ii+ijkz[ikz]]*pzz[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = xout[im]/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2xz;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2xz;jn2++)
              sum2 += fmidxz[im2*n2xz+jn2]*wp[jn2*m+im];

           sum1 += ldataxz[im*m2xz+im2]*sum2;
         }//im2 loop
         px[im] += creal(sum1);
       } 
       free(wp);
       ///////////////////////////////////////////////////////// P-wave's z-component
       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2zz);
       for(jn2=0;jn2<n2zz;jn2++)
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
                xin[i]=rdatazz[ii+ijkz[ikz]]*pzz[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = xout[im]/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2zz;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2zz;jn2++)
              sum2 += fmidzz[im2*n2zz+jn2]*wp[jn2*m+im];

           sum1 += ldatazz[im*m2zz+im2]*sum2;
         }//im2 loop
         pz[im] = creal(sum1);
       } 
       free(wp);

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2zx);
       for(jn2=0;jn2<n2zx;jn2++)
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
                xin[i]=rdatazx[ii+ijkz[ikz]]*pxx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = (xout[im])/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2zx;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2zx;jn2++)
              sum2 += fmidzx[im2*n2zx+jn2]*wp[jn2*m+im];

           sum1 += ldatazx[im*m2zx+im2]*sum2;
         }//im2 loop
         pz[im] += creal(sum1);
       } 
       free(wp);

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);

       free(pxx);
       free(pzz);
       free(xin);
       free(xout);
#else  // using FFTW in user's own computer
       sf_warning("============= using user installed FFTW ====");
#endif
}

/*****************************************************************************************/
void decomplowrank2ds_stagger(sf_complex *ldataxx,sf_complex *rdataxx,sf_complex *fmidxx,
                      sf_complex *ldataxz,sf_complex *rdataxz,sf_complex *fmidxz,
                      sf_complex *ldatazx,sf_complex *rdatazx,sf_complex *fmidzx,
                      sf_complex *ldatazz,sf_complex *rdatazz,sf_complex *fmidzz,
                      float *px, float *pz, int *ijkx, int *ijkz,
                      int nx, int nz, int m, int n, int MM,
                      int m2xx, int n2xx, int m2xz, int n2xz,int m2zx,int n2zx,int m2zz, int n2zz)
/*< decomplowrank2ds_stagger: SV-wave vector decomposition based on low-rank decomposition >*/
{
       int   i, im, im2, jn2, ikx, ikz;
       sf_complex sum1, sum2, *wp;

#ifdef SF_HAS_FFTW  // using FFTW in Madagascar
 
       sf_warning("============= using SF_HAS_FFTW ====");

       sf_complex *pxx, *xin, *xout;
       sf_complex *pzz;

       fftwf_plan xp;
       fftwf_plan xpi;

       xin=sf_complexalloc(m);
       xout=sf_complexalloc(n);
       pxx=sf_complexalloc(n);
       pzz=sf_complexalloc(n);

       xp=fftwf_plan_dft_2d(nx,nz, (fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_FORWARD,FFTW_ESTIMATE);

       xpi=fftwf_plan_dft_2d(nx,nz,(fftwf_complex *) xin, (fftwf_complex *) xout,
			    FFTW_BACKWARD,FFTW_ESTIMATE);

       /* FFT: from (x,z) to (kx, kz) domain */
       for(i=0;i<m;i++){
           xin[i]=sf_cmplx(px[i], 0.);
           px[i] = 0.0;
       }
       fftwf_execute(xp);
       for(i=0;i<n;i++) pxx[i] = xout[i];

       for(i=0;i<m;i++){
           xin[i]=sf_cmplx(pz[i], 0.);
           pz[i] = 0.0;
       }
       fftwf_execute(xp);
       for(i=0;i<n;i++) pzz[i] = xout[i];

       ///////////////////////////////////////////////////////// SV-wave's x-component
       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2zz);
       for(jn2=0;jn2<n2zz;jn2++)
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
                xin[i]=rdatazz[ii+ijkz[ikz]]*pxx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = (xout[im])/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2zz;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2zz;jn2++)
              sum2 += fmidzz[im2*n2zz+jn2]*wp[jn2*m+im];

           sum1 += ldatazz[im*m2zz+im2]*sum2;
         }//im2 loop
         px[im] = creal(sum1);
       } 
       free(wp);

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2xz);
       for(jn2=0;jn2<n2xz;jn2++)
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
                xin[i]=rdataxz[ii+ijkz[ikz]]*pzz[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = (xout[im])/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2xz;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2xz;jn2++)
              sum2 += fmidxz[im2*n2xz+jn2]*wp[jn2*m+im];

           sum1 += ldataxz[im*m2xz+im2]*sum2;
         }//im2 loop
         px[im] -= creal(sum1);
       } 
       free(wp);
       ///////////////////////////////////////////////////////// SV-wave's z-component
       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2xx);
       for(jn2=0;jn2<n2xx;jn2++)
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
                xin[i]=rdataxx[ii+ijkz[ikz]]*pzz[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = (xout[im])/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2xx;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2xx;jn2++)
              sum2 += fmidxx[im2*n2xx+jn2]*wp[jn2*m+im];

           sum1 += ldataxx[im*m2xx+im2]*sum2;
         }//im2 loop
         pz[im] = creal(sum1);
       } 
       free(wp);

       /* n2 IFFT from (kx, kz) to (x, z) domain*/
       wp = sf_complexalloc(m*n2zx);
       for(jn2=0;jn2<n2zx;jn2++)
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
                xin[i]=rdatazx[ii+ijkz[ikz]]*pxx[i];          
                i++;
              }
            }
            // (kx,kz) to (x, z) domain
            fftwf_execute(xpi);

            for(im=0;im<m;im++)
                wp[jn2*m+im] = (xout[im])/n;
       }

       // Matrix multiplication in space-domain 
       for(im=0;im<m;im++)
       {
         sum1=0.0;
         for(im2=0;im2<m2zx;im2++)
         {
           sum2=0.0;
           for(jn2=0;jn2<n2zx;jn2++)
              sum2 += fmidzx[im2*n2zx+jn2]*wp[jn2*m+im];

           sum1 += ldatazx[im*m2zx+im2]*sum2;
         }//im2 loop
         pz[im] -= creal(sum1);
       } 
       free(wp);

       fftwf_destroy_plan(xp);
       fftwf_destroy_plan(xpi);

       free(pxx);
       free(pzz);
       free(xin);
       free(xout);
#else  // using FFTW in user's own computer
       sf_warning("============= using user installed FFTW ====");
#endif
}
