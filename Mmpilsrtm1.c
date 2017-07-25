/* 2-D Low-rank One-step Least Pre-stack Reverse-Time-Migration in the complex domain (both img and data are complex valued)
img: crosscorrelation wite wource normalization (stdout)
*/
/*
  Copyright (C) 2014 University of Texas at Austin
  
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
#include <mpi.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef SF_HAS_FFTW
#include <fftw3.h>
#endif

/*******************************************************/
/* fft utils*/
static int n1, n2, nk;
static float wt;

static sf_complex **cc,**dd;

#ifdef SF_HAS_FFTW
static fftwf_plan cfg=NULL, icfg=NULL;
#else
static kiss_fft_cfg cfg1, icfg1, cfg2, icfg2;
static kiss_fft_cpx **tmp, *ctrace2;
static sf_complex *trace2;
#endif

int cfft2_init(int pad1           /* padding on the first axis */,
	       int nx,   int ny   /* input data size */, 
	       int *nx2, int *ny2 /* padded data size */)
/*< initialize >*/
{

#ifdef SF_HAS_FFTW
#ifdef _OPENMP
    fftwf_init_threads();
    sf_warning("Using threaded FFTW3! \n");
    fftwf_plan_with_nthreads(omp_get_max_threads());
#endif
#endif

#ifndef SF_HAS_FFTW
    int i2;
#endif

    nk = n1 = kiss_fft_next_fast_size(nx*pad1);
    
#ifndef SF_HAS_FFTW
    cfg1  = kiss_fft_alloc(n1,0,NULL,NULL);
    icfg1 = kiss_fft_alloc(n1,1,NULL,NULL);
#endif
  
    n2 = kiss_fft_next_fast_size(ny);

    cc = sf_complexalloc2(n1,n2);
    dd = sf_complexalloc2(nk,n2);
    
#ifndef SF_HAS_FFTW
    cfg2  = kiss_fft_alloc(n2,0,NULL,NULL);
    icfg2 = kiss_fft_alloc(n2,1,NULL,NULL);
 	
    tmp =    (kiss_fft_cpx **) sf_alloc(n2,sizeof(*tmp));
    tmp[0] = (kiss_fft_cpx *)  sf_alloc(nk*n2,sizeof(kiss_fft_cpx));
    for (i2=0; i2 < n2; i2++) {
	tmp[i2] = tmp[0]+i2*nk;
    }
	
    trace2 = sf_complexalloc(n2);
    ctrace2 = (kiss_fft_cpx *) trace2;
#endif

    *nx2 = n1;
    *ny2 = n2;
	
    wt =  1.0/(n1*n2);
	
    return (nk*n2);
}

void cfft2(sf_complex *inp /* [n1*n2] */, 
	   sf_complex *out /* [nk*n2] */)
/*< 2-D FFT >*/
{
    int i1, i2;

#ifdef SF_HAS_FFTW
    if (NULL==cfg) {
      cfg = fftwf_plan_dft_2d(n2,n1,
			      (fftwf_complex *) cc[0], 
			      (fftwf_complex *) dd[0],
			      FFTW_FORWARD, FFTW_MEASURE);
      if (NULL == cfg) sf_error("FFTW failure.");
    }
#endif

    /* FFT centering */
    for (i2=0; i2<n2; i2++) {
	for (i1=0; i1<n1; i1++) {
#ifdef SF_HAS_COMPLEX_H
		cc[i2][i1] = ((i2%2==0)==(i1%2==0))? inp[i2*n1+i1]:-inp[i2*n1+i1];
#else
		cc[i2][i1] = ((i2%2==0)==(i1%2==0))? inp[i2*n1+i1]:sf_cneg(inp[i2*n1+i1]);
#endif
	  /*
#ifdef SF_HAS_COMPLEX_H
		cc[i2][i1] = ((i2%2==0)==(i1%2==0))? inp[i2*n1+i1]:(-1*inp[i2*n1+i1]);
#else
		cc[i2][i1] = ((i2%2==0)==(i1%2==0))? inp[i2*n1+i1]:sf_cneg(inp[i2*n1+i1]);
#endif
	  */
	}
    }

#ifdef SF_HAS_FFTW
    fftwf_execute(cfg);
    for (i2=0; i2<n2; i2++) {
	for (i1=0; i1<nk; i1++) {
	    out[i2*nk+i1]=dd[i2][i1];
	}
    }
#else	
    for (i2=0; i2 < n2; i2++) {
	kiss_fft_stride(cfg1,(kiss_fft_cpx *) cc[i2],tmp[i2],1);
    }

    for (i1=0; i1 < nk; i1++) {
	kiss_fft_stride(cfg2,tmp[0]+i1,ctrace2,nk);
	for (i2=0; i2<n2; i2++) {
	    out[i2*nk+i1] = trace2[i2];
	}
    }
#endif
}

void icfft2_allocate(sf_complex *inp /* [nk*n2] */)
/*< allocate inverse transform >*/
{
#ifdef SF_HAS_FFTW
    icfg = fftwf_plan_dft_2d(n2,n1,
			     (fftwf_complex *) inp, 
			     (fftwf_complex *) cc[0],
			     FFTW_BACKWARD, FFTW_MEASURE);
    if (NULL == icfg) sf_error("FFTW failure.");
#endif
}

void icfft2(sf_complex *out /* [n1*n2] */, 
	    sf_complex *inp /* [nk*n2] */)
/*< 2-D inverse FFT >*/
{
    int i1, i2;

#ifdef SF_HAS_FFTW
    fftwf_execute(icfg);
#else
    for (i1=0; i1 < nk; i1++) {
	kiss_fft_stride(icfg2,(kiss_fft_cpx *) (inp+i1),ctrace2,nk);
		
	for (i2=0; i2<n2; i2++) {
	    tmp[i2][i1] = ctrace2[i2];
	}
    }
    for (i2=0; i2 < n2; i2++) {
	kiss_fft_stride(icfg1,tmp[i2],(kiss_fft_cpx *) cc[i2],1);
    }
#endif
    
    /* FFT centering and normalization*/
    for (i2=0; i2<n2; i2++) {
	for (i1=0; i1<n1; i1++) {
#ifdef SF_HAS_COMPLEX_H
		out[i2*n1+i1] = (((i2%2==0)==(i1%2==0))? wt:-wt) * cc[i2][i1];
#else
		out[i2*n1+i1] = sf_crmul(cc[i2][i1],(((i2%2==0)==(i1%2==0))? wt:-wt));
#endif
	}
    }
}

void cfft2_finalize()
/*< clean up fftw >*/
{
/* make sure everything is back to its pristine state */
#ifdef SF_HAS_FFTW
#ifdef _OPENMP
    fftwf_cleanup_threads();
#endif
    fftwf_destroy_plan(cfg);
    fftwf_destroy_plan(icfg);
    fftwf_cleanup();
    cfg=NULL;
    icfg=NULL;
#else
    free(cfg1); cfg1=NULL;
    free(icfg1); icfg1=NULL;
    free(cfg2); cfg2=NULL;
    free(icfg2); icfg2=NULL;
#endif

    free(*cc);
    free(cc);
    free(*dd);
    free(dd);
}

/*******************************************************/
/* wave propagation utils*/
typedef struct Geopar {
    int   nx;
    int   nz;
    int   nxb;
    int   nzb;
    float dx;
    float dz;
    float ox;
    float oz;
    int   spx;
    int   spz;
    int   gpz;
    int   gpx;
    int   gpl;
    int   snpint;
    /*absorbing boundary*/
    int top;
    int bot;
    int lft;
    int rht;
    /*source parameters*/
    int nt;
    float dt;
    float trunc;
    /*data parameters*/
    int shtnum;
} * geopar; /*geometry parameters*/

typedef struct Mpipar {
    int cpuid;
    int numprocs;
} * mpipar; /*geometry parameters*/

void reflgen(int nzb, int nxb, int spz, int spx,
             int rectz, int rectx, int nrep, /*smoothing parameters*/
	     float *refl/*reflectivity map*/)
/*< Generate reflectivity map with smoothing >*/
{   
    int iz, i, j, i0, irep;
    int nzx=nzb*nxb; 
    sf_triangle tr;
    int n[2],s[2],rect[2];
    bool diff[2],box[2];

    n[0]=nzb; n[1]=nxb;
    s[0]=1;   s[1]=nzb;
    rect[0]=rectz; rect[1]=rectx;
    diff[0]=false; diff[1]=false;
    box[0]=false; box[1]=false;
    
#ifdef _OPENMP
#pragma omp parallel for private(iz)
#endif
    for (iz=0; iz < nzx; iz++) {
      refl[iz]=0;
    } 
    j=spx*nzb+spz; /*point source position*/
    refl[j]=1;
    
    /* 2-d triangle smoothing */
    for (i=0;i<2;i++) {
      if (rect[i] <= 1) continue;
      tr = sf_triangle_init (rect[i],n[i],box[i]);
      for (j=0; j < nzx/n[i]; j++) {
	i0 = sf_first_index (i,j,2,n,s);
	for (irep=0; irep < nrep; irep++) {
	  sf_smooth2 (tr,i0,s[i],diff[i],refl); // why adjoint?
	}
      }
      sf_triangle_close(tr);
    }
}

int lrosfor2(sf_complex ***wavfld, float **sill, sf_complex **rcd, bool verb,
	     sf_complex **lt, sf_complex **rt, int m2,
	     geopar geop, sf_complex *ww, float *rr, int pad1, bool illum)
/*< low-rank one-step forward modeling >*/
{
    int it,iz,im,ik,ix,i,j;     /* index variables */
    int nxb,nzb,gpz,gpx,gpl,snpint,wfit;
    int nt,nz,nx, nk, nz2, nx2, nzx2;
    float dt;
    sf_complex c;
    sf_complex *cwave, *cwavem;
    sf_complex **wave, *curr;
#ifdef _OPENMP
    int nth;
#endif

    nx = geop->nx;
    nz = geop->nz;
    nxb = geop->nxb;
    nzb = geop->nzb;
/*    dx = geop->dx;
      dz = geop->dz; */

/*    spx = geop->spx;
      spz = geop->spz; */
    gpz  = geop->gpz;
    gpx  = geop->gpx;
    gpl  = geop->gpl;
    snpint = geop->snpint;
    
    nt = geop->nt;
    dt = geop->dt;

#ifdef _OPENMP
#pragma omp parallel  
{
    nth = omp_get_num_threads();
}
    sf_warning(">>>> Using %d threads <<<<<", nth);
#endif
    
    /*Matrix dimensions*/
    nk = cfft2_init(pad1,nzb,nxb,&nz2,&nx2);
/*    nzx = nzb*nxb; */
    nzx2 = nz2*nx2;

    curr   = sf_complexalloc(nzx2);
    cwave  = sf_complexalloc(nk);
    cwavem = sf_complexalloc(nk);
    wave   = sf_complexalloc2(nzx2,m2);

    icfft2_allocate(cwavem);

#ifdef _OPENMP
#pragma omp parallel for private(iz)
#endif
    for (iz=0; iz < nzx2; iz++) {
	curr[iz] = sf_cmplx(0.,0.);
    }

    if (illum) {
#ifdef _OPENMP
#pragma omp parallel for private(iz)
#endif
      for (ix=0; ix < nx; ix++) {
	for (iz=0; iz < nz; iz++) {
	  sill[ix][iz] = 0.f;
	}
      }
    }

    /*Main loop*/
    wfit = 0;
    for (it = 0; it < nt; it++) {
	if (verb) sf_warning("Forward source it=%d/%d;", it, nt-1);
	
	/*matrix multiplication*/
	cfft2(curr,cwave);

	for (im = 0; im < m2; im++) {
#ifdef _OPENMP
#pragma omp parallel for private(ik)
#endif
	    for (ik = 0; ik < nk; ik++) {
#ifdef SF_HAS_COMPLEX_H
		cwavem[ik] = cwave[ik]*rt[ik][im];
#else
		cwavem[ik] = sf_cmul(cwave[ik],rt[ik][im]);
#endif
	    }
	    icfft2(wave[im],cwavem);
	}

#ifdef _OPENMP
#pragma omp parallel for private(ix,iz,i,j,im,c) shared(curr,lt,wave)
#endif
	for (ix = 0; ix < nxb; ix++) {
	    for (iz=0; iz < nzb; iz++) {
		i = iz+ix*nzb;  /* original grid */
		j = iz+ix*nz2; /* padded grid */
		if ((it*dt)<=geop->trunc) {
#ifdef SF_HAS_COMPLEX_H
		  c = ww[it] * rr[i]; // source term
#else
		  c = sf_crmul(ww[it], rr[i]); // source term
#endif
		} else {
		  c = sf_cmplx(0.,0.);
		}
		for (im = 0; im < m2; im++) {
#ifdef SF_HAS_COMPLEX_H
		    c += lt[im][i]*wave[im][j];
#else
		    c = sf_cadd(c,sf_cmul(lt[im][i], wave[im][j]));
#endif
		}
		curr[j] = c;
	    }
	}

#ifdef _OPENMP
#pragma omp parallel for private(ix,j)
#endif	 
	for ( ix =0 ; ix < gpl; ix++) {
	    j = (gpz+geop->top-1)+(ix+gpx+geop->lft)*nz2; /* padded grid */
	    rcd[ix][it] = curr[j];
	}
	
	if ( it%snpint == 0 ) {
#ifdef _OPENMP
#pragma omp parallel for private(ix,iz,j)
#endif
	    for ( ix = 0; ix < nx; ix++) {
		for ( iz = 0; iz<nz; iz++ ) { 
		    j = (iz+geop->top)+(ix+geop->lft)*nz2; /* padded grid */
		    wavfld[wfit][ix][iz] = curr[j];
		    if (illum) sill[ix][iz] += pow(hypotf(crealf(curr[j]),cimagf(curr[j])),2);
		}
	    }
	    wfit++;
	}
    } /*Main loop*/
    if (verb) sf_warning(".");
    cfft2_finalize();
    return wfit;
}

int lrosback2(sf_complex **img, sf_complex ***wavfld, float **sill, sf_complex **rcdf, sf_complex **rcd, bool adj,
	      bool verb, bool wantwf, sf_complex **lt, sf_complex **rt, int m2,
              geopar geop, int pad1, bool illum)
/*< low-rank one-step backward propagation + imaging >*/
{
    int it,iz,im,ik,ix,i,j;     /* index variables */
    int nxb,nzb,gpz,gpx,gpl,snpint,wfit;
    int nt,nz,nx, nk, nz2, nx2, nzx2;
    sf_complex c;
    sf_complex *cwave, *cwavem, *currm;
    sf_complex **wave, *curr;
    sf_complex **ccr;

    nx = geop->nx;
    nz = geop->nz;
    nxb = geop->nxb;
    nzb = geop->nzb;
/*     dx = geop->dx;
       dz = geop->dz; */
    
    gpz  = geop->gpz;
    gpx  = geop->gpx;
    gpl  = geop->gpl;
    snpint = geop->snpint;
    
    nt = geop->nt;
    /* dt = geop->dt; */

    ccr = sf_complexalloc2(nz, nx);

    nk = cfft2_init(pad1,nzb,nxb,&nz2,&nx2);
    /* nzx = nzb*nxb; */
    nzx2 = nz2*nx2;

    curr = sf_complexalloc(nzx2);
    cwave  = sf_complexalloc(nk);
    wave = sf_complexalloc2(nzx2,m2);

    if (!adj) {
	currm  = sf_complexalloc(nzx2);
	icfft2_allocate(cwave);
    } else {
	cwavem = sf_complexalloc(nk);
	icfft2_allocate(cwavem);
    }

#ifdef _OPENMP
#pragma omp parallel for private(iz)
#endif
    for (iz=0; iz < nzx2; iz++) {
	curr[iz] = sf_cmplx(0.,0.);
    }

#ifdef _OPENMP
#pragma omp parallel for private(ix, iz)
#endif
    for (ix = 0; ix < nx; ix++) {
	for (iz = 0; iz < nz; iz++) {
	  ccr[ix][iz] = sf_cmplx(0.,0.);
	}
    }

    /* migration */
      /* step backward in time (PSPI) */
      /*Main loop*/
      wfit = (int)(nt-1)/snpint;
      for (it = nt-1; it>=0; it--) {
	if  (verb) sf_warning("Backward receiver it=%d/%d;", it, nt-1);
#ifdef _OPENMP
#pragma omp parallel for private(ix,j)
#endif
        for (ix=0; ix<gpl; ix++)  {
	  j = (gpz+geop->top)+(ix+gpx+geop->lft)*nz2; /* padded grid */
#ifdef SF_HAS_COMPLEX_H
	if(it>240)
	  curr[j]+=rcdf[ix][it]-rcd[ix][it]; /* data injection */
#else
	if(it>240)
	  curr[j]=sf_cadd(curr[j],rcdf[ix][it]-rcd[ix][it]);
#endif
	}
	/*matrix multiplication*/
	cfft2(curr,cwave);
	for (im = 0; im < m2; im++) {
#ifdef _OPENMP
#pragma omp parallel for private(ik)
#endif
	  for (ik = 0; ik < nk; ik++) {
#ifdef SF_HAS_COMPLEX_H
	    cwavem[ik] = cwave[ik]*rt[ik][im];
#else
	    cwavem[ik] = sf_cmul(cwave[ik],rt[ik][im]);
#endif
	  }
	  icfft2(wave[im],cwavem);
	}
	
#ifdef _OPENMP
#pragma omp parallel for private(ix,iz,i,j,im,c) shared(curr,lt,wave)
#endif
	for (ix = 0; ix < nxb; ix++) {
	  for (iz=0; iz < nzb; iz++) {
	    i = iz+ix*nzb;  /* original grid */
	    j = iz+ix*nz2; /* padded grid */
	    c = sf_cmplx(0.,0.); // initialize
	    for (im = 0; im < m2; im++) {
#ifdef SF_HAS_COMPLEX_H
	      c += lt[im][i]*wave[im][j];
#else
	      c = sf_cadd(c,sf_cmul(lt[im][i], wave[im][j]));
#endif
	    }
	    curr[j] = c;
	  }
	}

	/*cross-correlation imaging condition*/
	if (it%snpint == 0 ) {
#ifdef _OPENMP
#pragma omp parallel for private(ix,iz,j)
#endif
	  for (ix=0; ix<nx; ix++) {
	    for (iz=0; iz<nz; iz++) {
	      j = (iz+geop->top)+(ix+geop->lft)*nz2; /* padded grid */
#ifdef SF_HAS_COMPLEX_H
	      ccr[ix][iz] += conjf(wavfld[wfit][ix][iz])*curr[j];
#else
	      ccr[ix][iz] = sf_cadd(ccr[ix][iz],sf_cmul(conjf(wavfld[wfit][ix][iz]),curr[j]));
#endif
	    }
	  }
	  wfit--;
	}
      } /*Main loop*/
      if (verb) sf_warning(".");
#ifdef _OPENMP
#pragma omp parallel for private(ix, iz)
#endif    
      for (ix=0; ix<nx; ix++) {
	for (iz=0; iz<nz; iz++) {
	  if (illum) {
#ifdef SF_HAS_COMPLEX_H
	    img[ix][iz] = ccr[ix][iz]/(sill[ix][iz]+SF_EPS);
#else
	    img[ix][iz] = sf_crmul(ccr[ix][iz],1./(sill[ix][iz]+SF_EPS));
#endif
	  } else img[ix][iz] = ccr[ix][iz];
	}
      } 

    cfft2_finalize();
    return 0;
}

/*******************************************************/
/* main function */
int main(int argc, char* argv[]) 
{
    clock_t tstart,tend;
    double duration;

    /*flags*/
    bool verb, adj; /* migration(adjoint) flag */
    bool wantwf; /* outputs wavefield snapshots */
    bool wantrecord; /* actually means "need record" */
    bool illum; /* source illumination flag*/
    bool roll; /* survey strategy */
    
    /*I/O*/
    sf_file Fvel;
    sf_file left, right, leftb, rightb;
    sf_file Fsrc, Frcd/*source and record*/;
    sf_file Ftmpwf;
    sf_file Fimg;

	sf_file Frcd1; /* for forward test*/

    /*axis*/
    sf_axis at, ax, az, as;

    /*grid index variables*/
    int nx, nz, nt, wfnt;
    int nzx, nx2, nz2, n2, m2, m2b, pad1, nk;
    int ix, iz, it, is;
    int nxb, nzb;
    int snpint;
    float dt, dx, dz, wfdt;
    float ox, oz;

    /*source/geophone location*/
    int   spx, spz;
    int   gpz,gpx,gpl; /*geophone depth/x-crd/length*/

    /*Model*/
    sf_complex **lt, **rt;
    sf_complex **ltb, **rtb;

    /*Data*/
    sf_complex ***wavefld;
    sf_complex ***record, **tmprec, **tmprecf, **img, **imgsum;
    float **sill;

    /*source*/
    sf_complex *ww;
    float *rr;
    int rectz,rectx,repeat; /*smoothing parameters*/
    float trunc;
    int sht0,shtbgn,shtend,shtnum,shtnum0,shtint,shtcur;

    /*abc boundary*/
    int top,bot,lft,rht;

    /*tmp*/
    int tmpint;

    /*parameter structs*/
    geopar geop;
    mpipar mpip;

    /*MPI*/
    int rank, nodes;
    sf_complex *sendbuf, *recvbuf;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);

    sf_init(argc, argv);

    if(rank==0) sf_warning("nodes=%d",nodes);

    if (!sf_getbool("verb", &verb)) verb=false; /*verbosity*/
    if (!sf_getbool("adj", &adj)) adj=true; /*migration*/
    if (!sf_getbool("wantwf", &wantwf)) wantwf=false; /*output forward and backward wavefield*/
    if (!sf_getbool("wantrecord", &wantrecord)) wantrecord=true; /*if n, using record data generated by this program */
    if (!sf_getbool("illum", &illum)) illum=false; /*if n, no source illumination applied */
    if (!sf_getbool("roll", &roll)) roll=false; /*if n, receiver is independent of source location and gpl=nx*/
    /* source/receiver info */
    if (!sf_getint("shtbgn", &shtbgn)) sf_error("Need shot starting location on grid!");
    if (!sf_getint("sht0", &sht0)) sht0=shtbgn; /*actual shot origin on grid*/
    if (!sf_getint("shtend", &shtend)) sf_error("Need shot ending location on grid!");
    if (!sf_getint("shtint", &shtint)) sf_error("Need shot interval on grid!");
    shtnum = (int)((shtend-shtbgn)/shtint) + 1;
    shtnum0 = shtnum;
    if (!sf_getint("spz", &spz)) sf_error("Need source depth!");
    if (!sf_getint("gpz", &gpz)) sf_error("Need receiver depth!");
    if (roll) if (!sf_getint("gpl", &gpl)) sf_error("Need receiver length");
    if (!sf_getint("snapinter", &snpint)) snpint=1;     /* snap interval */
    /*--- parameters of source ---*/
    if (!sf_getfloat("srctrunc", &trunc)) trunc=0.4;
    if (!sf_getint("rectz", &rectz)) rectz=2;
    if (!sf_getint("rectx", &rectx)) rectx=2;
    if (!sf_getint("repeat", &repeat)) repeat=1;
    /* abc parameters */
    if (!sf_getint("top", &top)) top=40;
    if (!sf_getint("bot", &bot)) bot=40;
    if (!sf_getint("lft", &lft)) lft=40;
    if (!sf_getint("rht", &rht)) rht=40;

    /*Set I/O file*/
	Frcd = sf_input("rcd"); /*record from elsewhere*/
	Fsrc  = sf_input("src");   /*source wavelet*/      
    Fimg  = sf_output("img");
    left  = sf_input("left");
    right = sf_input("right");
    leftb  = sf_input("leftb");
    rightb = sf_input("rightb");
    Fvel  = sf_input("vel");  /*velocity - just for model dimension*/
    if (wantwf) {
	Ftmpwf  = sf_output("tmpwf");/*wavefield snap*/
    } else {
	Ftmpwf  = NULL;
    }
    Frcd1 = sf_output("rcd1"); //for test

    /*--- Axes parameters ---*/
    at = sf_iaxa(Fsrc, 1); nt = sf_n(at);  dt = sf_d(at);      
    az = sf_iaxa(Fvel, 1); nzb = sf_n(az); dz = sf_d(az); oz = sf_o(az);
    ax = sf_iaxa(Fvel, 2); nxb = sf_n(ax); dx = sf_d(ax); ox = sf_o(ax);
    nzx = nzb*nxb;
    nz = nzb - top - bot;
    nx = nxb - lft - rht;
    if (!roll) gpl = nx; /* global survey setting */
    /* wavefield axis */
    wfnt = (int)(nt-1)/snpint+1;
    wfdt = dt*snpint;

    /* propagator matrices */
    if (!sf_getint("pad1",&pad1)) pad1=1; /* padding factor on the first axis */
    nz2 = kiss_fft_next_fast_size(nzb*pad1);
    nx2 = kiss_fft_next_fast_size(nxb);
    nk = nz2*nx2; /*wavenumber*/
    if (!sf_histint(left,"n1",&n2) || n2 != nzx) sf_error("Need n1=%d in left",nzx);
    if (!sf_histint(left,"n2",&m2))  sf_error("Need n2= in left");
    if (!sf_histint(right,"n1",&n2) || n2 != m2) sf_error("Need n1=%d in right",m2);
    if (!sf_histint(right,"n2",&n2) || n2 != nk) sf_error("Need n2=%d in right",nk);

    if (!sf_histint(leftb,"n1",&n2) || n2 != nzx) sf_error("Need n1=%d in left",nzx);
    if (!sf_histint(leftb,"n2",&m2b))  sf_error("Need n2= in left");
    if (!sf_histint(rightb,"n1",&n2) || n2 != m2b) sf_error("Need n1=%d in right",m2b);
    if (!sf_histint(rightb,"n2",&n2) || n2 != nk) sf_error("Need n2=%d in right",nk);

    /*check record data*/
    if (adj && wantrecord){
	sf_histint(Frcd,"n1", &tmpint);
	if (tmpint != nt ) sf_error("Error parameter n1 in record!");
	sf_histint(Frcd,"n2", &tmpint);
	if (tmpint != gpl ) sf_error("Error parameter n2 in record!");
	sf_histint(Frcd,"n3", &tmpint);
	if (tmpint != shtnum0 ) sf_error("Error parameter n3 in record!");
    }

    /*allocate memory*/
    ww=sf_complexalloc(nt);
    rr=sf_floatalloc(nzx);
    lt = sf_complexalloc2(nzx,m2);
    rt = sf_complexalloc2(m2,nk);
    ltb = sf_complexalloc2(nzx,m2b);
    rtb = sf_complexalloc2(m2b,nk);
    geop = (geopar) sf_alloc(1, sizeof(*geop));
    mpip = (mpipar) sf_alloc(1, sizeof(*mpip));
    tmprec = sf_complexalloc2(nt, gpl);
    tmprecf = sf_complexalloc2(nt, gpl);
    if (shtnum%nodes!=0) {
      shtnum += nodes-shtnum%nodes;
      if (verb) sf_warning("Total shot number is not divisible by total number of nodes! shunum padded to %d.", shtnum);
    }
    if (rank==0) {
      record = sf_complexalloc3(nt, gpl, shtnum);
    } else record = NULL;
    wavefld = sf_complexalloc3(nz, nx, wfnt);
    if (illum) sill = sf_floatalloc2(nz, nx);
    else sill = NULL;
    img = sf_complexalloc2(nz, nx);
    if (adj) {
      imgsum = sf_complexalloc2(nz, nx);
#ifdef _OPENMP
#pragma omp parallel for private(ix,iz)
#endif
	for (ix=0; ix<nx; ix++)
	  for (iz=0; iz<nz; iz++)
	    imgsum[ix][iz] = sf_cmplx(0.,0.);
    }
    /*read from files*/
    sf_complexread(ww,nt,Fsrc);
    sf_complexread(lt[0],nzx*m2,left);
    sf_complexread(rt[0],m2*nk,right);
    sf_complexread(ltb[0],nzx*m2b,leftb);
    sf_complexread(rtb[0],m2b*nk,rightb);
    if (rank==0) {
	sf_complexread(record[0][0], shtnum0*gpl*nt, Frcd);
	if (shtnum0%nodes!=0) {
#ifdef _OPENMP
#pragma omp parallel for private(is,ix,it)
#endif
	  for (is=shtnum0; is<shtnum; is++)
	    for (ix=0; ix<gpl; ix++)
	      for (it=0; it<nt; it++)
		record[is][ix][it] = sf_cmplx(0.,0.);
	}
    }
    
    /*close RSF files*/
    sf_fileclose(Fsrc);
    sf_fileclose(left);
    sf_fileclose(right);
    sf_fileclose(leftb);
    sf_fileclose(rightb);
	sf_fileclose(Frcd);

    /*load constant geopar elements*/
    mpip->cpuid=rank;
    mpip->numprocs=nodes;
    /*load constant geopar elements*/
    geop->nx  = nx;
    geop->nz  = nz;
    geop->nxb = nxb;
    geop->nzb = nzb;
    geop->dx  = dx;
    geop->dz  = dz;
    geop->ox  = ox;
    geop->oz  = oz;
    geop->snpint = snpint;
    geop->spz = spz;
    geop->gpz = gpz;
    geop->gpl = gpl;
    geop->top = top;
    geop->bot = bot;
    geop->lft = lft;
    geop->rht = rht;
    geop->nt = nt;
    geop->dt = dt;
    geop->trunc = trunc;
    geop->shtnum = shtnum;

    /* output RSF files */

    if (rank==0) {
      sf_setn(ax, gpl);
      sf_setn(az, nz);
      as = sf_iaxa(Fvel, 2);
      sf_setn(as,shtnum0);
      sf_setd(as,shtint*dx);
      sf_seto(as,shtbgn*dx+ox);
      
	sf_setn(ax, nx);
	/*write image*/
	sf_oaxa(Fimg, az, 1);
	sf_oaxa(Fimg, ax, 2);
	sf_settype(Fimg,SF_COMPLEX);
      
      if (wantwf) {
	sf_setn(ax, nx);
	/*write temp wavefield */
	sf_setn(at, wfnt);
	sf_setd(at, wfdt);
	
	sf_oaxa(Ftmpwf, az, 1);
	sf_oaxa(Ftmpwf, ax, 2);
	sf_oaxa(Ftmpwf, at, 3);
	sf_settype(Ftmpwf,SF_COMPLEX);

	sf_oaxa(Frcd1, ax, 2);
	sf_oaxa(Frcd1, at, 1);
	sf_settype(Frcd1,SF_COMPLEX);
      }
    }
    
    tstart = clock();

    for (is=0; is*nodes<shtnum; is++){

      shtcur = is*nodes+rank; // current shot index

      if (shtcur<shtnum0) {
	spx = shtbgn + shtint*(shtcur);
	if (roll)
	  gpx = spx - (int)(gpl/2);
	else
	  gpx = 0;
	geop->spx = spx;
	geop->gpx = gpx;
	
	if (verb) {
	  sf_warning("============================");
	  sf_warning("processing shot #%d", shtcur);
	  sf_warning("nx=%d nz=%d nt=%d", geop->nx, geop->nz, geop->nt);
	  sf_warning("nxb=%d nzb=%d ", geop->nxb, geop->nzb);
	  sf_warning("dx=%f dz=%f dt=%f", geop->dx, geop->dz, geop->dt);
	  sf_warning("top=%d bot=%d lft=%d rht=%d", geop->top, geop->bot, geop->lft, geop->rht);
	  sf_warning("rectz=%d rectx=%d repeat=%d srctrunc=%f",rectz,rectx,repeat,geop->trunc);
	  sf_warning("spz=%d spx=%d gpz=%d gpx=%d gpl=%d", spz, spx, gpz, gpx, gpl);
	  sf_warning("snpint=%d wfdt=%f wfnt=%d ", snpint, wfdt, wfnt);
	  sf_warning("sht0=%d shtbgn=%d shtend=%d shtnum0=%d shtnum=%d", sht0, shtbgn, shtend, shtnum0, shtnum);
	  if (roll) sf_warning("Rolling survey!");
	  else sf_warning("Global survey (gpl=nx)!");
	  if (illum) sf_warning("Using source illumination!");
	  else sf_warning("No source illumination!");
	  sf_warning("============================");
	}
	
	/*generate reflectivity map*/
	reflgen(nzb, nxb, spz+top, spx+lft, rectz, rectx, repeat, rr);
	
	lrosfor2(wavefld, sill, tmprec, verb, lt, rt, m2, geop, ww, rr, pad1, illum);

		sf_complexwrite(tmprec[0], nx*nt, Frcd1);
		sf_fileclose(Frcd1);
      }

      if(adj && wantrecord) {
	if (rank==0) sendbuf = record[is*nodes][0];
	else sendbuf = NULL;
	recvbuf = tmprecf[0];
	MPI_Scatter(sendbuf, gpl*nt, MPI_COMPLEX, recvbuf, gpl*nt, MPI_COMPLEX, 0, MPI_COMM_WORLD); // tmprecf[ix][it] = record[is][ix][it];
      }
      
      if (shtcur<shtnum0) {
	lrosback2(img, wavefld, sill, tmprecf, tmprec, adj, verb, wantwf, ltb, rtb, m2b, geop, pad1, illum);
	if (adj) { /*local image reduction*/
#ifdef _OPENMP
#pragma omp parallel for private(ix,iz)
#endif
	  for (ix=0; ix<nx; ix++) {
	    for (iz=0; iz<nz; iz++) {
#ifdef SF_HAS_COMPLEX_H
	      imgsum[ix][iz] += img[ix][iz];
#else
	      imgsum[ix][iz] = sf_cadd(imgsum[ix][iz],img[ix][iz]);
#endif      
	    }
	  }
	}
      }

    if (wantwf && shtcur==0)
		sf_complexwrite(wavefld[0][0], wfnt*nx*nz, Ftmpwf);
    } /*shot iteration*/

    MPI_Barrier(MPI_COMM_WORLD);
    /*write record/image*/
    if (adj) {
      if (rank==0) {
#if MPI_VERSION >= 2
	sendbuf = (sf_complex *) MPI_IN_PLACE;
#else /* will fail */
	sendbuf = NULL;
#endif 
	recvbuf = imgsum[0];
      } else {
	sendbuf = imgsum[0];
      	recvbuf = NULL;
      }
      MPI_Reduce(sendbuf, recvbuf, nx*nz, MPI_COMPLEX, MPI_SUM, 0, MPI_COMM_WORLD); 
      if (rank==0)
	sf_complexwrite(imgsum[0], nx*nz, Fimg);
    }

	sf_fileclose(Fimg);
	sf_fileclose(Ftmpwf);
	sf_fileclose(Fvel);

    /*free memory*/
    free(ww); free(rr);
    free(*lt); free(lt);
    free(*rt); free(rt);
    free(*ltb);free(ltb);
    free(*rtb);free(rtb);
    free(geop);free(mpip);
    free(*tmprec); free(tmprec);
    free(*tmprecf); free(tmprecf);
    if (rank==0) {free(**record); free(*record); free(record);}
    free(**wavefld); free(*wavefld); free(wavefld);
    if (illum) {
      free(*sill); free(sill);
    }
    free(*img); free(img);
    if (adj) {
      free(*imgsum); free(imgsum);
    }

    tend = clock();
    duration=(double)(tend-tstart)/CLOCKS_PER_SEC;
    sf_warning(">> The CPU time of single shot migration is: %f seconds << ", duration);

    MPI_Finalize();
    exit(0);
}

