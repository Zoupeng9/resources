// Complex lowrank decomposition for 2-D viscoacoustic isotropic wave propagation. 
// deprecated -- please use sfzfraclr2
//   Copyright (C) 2014 University of Texas at Austin
//  
//   This program is free software; you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation; either version 2 of the License, or
//   (at your option) any later version.
//  
//   This program is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//  
//   You should have received a copy of the GNU General Public License
//   along with this program; if not, write to the Free Software
//   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
#include <time.h>
#include <rsf.hh>

#include "vecmatop.hh"
#include "serialize.hh"

using namespace std;

static std::valarray<float> vs,qs;
static std::valarray<float> ksz,ksx;
static float ct,cb,cl,cr;
static int nkzs,nkxs,nz,nx,nbt,nbb,nbl,nbr;
static float w0,gama;
static bool compen,avg;
static int mode,sign,abc;

int sample(vector<int>& rs, vector<int>& cs, CpxNumMat& res)
{
    int nr = rs.size();
    int nc = cs.size();
    res.resize(nr,nc);  
    setvalue(res,cpx(0.0f,0.0f));
    for(int a=0; a<nr; a++) {
	for(int b=0; b<nc; b++) {
	    int ikz = cs[b] % nkzs;
	    int ikx = (int) cs[b]/nkzs;
	    int iz  = rs[a] % nz;
	    int ix  = (int) rs[a]/nz;
	    float hypk = hypot(ksz[ikz],ksx[ikx]) + SF_EPS;
	    float gamma = atanf(1./qs[rs[a]])/SF_PI;
	    float c0 = vs[rs[a]];
	    float c = c0*cosf(SF_PI*gamma/2.);
	    if (c<=0) sf_error("c negative!");
	    cpx phf;
	    if (mode == 0) 
		{ /*viscoacoustic*/
			float eta = -powf(c0,2.*gamma)*powf(w0,-2.*gamma)*cosf(SF_PI*gamma);
			float tao = -powf(c0,2.*gamma-1.)*powf(w0,-2.*gamma)*sinf(SF_PI*gamma);
			float cc = c0*cosf(SF_PI*gamma/2.);
			if (avg) gamma = gama;

			float logck = -0.5*powf(c0*hypk/w0,2)+2.*c0*hypk/w0 - 1.5; //logf(c0*hypk/w0)
			//float logck = logf(c0*hypk/w0);
			float temp1 = -0.5*powf(c0,2.*gamma+1)*powf(hypk,2.*gamma+1)*powf(w0,-2.*gamma)*cosf(SF_PI*gamma/2.)*cosf(SF_PI*gamma/2.);
			float temp2 = -SF_PI + 2.*SF_PI*cosf(SF_PI*gamma) + 2*logck*sinf(SF_PI*gamma);
		//	float temp3 = 0.5*c0*powf(hypk,1+gamma)*sqrtf(-eta)/cosf(SF_PI*gamma);
		//	float temp4 = (cosf(SF_PI*gamma/2.)+cosf(3.*SF_PI*gamma/2.))*logck - SF_PI*sinf(3.*SF_PI*gamma/2.);

			float temp7 = 0.5*powf(c0*hypk,2.*gamma+1.)*powf(w0,-4.*gamma)/sqrtf(-powf(tao*cc*cc,2)*powf(hypk,4.*gamma+2.)-4.*eta*cc*cc*powf(hypk,2.*gamma+2.));
			float temp8 = -2.*powf(c0*hypk,2.*gamma)*powf(cosf(SF_PI*gamma/2.),5)*sinf(SF_PI*gamma)*temp2;
			float temp9 = powf(w0,2.*gamma)*(logck+(2*cosf(SF_PI*gamma)+cosf(2.*SF_PI*gamma))*logck - SF_PI*(sinf(SF_PI*gamma)+sinf(2.*SF_PI*gamma)));


			float drdq = -1.0/(SF_PI*(1+qs[rs[a]]*qs[rs[a]]));

	//		phf = cpx(temp1*temp2*drdq,temp3*temp4*drdq);
			phf = cpx(temp1*temp2*drdq,temp7*(temp8+temp9)*drdq);

		//	float temp5 = -0.5*(2.*gamma+1.)*powf(c0,2.*gamma)*powf(w0,-2.*gamma)*cosf(SF_PI*gamma/2.)*cosf(SF_PI*gamma/2.)*powf(hypk,2.*gamma+1)*sinf(SF_PI*gamma);
		//	float temp6 = (gamma+1.)*powf(c0,gamma)*powf(w0,-gamma)*sqrtf(cosf(SF_PI*gamma))*cosf(SF_PI*gamma/2.)*powf(hypk,gamma+1);
		//	phf = cpx(temp5,temp6);
		//    phf = cpx(1,0);
	    }

	    /* absorbing boundary */
            if (abc==0) {
              if (iz < nbt)
		phf *= exp(-powf(ct*(nbt-iz)*abs(ksz[ikz]/hypk),2));
              else if (iz > nz-1-nbb)
		phf *= exp(-powf(cb*(iz-nz+1+nbb)*abs(ksz[ikz]/hypk),2));
              if (ix < nbl)
		phf *= exp(-powf(cl*(nbl-ix)*abs(ksx[ikx]/hypk),2));
              else if (ix > nx-1-nbr)
		phf *= exp(-powf(cr*(ix-nx+1+nbr)*abs(ksx[ikx]/hypk),2));
            } else {
              if (iz < nbt)
		phf *= exp(-powf(ct*(nbt-iz),2));
              else if (iz > nz-1-nbb)
		phf *= exp(-powf(cb*(iz-nz+1+nbb),2));
              if (ix < nbl)
		phf *= exp(-powf(cl*(nbl-ix),2));
              else if (ix > nx-1-nbr)
		phf *= exp(-powf(cr*(ix-nx+1+nbr),2));
            }
	    res(a,b) = phf;
	}
    }
    return 0;
}

int tukey(float a, float cutoff, float vm, vector<float>& tuk)
{
    float kmax = hypot(ksz[nkzs-1],ksx[nkxs-1]);
    float kbond;
    kbond = 2*SF_PI*cutoff/vm;
    if (kbond > kmax) {
        sf_warning("cutoff wavenumber %f larger than maximum wavenumber %f! Setting kbond = kmax...",kbond,kmax);
	kbond = kmax;
    }
    for (int ikx=0; ikx<nkxs; ikx++) {
	for (int ikz=0; ikz<nkzs; ikz++) {
   	    float k = hypot(ksz[ikz],ksx[ikx]);
	    int ik = ikz+ikx*nkzs;
	    if (k > kbond)
		tuk[ik] = 0.;
	    else if (k >= kbond*(1.-0.5*a))
		tuk[ik] = 0.5*(1.+cos(SF_PI*(2.*k/(a*kbond)-2./a+1.)));
	    else
		tuk[ik] = 1.;
	}
    }
    return 0;
}

int main(int argc, char** argv)
{   
    sf_init(argc,argv); // Initialize RSF

    iRSF par(0);
    int seed;

    par.get("seed",seed,time(NULL)); // seed for random number generator
    srand48(seed);

    float eps;
    par.get("eps",eps,1.e-4); // tolerance

    int npk;
    par.get("npk",npk,20); // maximum rank

    par.get("w0",w0); // reference frequency
    w0 *= 2*SF_PI;
    
    par.get("mode",mode,0); // mode of propagation: 0 is viscoacoustic (default); 1 is loss-dominated; 2 is dispersion dominated; 3 is acoustic
    float aa,cut,vmax;
    par.get("compen",compen,false); // compensate attenuation, only works if mode=0,1 (viscoacoustic)
    if (mode==0 || mode==1)
	if (compen) {
	    par.get("taper",aa,0.2); // taper ratio for tukey window
	    par.get("cutoff",cut,250.); // cutoff frequency
	    par.get("vmax",vmax,6000.); // maximum velocity
	    sf_warning("Compensating for attenuation!");
	}
    par.get("sign",sign,0); // sign of solution: 0 is positive, 1 is negative
    par.get("avg",avg,false); // whether use average value of gamma
    if (avg) {
	par.get("gamma",gama);
	sf_warning("Gamma_avg = %f",gama);
    }
    
    par.get("abc",abc,0); /*absorbing mode: 0-> direction dependent; 1-> direction independent.*/

    par.get("nbt",nbt,0);
    par.get("nbb",nbb,0);
    par.get("nbl",nbl,0);
    par.get("nbr",nbr,0);

    par.get("ct",ct,0.0);
    par.get("cb",cb,0.0);
    par.get("cl",cl,0.0);
    par.get("cr",cr,0.0);

    iRSF vel, q("q");

    vel.get("n1",nz);
    vel.get("n2",nx);
    int m = nx*nz;

    vs.resize(m);
    qs.resize(m);

    vel >> vs;
    q >> qs;

    iRSF fft("fft");

    int nkz,nkx;
    fft.get("n1",nkz);
    fft.get("n2",nkx);
    nkzs = nkz;
    nkxs = nkx;

    float dkz,dkx;
    fft.get("d1",dkz);
    fft.get("d2",dkx);
    
    float kz0,kx0;
    fft.get("o1",kz0);
    fft.get("o2",kx0);

    int n = nkx*nkz;

    std::valarray<float> kz(nkz),kx(nkx);
    for (int iz=0; iz < nkz; iz++)
	kz[iz] = 2*SF_PI*(kz0+iz*dkz);
    for (int ix=0; ix < nkx; ix++)
	kx[ix] = 2*SF_PI*(kx0+ix*dkx);

    ksz.resize(nkz);
    ksz = kz;
    ksx.resize(nkx);
    ksx = kx;

    vector<int> lidx, ridx;
    CpxNumMat mid;

    iC( lowrank(m,n,sample,eps,npk,lidx,ridx,mid) );

    int n2=mid.n();
    int m2=mid.m();

    vector<int> midx(m), nidx(n);
    for (int k=0; k < m; k++) 
	midx[k] = k;
    for (int k=0; k < n; k++) 
	nidx[k] = k;    

    CpxNumMat lmat(m,m2);
    iC ( sample(midx,lidx,lmat) );

    CpxNumMat lmat2(m,n2);
    iC( zgemm(1.0, lmat, mid, 0.0, lmat2) );

    cpx *ldat = lmat2.data();
    std::valarray<sf_complex> ldata(m*n2);
    for (int k=0; k < m*n2; k++) {
	ldata[k] = sf_cmplx(real(ldat[k]),imag(ldat[k]));
    }

    oRSF left("left");
    left.type(SF_COMPLEX);
    left.put("n1",m);
    left.put("n2",n2);
    left << ldata;

    CpxNumMat rmat(n2,n);
    iC ( sample(ridx,nidx,rmat) );

    cpx *rdat = rmat.data();
    std::valarray<sf_complex> rdata(n2*n);    
    if (!compen || mode==2 || mode==3) {
	for (int k=0; k < n2*n; k++) {
	    rdata[k] = sf_cmplx(real(rdat[k]),imag(rdat[k]));
	} 
    } else {
	vector<float> lpass(n);
	tukey(aa, cut, vmax, lpass);
	for (int k1=0; k1 < n; k1++) {
	    for (int k2=0; k2 < n2; k2++) {
		int k = k2 + k1*n2;
		rdata[k] = sf_cmplx(real(rdat[k])*lpass[k1],imag(rdat[k])*lpass[k1]);
	    }
	}
    }

    oRSF right;
    right.type(SF_COMPLEX);
    right.put("n1",n2);
    right.put("n2",n);
    right << rdata;

    exit(0);
}
