/* Visco-acoustic forward modeling */
/*
 Copyright (C) 2017 Tongji University, Shanghai, China
 Author: Peng Zou
 
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
#include <omp.h>
#include "derivative.h"
#include "Qfwi_commons.h"
/*^*/

void forward_cpml(float ***rcd, float ***dd, float *fcost, float **sillum,  sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic forward modeling >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **tau, **taus, **K0, **dK;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **p1, **p2;
	float **rp1, **rp2;
	float **phi_vx1, **phi_vx2, **phi_vz1, **phi_vz2;
	float **phi_px1, **phi_px2, **phi_pz1, **phi_pz2;
	float **dpx, **dpz, **dvx, **dvz;

	float **dvx1, **dvx2, **dvz1, **dvz2;
	float **dp1, **dp2;
	float **drp1, **drp2;
	float **dphi_vx1, **dphi_vx2, **dphi_vz1, **dphi_vz2;
	float **dphi_px1, **dphi_px2, **dphi_pz1, **dphi_pz2;
	float **ddvx, **ddvz;

	float kax, kaz, alphax_max, alphaz_max, d0x, d0z;
	float **alphax, **dx1, **bx, **ax1;
	float **alphaz, **dz1, **bz, **az1; //CMPL variables

    float **wave, **dwave; //for save K0(dvx/dx+dvz/dz) and K0(ddvx/dx+ddvz/dz)
	FILE *fwave, *fdwave;
	int interval, wnt, wit;

	float fcost1 = 0.0;

	MPI_Comm comm=MPI_COMM_WORLD;

	fwave=fopen("wave", "wb+");
	fdwave=fopen("dwave", "wb+");

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	float ***rcd1, **sillum1; // temporary variables for MPI_Reduce
	rcd1=sf_floatalloc3(nt,nr,acpar->ns);
	sillum1=sf_floatalloc2(nz,nx);
	memset(sillum1[0], 0., nx*nz*sizeof(float));

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

	interval = acpar->interval;
	wnt = (nt-1)/interval+1;
	wave=sf_floatalloc2(nz,nx);
	dwave=sf_floatalloc2(nz,nx);

	vv = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	K0=sf_floatalloc2(padnz, padnx);
	dK=sf_floatalloc2(padnz, padnx);

	p1=sf_floatalloc2(padnz, padnx);
	p2=sf_floatalloc2(padnz, padnx);
	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);
	rp1=sf_floatalloc2(padnz, padnx);
	rp2=sf_floatalloc2(padnz, padnx);

	phi_vx1=sf_floatalloc2(padnz, padnx);
	phi_vx2=sf_floatalloc2(padnz, padnx);
	phi_vz1=sf_floatalloc2(padnz, padnx);
	phi_vz2=sf_floatalloc2(padnz, padnx);
	phi_px1=sf_floatalloc2(padnz, padnx);
	phi_px2=sf_floatalloc2(padnz, padnx);
	phi_pz1=sf_floatalloc2(padnz, padnx);
	phi_pz2=sf_floatalloc2(padnz, padnx);

	dp1=sf_floatalloc2(padnz, padnx);
	dp2=sf_floatalloc2(padnz, padnx);
	dvx1=sf_floatalloc2(padnz, padnx);
	dvx2=sf_floatalloc2(padnz, padnx);
	dvz1=sf_floatalloc2(padnz, padnx);
	dvz2=sf_floatalloc2(padnz, padnx);
	drp1=sf_floatalloc2(padnz, padnx);
	drp2=sf_floatalloc2(padnz, padnx);

	dphi_vx1=sf_floatalloc2(padnz, padnx);
	dphi_vx2=sf_floatalloc2(padnz, padnx);
	dphi_vz1=sf_floatalloc2(padnz, padnx);
	dphi_vz2=sf_floatalloc2(padnz, padnx);
	dphi_px1=sf_floatalloc2(padnz, padnx);
	dphi_px2=sf_floatalloc2(padnz, padnx);
	dphi_pz1=sf_floatalloc2(padnz, padnx);
	dphi_pz2=sf_floatalloc2(padnz, padnx);

	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);
	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);
	ddvx=sf_floatalloc2(padnz, padnx);
	ddvz=sf_floatalloc2(padnz, padnx);

	rr=sf_floatalloc(padnzx);

	/* PML boundary condition*/
	alphax=sf_floatalloc2(padnz, padnx);
	dx1=sf_floatalloc2(padnz, padnx);
	bx=sf_floatalloc2(padnz, padnx);
	ax1=sf_floatalloc2(padnz, padnx);
	alphaz=sf_floatalloc2(padnz, padnx);
	dz1=sf_floatalloc2(padnz, padnx);
	bz=sf_floatalloc2(padnz, padnx);
	az1=sf_floatalloc2(padnz, padnx);
	kax = 1.;
	kaz = 1.;
	alphax_max = SF_PI*20.;
	alphaz_max = SF_PI*20.;
	d0x = -3.8*4.500*log(0.001)/2./nb/dx;
	d0z = -3.8*4.500*log(0.001)/2./nb/dz;
	for(ix=0;ix<padnx;ix++)
		for(iz=0;iz<padnz;iz++)
		{
		   alphax[ix][iz] = 0.0;
		   alphaz[ix][iz] = 0.0;
		   dx1[ix][iz] = 0.0;
		   dz1[ix][iz] = 0.0;
		   bx[ix][iz] = 0.0;
		   bz[ix][iz] = 0.0;
		   ax1[ix][iz] = 0.0;
		   az1[ix][iz] = 0.0;

           if(ix<nb)
           {
               alphax[ix][iz] = alphax_max*ix/nb;
               dx1[ix][iz] = d0x*pow((nb-ix)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.000001)*(bx[ix][iz]-1.);
           }
           if(ix>padnx-nb)
           {
               alphax[ix][iz] = alphax_max*(padnx-ix)/nb;
               dx1[ix][iz] = d0x*pow((ix+nb-padnx)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.000001)*(bx[ix][iz]-1.);
           }
           if(iz<nb)
           {
               alphaz[ix][iz] = alphaz_max*iz/nb;
               dz1[ix][iz] = d0z*pow((nb-iz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.000001)*(bz[ix][iz]-1.);
           }
           if(iz>padnz-nb)
           {
               alphaz[ix][iz] = alphaz_max*(padnz-iz)/nb;
               dz1[ix][iz] = d0x*pow((iz+nb-padnz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.000001)*(bz[ix][iz]-1.);
           }
           bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
           bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
      }

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->dvv, dK, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(ix=0; ix<padnx; ix++)
		for(iz=0; iz<padnz; iz++)
		{
			K0[ix][iz] = vv[ix][iz]*vv[ix][iz];
			dK[ix][iz] = 2.0*vv[ix][iz]*dK[ix][iz]+dK[ix][iz]*dK[ix][iz];
		}

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){
		sf_warning("###### is=%d ######", is+1);

		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(rp1[0], 0., padnzx*sizeof(float));
		memset(rp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		memset(phi_vx1[0], 0., padnzx*sizeof(float));
		memset(phi_vx2[0], 0., padnzx*sizeof(float));
		memset(phi_vz1[0], 0., padnzx*sizeof(float));
		memset(phi_vz2[0], 0., padnzx*sizeof(float));
		memset(phi_px1[0], 0., padnzx*sizeof(float));
		memset(phi_px2[0], 0., padnzx*sizeof(float));
		memset(phi_pz1[0], 0., padnzx*sizeof(float));
		memset(phi_pz2[0], 0., padnzx*sizeof(float));
		memset(dp1[0], 0., padnzx*sizeof(float));
		memset(dp2[0], 0., padnzx*sizeof(float));
		memset(drp1[0], 0., padnzx*sizeof(float));
		memset(drp2[0], 0., padnzx*sizeof(float));
		memset(dvx1[0], 0., padnzx*sizeof(float));
		memset(dvx2[0], 0., padnzx*sizeof(float));
		memset(dvz1[0], 0., padnzx*sizeof(float));
		memset(dvz2[0], 0., padnzx*sizeof(float));
		memset(dphi_vx1[0], 0., padnzx*sizeof(float));
		memset(dphi_vx2[0], 0., padnzx*sizeof(float));
		memset(dphi_vz1[0], 0., padnzx*sizeof(float));
		memset(dphi_vz2[0], 0., padnzx*sizeof(float));
		memset(dphi_px1[0], 0., padnzx*sizeof(float));
		memset(dphi_px2[0], 0., padnzx*sizeof(float));
		memset(dphi_pz1[0], 0., padnzx*sizeof(float));
		memset(dphi_pz2[0], 0., padnzx*sizeof(float));
		memset(dvx[0], 0., padnzx*sizeof(float));
		memset(dvz[0], 0., padnzx*sizeof(float));
		memset(dpx[0], 0., padnzx*sizeof(float));
		memset(dpz[0], 0., padnzx*sizeof(float));
		memset(ddvx[0], 0., padnzx*sizeof(float));
		memset(ddvz[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		wit=0;
		for(it=0; it<nt; it++){
			if(verb && it%200==0) sf_warning("Forward modeling is=%d; it=%d;", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				rcd1[is][acpar->r02[is]+ir][it]=dp2[rx][rz]+p2[rx][rz]-dd[is][acpar->r02[is]+ir][it];
				if(it<abs(sx-rx)*dx/vv[rx][sz]/dt+300+300./(0.1*abs(sx-rx)+1))
					rcd1[is][acpar->r02[is]+ir][it] = 0;
				fcost1 += 0.5*rcd1[is][acpar->r02[is]+ir][it]*rcd1[is][acpar->r02[is]+ir][it];
			}

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					rp2[ix][iz] = -dt/taus[ix][iz]*(rp1[ix][iz]+tau[ix][iz]*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]
								+phi_vz2[ix][iz]+phi_vx2[ix][iz])) + rp1[ix][iz];
					p2[ix][iz] = dt*((tau[ix][iz]+1)*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]+phi_vz2[ix][iz]
								+phi_vx2[ix][iz])+rp2[ix][iz])+p1[ix][iz] + rr[ix*padnz+iz]*array->ww[it];

					if(iz<nb || iz>=padnz-nb)
						phi_vz2[ix][iz] = bz[ix][iz]*phi_vz1[ix][iz] + az1[ix][iz]*dvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_vx2[ix][iz] = bx[ix][iz]*phi_vx1[ix][iz] + ax1[ix][iz]*dvx[ix][iz];
				}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					vz2[ix][iz] = dt*(dpz[ix][iz]+phi_pz2[ix][iz]) +vz1[ix][iz];
					vx2[ix][iz] = dt*(dpx[ix][iz]+phi_px2[ix][iz]) +vx1[ix][iz];
					if(iz<nb || iz>=padnz-nb)
						phi_pz2[ix][iz] = bz[ix][iz]*phi_pz1[ix][iz] + az1[ix][iz]*dpz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_px2[ix][iz] = bx[ix][iz]*phi_px1[ix][iz] + ax1[ix][iz]*dpx[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p1[ix][iz] = p2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					rp1[ix][iz] = rp2[ix][iz];
					phi_px1[ix][iz] = phi_px2[ix][iz];
					phi_pz1[ix][iz] = phi_pz2[ix][iz];
					phi_vx1[ix][iz] = phi_vx2[ix][iz];
					phi_vz1[ix][iz] = phi_vz2[ix][iz];
				}

			//Born modeling
			staggerFD1order2Ddx1(dvx2,ddvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(dvz2,ddvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					drp2[ix][iz] = -dt/taus[ix][iz]*(drp1[ix][iz]+tau[ix][iz]*((K0[ix][iz]/*+dK[ix][iz]*/)*(ddvz[ix][iz]+ddvx[ix][iz]
								+dphi_vz2[ix][iz]+dphi_vx2[ix][iz])+dK[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]
									+phi_vx2[ix][iz]+phi_vz2[ix][iz]))) + drp1[ix][iz];

					dp2[ix][iz] = dt*((tau[ix][iz]+1)*((K0[ix][iz]/*+dK[ix][iz]*/)*(ddvz[ix][iz]+ddvx[ix][iz]+dphi_vz2[ix][iz]
								+dphi_vx2[ix][iz])+dK[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]
									+phi_vx2[ix][iz]+phi_vz2[ix][iz]))+drp2[ix][iz])+dp1[ix][iz];

					/* acoustic */
					//dp2[ix][iz] = dt*(K0[ix][iz]*(ddvz[ix][iz]+ddvx[ix][iz]+dphi_vz2[ix][iz]+dphi_vx2[ix][iz])
					//		      +dK[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]+phi_vx2[ix][iz]+phi_vz2[ix][iz])) + dp1[ix][iz];

					if(iz<nb || iz>=padnz-nb)
						dphi_vz2[ix][iz] = bz[ix][iz]*dphi_vz1[ix][iz] + az1[ix][iz]*ddvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						dphi_vx2[ix][iz] = bx[ix][iz]*dphi_vx1[ix][iz] + ax1[ix][iz]*ddvx[ix][iz];
				}

			staggerFD1order2Ddx1(dp2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(dp2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					dvz2[ix][iz] = dt*(dpz[ix][iz]+dphi_pz2[ix][iz]) +dvz1[ix][iz];
					dvx2[ix][iz] = dt*(dpx[ix][iz]+dphi_px2[ix][iz]) +dvx1[ix][iz];
					if(iz<nb || iz>=padnz-nb)
						dphi_pz2[ix][iz] = bz[ix][iz]*dphi_pz1[ix][iz] + az1[ix][iz]*dpz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						dphi_px2[ix][iz] = bx[ix][iz]*dphi_px1[ix][iz] + ax1[ix][iz]*dpx[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					dp1[ix][iz] = dp2[ix][iz];
					dvx1[ix][iz] = dvx2[ix][iz];
					dvz1[ix][iz] = dvz2[ix][iz];
					drp1[ix][iz] = drp2[ix][iz];
					dphi_px1[ix][iz] = dphi_px2[ix][iz];
					dphi_pz1[ix][iz] = dphi_pz2[ix][iz];
					dphi_vx1[ix][iz] = dphi_vx2[ix][iz];
					dphi_vz1[ix][iz] = dphi_vz2[ix][iz];
				}

			if(it%interval==0)
			{
				for(ix=0; ix<nx; ix++)
					for(iz=0; iz<nz; iz++)
					{
						wave[ix][iz]=dvx[ix+nb][iz+nb]+dvz[ix+nb][iz+nb];
						dwave[ix][iz]=ddvx[ix+nb][iz+nb]+ddvz[ix+nb][iz+nb];
						sillum1[ix][iz] += p2[ix+nb][iz+nb]*p2[ix+nb][iz+nb];
					}
				fseeko(fwave, (is*wnt+wit)*nx*nz*sizeof(float), SEEK_SET);
				fseeko(fdwave, (is*wnt+wit)*nx*nz*sizeof(float), SEEK_SET);
				fwrite(wave[0],sizeof(float), nx*nz, fwave);
				fwrite(dwave[0],sizeof(float), nx*nz, fdwave);
				wit++;
			}

		} // end of time loop

		/* check */
		if(wit != wnt) sf_error("Incorrect number of wavefield snapshots");
		wit--;
	}// end of shot loop

	fclose(fwave);
	fclose(fdwave);
	MPI_Barrier(comm);

	/* record and sillum reduction */
	MPI_Reduce(&rcd1[0][0][0], &rcd[0][0][0], acpar->ns*nr*nt, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(&rcd[0][0][0], acpar->ns*nr*nt, MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	FILE *fpp = fopen("rcdd","wb");
		fwrite(&dd[0][0][0],sizeof(float),acpar->ns*nr*nt,fpp);
	fclose(fpp);

	MPI_Reduce(sillum1[0], sillum[0], nx*nz, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(sillum[0], nx*nz, MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	MPI_Reduce(&fcost1, fcost, 1,  MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(fcost, 1,  MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	/* release allocated memory */
	free(*wave); free(wave);
	free(*dwave); free(dwave);
	free(*p2); free(p2); 
	free(*p1); free(p1);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*rp2); free(rp2);
	free(*rp1); free(rp1);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*dp2); free(dp2); 
	free(*dp1); free(dp1);
	free(*dvx1); free(dvx1); free(*dvx2); free(dvx2);
	free(*dvz1); free(dvz1); free(*dvz2); free(dvz2);
	free(*drp2); free(drp2);
	free(*drp1); free(drp1);
	free(*ddvx); free(ddvx); free(*ddvz); free(ddvz);
	free(*phi_vx1); free(phi_vx1);
	free(*phi_vx2); free(phi_vx2);
	free(*phi_vz1); free(phi_vz1);
	free(*phi_vz2); free(phi_vz2);
	free(*phi_px1); free(phi_px1);
	free(*phi_px2); free(phi_px2);
	free(*phi_pz1); free(phi_pz1);
	free(*phi_pz2); free(phi_pz2);
	free(*dphi_vx1); free(dphi_vx1);
	free(*dphi_vx2); free(dphi_vx2);
	free(*dphi_vz1); free(dphi_vz1);
	free(*dphi_vz2); free(dphi_vz2);
	free(*dphi_px1); free(dphi_px1);
	free(*dphi_px2); free(dphi_px2);
	free(*dphi_pz1); free(dphi_pz1);
	free(*dphi_pz2); free(dphi_pz2);
	free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*K0); free(K0); free(*dK); free(dK);
	free(rr); 
	free(*alphax); free(alphax);
	free(*alphaz); free(alphaz);
	free(*dx1); free(dx1);
	free(*ax1); free(ax1);
	free(*bx); free(bx);
	free(*dz1); free(dz1);
	free(*az1); free(az1);
	free(*bz); free(bz);
	free(**rcd1); free(*rcd1); free(rcd1);
	free(*sillum1); free(sillum1);
	MPI_Barrier(comm);
}

void adjoint_cpml(float **grads, float **gradr, float ***rcd, float **sillum, sf_mpi *mpipar, 
		sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic adjoint Born  modeling with CPML boundary condition >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **tau, **taus, **K0, **dK;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **p1, **p2;
	float **rp1, **rp2;
	float **phi_vx1, **phi_vx2, **phi_vz1, **phi_vz2;
	float **phi_px1, **phi_px2, **phi_pz1, **phi_pz2;
	float **phi_rpx1, **phi_rpx2, **phi_rpz1, **phi_rpz2;
	float **dpx, **dpz, **dvx, **dvz;
	float **drpx, **drpz;

	float **dvx1, **dvx2, **dvz1, **dvz2;
	float **dp1, **dp2;
	float **drp1, **drp2;
	float **dphi_vx1, **dphi_vx2, **dphi_vz1, **dphi_vz2;
	float **dphi_px1, **dphi_px2, **dphi_pz1, **dphi_pz2;
	float **dphi_rpx1, **dphi_rpx2, **dphi_rpz1, **dphi_rpz2;
	float **ddpx, **ddpz, **ddvx, **ddvz;
	float **ddrpx, **ddrpz;

	float **dKp2, **dKp2x, **dKp2z;
	float **dKrp2, **dKrp2x, **dKrp2z;
	float **phi_dKp2x1, **phi_dKp2x2, **phi_dKp2z1, **phi_dKp2z2;
	float **phi_dKrp2x1, **phi_dKrp2x2, **phi_dKrp2z1, **phi_dKrp2z2; //for born modeling

	float kax, kaz, alphax_max, alphaz_max, d0x, d0z;
	float **alphax, **dx1, **bx, **ax1;
	float **alphaz, **dz1, **bz, **az1; //CMPL variables

    float **wave, **dwave; //for load K0(dvx/dx+dvz/dz) and K0(ddvx/dx+ddvz/dz)
	FILE *fwave, *fdwave;
	int interval, wnt, wit;

	// for test
  /*  float **wave1, **dwave1; 
    float **wave2, **dwave2, **dd, **dd1; 
	FILE *fwave1, *fdwave1, *swap, *swap1;
	fwave1 = fopen("wave1","wb");
	fdwave1 = fopen("dwave1","wb");
	swap = fopen("arcd","wb");
	swap1 = fopen("arcd1","wb");*/

	MPI_Comm comm=MPI_COMM_WORLD;

	fwave = fopen("wave","rb");
	fdwave = fopen("dwave","rb");

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	float **gradr1, **grads1, **sillum1; //temporary variables for MPI_Reduce
	gradr1=sf_floatalloc2(nz,nx);
	grads1=sf_floatalloc2(nz,nx);
	sillum1=sf_floatalloc2(nz,nx);
	memset(gradr1[0], 0., nx*nz*sizeof(float));
	memset(grads1[0], 0., nx*nz*sizeof(float));
	memset(sillum1[0], 0., nx*nz*sizeof(float));

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

	interval = acpar->interval;
	wnt = (nt-1)/interval+1;
	wave=sf_floatalloc2(nz,nx);
	dwave=sf_floatalloc2(nz,nx);

	/*wave1=sf_floatalloc2(nz,nx);
	dwave1=sf_floatalloc2(nz,nx);
	wave2=sf_floatalloc2(nz,nx);
	dwave2=sf_floatalloc2(nz,nx);
	dd=sf_floatalloc2(nt,nr);
	dd1=sf_floatalloc2(nt,nr);*/

	vv = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	K0=sf_floatalloc2(padnz, padnx);
	dK=sf_floatalloc2(padnz, padnx);

	p1=sf_floatalloc2(padnz, padnx);
	p2=sf_floatalloc2(padnz, padnx);
	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);
	rp1=sf_floatalloc2(padnz, padnx);
	rp2=sf_floatalloc2(padnz, padnx);

	dp1=sf_floatalloc2(padnz, padnx);
	dp2=sf_floatalloc2(padnz, padnx);
	dvx1=sf_floatalloc2(padnz, padnx);
	dvx2=sf_floatalloc2(padnz, padnx);
	dvz1=sf_floatalloc2(padnz, padnx);
	dvz2=sf_floatalloc2(padnz, padnx);
	drp1=sf_floatalloc2(padnz, padnx);
	drp2=sf_floatalloc2(padnz, padnx);

	phi_vx1=sf_floatalloc2(padnz, padnx);
	phi_vx2=sf_floatalloc2(padnz, padnx);
	phi_vz1=sf_floatalloc2(padnz, padnx);
	phi_vz2=sf_floatalloc2(padnz, padnx);
	phi_px1=sf_floatalloc2(padnz, padnx);
	phi_px2=sf_floatalloc2(padnz, padnx);
	phi_pz1=sf_floatalloc2(padnz, padnx);
	phi_pz2=sf_floatalloc2(padnz, padnx);
	phi_rpx1=sf_floatalloc2(padnz, padnx);
	phi_rpx2=sf_floatalloc2(padnz, padnx);
	phi_rpz1=sf_floatalloc2(padnz, padnx);
	phi_rpz2=sf_floatalloc2(padnz, padnx);

	dphi_vx1=sf_floatalloc2(padnz, padnx);
	dphi_vx2=sf_floatalloc2(padnz, padnx);
	dphi_vz1=sf_floatalloc2(padnz, padnx);
	dphi_vz2=sf_floatalloc2(padnz, padnx);
	dphi_px1=sf_floatalloc2(padnz, padnx);
	dphi_px2=sf_floatalloc2(padnz, padnx);
	dphi_pz1=sf_floatalloc2(padnz, padnx);
	dphi_pz2=sf_floatalloc2(padnz, padnx);
	dphi_rpx1=sf_floatalloc2(padnz, padnx);
	dphi_rpx2=sf_floatalloc2(padnz, padnx);
	dphi_rpz1=sf_floatalloc2(padnz, padnx);
	dphi_rpz2=sf_floatalloc2(padnz, padnx);

	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);
	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);
	drpx=sf_floatalloc2(padnz, padnx);
	drpz=sf_floatalloc2(padnz, padnx);
	ddpx=sf_floatalloc2(padnz, padnx);
	ddpz=sf_floatalloc2(padnz, padnx);
	ddvx=sf_floatalloc2(padnz, padnx);
	ddvz=sf_floatalloc2(padnz, padnx);
	ddrpx=sf_floatalloc2(padnz, padnx);
	ddrpz=sf_floatalloc2(padnz, padnx);
	rr=sf_floatalloc(padnzx);

	dKp2=sf_floatalloc2(padnz, padnx);
	dKp2x=sf_floatalloc2(padnz, padnx);
	dKp2z=sf_floatalloc2(padnz, padnx);
	dKrp2=sf_floatalloc2(padnz, padnx);
	dKrp2x=sf_floatalloc2(padnz, padnx);
	dKrp2z=sf_floatalloc2(padnz, padnx);
	phi_dKp2x1=sf_floatalloc2(padnz, padnx);
	phi_dKp2x2=sf_floatalloc2(padnz, padnx);
	phi_dKp2z1=sf_floatalloc2(padnz, padnx);
	phi_dKp2z2=sf_floatalloc2(padnz, padnx);
	phi_dKrp2x1=sf_floatalloc2(padnz, padnx);
	phi_dKrp2x2=sf_floatalloc2(padnz, padnx);
	phi_dKrp2z1=sf_floatalloc2(padnz, padnx);
	phi_dKrp2z2=sf_floatalloc2(padnz, padnx);

	/* PML boundary condition*/
	alphax=sf_floatalloc2(padnz, padnx);
	dx1=sf_floatalloc2(padnz, padnx);
	bx=sf_floatalloc2(padnz, padnx);
	ax1=sf_floatalloc2(padnz, padnx);
	alphaz=sf_floatalloc2(padnz, padnx);
	dz1=sf_floatalloc2(padnz, padnx);
	bz=sf_floatalloc2(padnz, padnx);
	az1=sf_floatalloc2(padnz, padnx);
	kax = 1.;
	kaz = 1.;
	alphax_max = SF_PI*20.;
	alphaz_max = SF_PI*20.;
	d0x = -3.8*4.500*log(0.001)/2./nb/dx;
	d0z = -3.8*4.500*log(0.001)/2./nb/dz;
	for(ix=0;ix<padnx;ix++)
		for(iz=0;iz<padnz;iz++)
		{
		   alphax[ix][iz] = 0.0;
		   alphaz[ix][iz] = 0.0;
		   dx1[ix][iz] = 0.0;
		   dz1[ix][iz] = 0.0;
		   bx[ix][iz] = 0.0;
		   bz[ix][iz] = 0.0;
		   ax1[ix][iz] = 0.0;
		   az1[ix][iz] = 0.0;

           if(ix<nb)
           {
               alphax[ix][iz] = alphax_max*ix/nb;
               dx1[ix][iz] = d0x*pow((nb-ix)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.000001)*(bx[ix][iz]-1.);
           }
           if(ix>padnx-nb)
           {
               alphax[ix][iz] = alphax_max*(padnx-ix)/nb;
               dx1[ix][iz] = d0x*pow((ix+nb-padnx)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.000001)*(bx[ix][iz]-1.);
           }
           if(iz<nb)
           {
               alphaz[ix][iz] = alphaz_max*iz/nb;
               dz1[ix][iz] = d0z*pow((nb-iz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.000001)*(bz[ix][iz]-1.);
           }
           if(iz>padnz-nb)
           {
               alphaz[ix][iz] = alphaz_max*(padnz-iz)/nb;
               dz1[ix][iz] = d0x*pow((iz+nb-padnz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.000001)*(bz[ix][iz]-1.);
           }
           bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
           bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
      }

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->dvv, dK, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(ix=0; ix<padnx; ix++)
		for(iz=0; iz<padnz; iz++)
		{
			K0[ix][iz] = vv[ix][iz]*vv[ix][iz];
			dK[ix][iz] = 2.0*vv[ix][iz]*dK[ix][iz]+dK[ix][iz]*dK[ix][iz];
		//	if(iz<nb+10)
		//		dK[ix][iz] = 0;
		 //  if(iz==130)
		//	   dK[ix][iz] = 1.6;
		//K0[ix][iz] += dK[ix][iz];
		}

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){

		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(rp1[0], 0., padnzx*sizeof(float));
		memset(rp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		memset(phi_vx1[0], 0., padnzx*sizeof(float));
		memset(phi_vx2[0], 0., padnzx*sizeof(float));
		memset(phi_vz1[0], 0., padnzx*sizeof(float));
		memset(phi_vz2[0], 0., padnzx*sizeof(float));
		memset(phi_px1[0], 0., padnzx*sizeof(float));
		memset(phi_px2[0], 0., padnzx*sizeof(float));
		memset(phi_pz1[0], 0., padnzx*sizeof(float));
		memset(phi_pz2[0], 0., padnzx*sizeof(float));
		memset(phi_rpx1[0], 0., padnzx*sizeof(float));
		memset(phi_rpx2[0], 0., padnzx*sizeof(float));
		memset(phi_rpz1[0], 0., padnzx*sizeof(float));
		memset(phi_rpz2[0], 0., padnzx*sizeof(float));
		memset(dp1[0], 0., padnzx*sizeof(float));
		memset(dp2[0], 0., padnzx*sizeof(float));
		memset(drp1[0], 0., padnzx*sizeof(float));
		memset(drp2[0], 0., padnzx*sizeof(float));
		memset(dvx1[0], 0., padnzx*sizeof(float));
		memset(dvx2[0], 0., padnzx*sizeof(float));
		memset(dvz1[0], 0., padnzx*sizeof(float));
		memset(dvz2[0], 0., padnzx*sizeof(float));
		memset(dphi_vx1[0], 0., padnzx*sizeof(float));
		memset(dphi_vx2[0], 0., padnzx*sizeof(float));
		memset(dphi_vz1[0], 0., padnzx*sizeof(float));
		memset(dphi_vz2[0], 0., padnzx*sizeof(float));
		memset(dphi_px1[0], 0., padnzx*sizeof(float));
		memset(dphi_px2[0], 0., padnzx*sizeof(float));
		memset(dphi_pz1[0], 0., padnzx*sizeof(float));
		memset(dphi_pz2[0], 0., padnzx*sizeof(float));
		memset(dphi_rpx1[0], 0., padnzx*sizeof(float));
		memset(dphi_rpx2[0], 0., padnzx*sizeof(float));
		memset(dphi_rpz1[0], 0., padnzx*sizeof(float));
		memset(dphi_rpz2[0], 0., padnzx*sizeof(float));
		memset(dvx[0], 0., padnzx*sizeof(float));
		memset(dvz[0], 0., padnzx*sizeof(float));
		memset(dpx[0], 0., padnzx*sizeof(float));
		memset(dpz[0], 0., padnzx*sizeof(float));
		memset(drpx[0], 0., padnzx*sizeof(float));
		memset(drpz[0], 0., padnzx*sizeof(float));
		memset(ddvx[0], 0., padnzx*sizeof(float));
		memset(ddvz[0], 0., padnzx*sizeof(float));
		memset(ddpx[0], 0., padnzx*sizeof(float));
		memset(ddpz[0], 0., padnzx*sizeof(float));
		memset(ddrpx[0], 0., padnzx*sizeof(float));
		memset(ddrpz[0], 0., padnzx*sizeof(float));
		memset(dKp2[0], 0., padnzx*sizeof(float));
		memset(dKp2x[0], 0., padnzx*sizeof(float));
		memset(dKp2z[0], 0., padnzx*sizeof(float));
		memset(dKrp2[0], 0., padnzx*sizeof(float));
		memset(dKrp2x[0], 0., padnzx*sizeof(float));
		memset(dKrp2z[0], 0., padnzx*sizeof(float));
		memset(phi_dKp2x1[0], 0., padnzx*sizeof(float));
		memset(phi_dKp2x2[0], 0., padnzx*sizeof(float));
		memset(phi_dKp2z1[0], 0., padnzx*sizeof(float));
		memset(phi_dKp2z2[0], 0., padnzx*sizeof(float));
		memset(phi_dKrp2x1[0], 0., padnzx*sizeof(float));
		memset(phi_dKrp2x2[0], 0., padnzx*sizeof(float));
		memset(phi_dKrp2z1[0], 0., padnzx*sizeof(float));
		memset(phi_dKrp2z2[0], 0., padnzx*sizeof(float));
		/*for(ix=0; ix<padnx; ix++)
			for(iz=0; iz<padnz; iz++)
			{
			    p1[ix][iz] = 0.;
				p2[ix][iz] = 0.;
				rp1[ix][iz] = 0.;
				rp2[ix][iz] = 0.;
				vx1[ix][iz] = 0.;
				vx2[ix][iz] = 0.;
				vz1[ix][iz] = 0.;
				vz2[ix][iz] = 0.;
				phi_vx1[ix][iz] = 0.;
				phi_vx2[ix][iz] = 0.;
				phi_vz1[ix][iz] = 0.;
				phi_vz2[ix][iz] = 0.;
				phi_px1[ix][iz] = 0.;
				phi_px2[ix][iz] = 0.;
				phi_pz1[ix][iz] = 0.;
				phi_pz2[ix][iz] = 0.;
				phi_rpx1[ix][iz] = 0.;
				phi_rpx2[ix][iz] = 0.;
				phi_rpz1[ix][iz] = 0.;
				phi_rpz2[ix][iz] = 0.;
				dp1[ix][iz] = 0.;
				dp2[ix][iz] = 0.;
				drp1[ix][iz] = 0.;
				drp2[ix][iz] = 0.;
				dvx1[ix][iz] = 0.;
				dvx2[ix][iz] = 0.;
				dvz1[ix][iz] = 0.;
				dvz2[ix][iz] = 0.;
				dphi_vx1[ix][iz] = 0.;
				dphi_vx2[ix][iz] = 0.;
				dphi_vz1[ix][iz] = 0.;
				dphi_vz2[ix][iz] = 0.;
				dphi_px1[ix][iz] = 0.;
				dphi_px2[ix][iz] = 0.;
				dphi_pz1[ix][iz] = 0.;
				dphi_pz2[ix][iz] = 0.;
				dphi_rpx1[ix][iz] = 0.;
				dphi_rpx2[ix][iz] = 0.;
				dphi_rpz1[ix][iz] = 0.;
				dphi_rpz2[ix][iz] = 0.;
				dvx[ix][iz] = 0.;
				dvz[ix][iz] = 0.;
				dpx[ix][iz] = 0.;
				dpz[ix][iz] = 0.;
				drpx[ix][iz] = 0.;
				drpz[ix][iz] = 0.;
				ddvx[ix][iz] = 0.;
				ddvz[ix][iz] = 0.;
				ddpx[ix][iz] = 0.;
				ddpz[ix][iz] = 0.;
				ddrpx[ix][iz] = 0.;
				ddrpz[ix][iz] = 0.;
			}*/
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		wit = wnt;
		for(it=nt-1; it>=0; it--){
			if(verb && it%200==0) sf_warning("Backward modeling is=%d; it=%d;", is+1, it);

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p2[ix][iz] = -dt*(dvz[ix][iz]+dvx[ix][iz]+phi_vz2[ix][iz]
								+phi_vx2[ix][iz])+p1[ix][iz];// + rr[ix*padnz+iz]*array->ww[nt-it-1];

					rp2[ix][iz] = -dt*(rp1[ix][iz]/taus[ix][iz]-p2[ix][iz]) + rp1[ix][iz];

					if(iz<nb || iz>=padnz-nb)
						phi_vz2[ix][iz] = bz[ix][iz]*phi_vz1[ix][iz] + az1[ix][iz]*dvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_vx2[ix][iz] = bx[ix][iz]*phi_vx1[ix][iz] + ax1[ix][iz]*dvx[ix][iz];

				//	dKp2[ix][iz] = K0[ix][iz]*p2[ix][iz]*(tau[ix][iz]+1.0);
				//	dKrp2[ix][iz] = K0[ix][iz]*rp2[ix][iz]*tau[ix][iz]/taus[ix][iz];
				}

			/* load data residual */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
			    p2[rx][rz]+=rcd[is][acpar->r02[is]+ir][it];
			//	dd[acpar->r02[is]+ir][it]=p2[rx][rz];
			//	dd1[acpar->r02[is]+ir][nt-it-1]=dp2[rx][rz];
			}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			staggerFD1order2Ddx1(rp2,drpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(rp2,drpz,padnx,padnz,dx,dz,5,1,0);
			//staggerFD1order2Ddx1(dKp2,dpz,padnx,padnz,dx,dz,5,1,0);
			//staggerFD1order2Ddx1(dKrp2,drpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					vz2[ix][iz] = -dt*((K0[ix][iz])*((tau[ix][iz]+1.0)*(dpz[ix][iz]+phi_pz2[ix][iz])
								-tau[ix][iz]/taus[ix][iz]*(drpz[ix][iz]+phi_rpz2[ix][iz]))) +vz1[ix][iz];
			//		vz2[ix][iz] = -dt*(dpz[ix][iz]+phi_pz2[ix][iz]-drpz[ix][iz]-phi_rpz2[ix][iz]) +vz1[ix][iz];

					vx2[ix][iz] = -dt*((K0[ix][iz])*((tau[ix][iz]+1.0)*(dpx[ix][iz]+phi_px2[ix][iz])
								-tau[ix][iz]/taus[ix][iz]*(drpx[ix][iz]+phi_rpx2[ix][iz]))) +vx1[ix][iz]; 

					/* acoustic */
			//		vz2[ix][iz] = -dt*(K0[ix][iz]*(dpz[ix][iz]+phi_pz2[ix][iz])+dK[ix][iz]/dz*p2[ix][iz])+vz1[ix][iz];
			//		vx2[ix][iz] = -dt*(K0[ix][iz]*(dpx[ix][iz]+phi_px2[ix][iz]))+vx1[ix][iz];

					if(iz<nb || iz>=padnz-nb)
					{
						phi_pz2[ix][iz] = bz[ix][iz]*phi_pz1[ix][iz] + az1[ix][iz]*dpz[ix][iz];
						phi_rpz2[ix][iz] = bz[ix][iz]*phi_rpz1[ix][iz] + az1[ix][iz]*drpz[ix][iz];
					}
					if(ix<nb || ix>=padnx-nb)
					{
						phi_px2[ix][iz] = bx[ix][iz]*phi_px1[ix][iz] + ax1[ix][iz]*dpx[ix][iz];
						phi_rpx2[ix][iz] = bx[ix][iz]*phi_rpx1[ix][iz] + ax1[ix][iz]*drpx[ix][iz];
					}
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p1[ix][iz] = p2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					rp1[ix][iz] = rp2[ix][iz];
					phi_px1[ix][iz] = phi_px2[ix][iz];
					phi_pz1[ix][iz] = phi_pz2[ix][iz];
					phi_vx1[ix][iz] = phi_vx2[ix][iz];
					phi_vz1[ix][iz] = phi_vz2[ix][iz];
					phi_rpx1[ix][iz] = phi_rpx2[ix][iz];
					phi_rpz1[ix][iz] = phi_rpz2[ix][iz];
				}

			staggerFD1order2Ddx1(dvx2,ddvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(dvz2,ddvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					dp2[ix][iz] = -dt*(ddvz[ix][iz]+ddvx[ix][iz]+dphi_vz2[ix][iz]
								+dphi_vx2[ix][iz])+dp1[ix][iz];

					drp2[ix][iz] = -dt*(drp1[ix][iz]/taus[ix][iz]-dp2[ix][iz]) + drp1[ix][iz];

					// acoustic 
				//	dp2[ix][iz] = -dt*(K0[ix][iz]*(ddvz[ix][iz]+ddvx[ix][iz]+dphi_vz2[ix][iz]+dphi_vx2[ix][iz])
				//			      +dK[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]+phi_vz2[ix][iz]+phi_vx2[ix][iz]))+dp1[ix][iz];

					if(iz<nb || iz>=padnz-nb)
						dphi_vz2[ix][iz] = bz[ix][iz]*dphi_vz1[ix][iz] + az1[ix][iz]*ddvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						dphi_vx2[ix][iz] = bx[ix][iz]*dphi_vx1[ix][iz] + ax1[ix][iz]*ddvx[ix][iz];

					dKp2[ix][iz] = dK[ix][iz]*p2[ix][iz]*(tau[ix][iz]+1.0);
					dKrp2[ix][iz] = dK[ix][iz]*rp2[ix][iz]*tau[ix][iz]/taus[ix][iz];
				}

			staggerFD1order2Ddx1(dp2,ddpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(dp2,ddpz,padnx,padnz,dx,dz,5,1,0);
			staggerFD1order2Ddx1(drp2,ddrpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(drp2,ddrpz,padnx,padnz,dx,dz,5,1,0);
			staggerFD1order2Ddx1(dKp2,dKp2x,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(dKp2,dKp2z,padnx,padnz,dx,dz,5,1,0);
			staggerFD1order2Ddx1(dKrp2,dKrp2x,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(dKrp2,dKrp2z,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					dvz2[ix][iz] = -dt*((K0[ix][iz])*((tau[ix][iz]+1.0)*(ddpz[ix][iz]+dphi_pz2[ix][iz])
								-tau[ix][iz]/taus[ix][iz]*(ddrpz[ix][iz]+dphi_rpz2[ix][iz])) 
								+dKp2z[ix][iz]+phi_dKp2z2[ix][iz]-(dKrp2z[ix][iz]+phi_dKrp2z2[ix][iz])) +dvz1[ix][iz];

					dvx2[ix][iz] = -dt*((K0[ix][iz])*((tau[ix][iz]+1.0)*(ddpx[ix][iz]+dphi_px2[ix][iz])
								-tau[ix][iz]/taus[ix][iz]*(ddrpx[ix][iz]+dphi_rpx2[ix][iz])) 
								+dKp2x[ix][iz]+phi_dKp2x2[ix][iz]-(dKrp2x[ix][iz]+phi_dKrp2x2[ix][iz])) +dvx1[ix][iz];

					/* acoustic */
				//	dvz2[ix][iz] = -dt*(K0[ix][iz]*(ddpz[ix][iz]+dphi_pz2[ix][iz])+(dKp2z[ix][iz]+phi_pz2[ix][iz])) +dvz1[ix][iz];
				//	dvx2[ix][iz] = -dt*(K0[ix][iz]*(ddpx[ix][iz]+dphi_px2[ix][iz])+dK[ix][iz]*(dpx[ix][iz]+phi_px2[ix][iz])) +dvx1[ix][iz];

					if(iz<nb || iz>=padnz-nb)
					{
						dphi_pz2[ix][iz] = bz[ix][iz]*dphi_pz1[ix][iz] + az1[ix][iz]*ddpz[ix][iz];
						dphi_rpz2[ix][iz] = bz[ix][iz]*dphi_rpz1[ix][iz] + az1[ix][iz]*ddrpz[ix][iz];
						phi_dKp2z2[ix][iz] = bz[ix][iz]*phi_dKp2z1[ix][iz] + az1[ix][iz]*dKp2z[ix][iz];
						phi_dKrp2z2[ix][iz] = bz[ix][iz]*phi_dKrp2z1[ix][iz] + az1[ix][iz]*dKrp2z[ix][iz];
					}
					if(ix<nb || ix>=padnx-nb)
					{
						dphi_px2[ix][iz] = bx[ix][iz]*dphi_px1[ix][iz] + ax1[ix][iz]*ddpx[ix][iz];
						dphi_rpx2[ix][iz] = bx[ix][iz]*dphi_rpx1[ix][iz] + ax1[ix][iz]*ddrpx[ix][iz];
						phi_dKp2x2[ix][iz] = bz[ix][iz]*phi_dKp2x1[ix][iz] + az1[ix][iz]*dKp2x[ix][iz];
						phi_dKrp2x2[ix][iz] = bz[ix][iz]*phi_dKrp2x1[ix][iz] + az1[ix][iz]*dKrp2x[ix][iz];
					}
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					dp1[ix][iz] = dp2[ix][iz];
					dvx1[ix][iz] = dvx2[ix][iz];
					dvz1[ix][iz] = dvz2[ix][iz];
					drp1[ix][iz] = drp2[ix][iz];
					dphi_px1[ix][iz] = dphi_px2[ix][iz];
					dphi_pz1[ix][iz] = dphi_pz2[ix][iz];
					dphi_vx1[ix][iz] = dphi_vx2[ix][iz];
					dphi_vz1[ix][iz] = dphi_vz2[ix][iz];
					dphi_rpx1[ix][iz] = dphi_rpx2[ix][iz];
					dphi_rpz1[ix][iz] = dphi_rpz2[ix][iz];
					phi_dKp2x1[ix][iz] = phi_dKp2x2[ix][iz];
					phi_dKp2z1[ix][iz] = phi_dKp2z2[ix][iz];
					phi_dKrp2x1[ix][iz] = phi_dKrp2x2[ix][iz];
					phi_dKrp2z1[ix][iz] = phi_dKrp2z2[ix][iz];
				}

			/* calculate gradient  */
			if(it%interval==0)
			{
				fseeko(fwave, (is*wnt+wit-1)*nx*nz*sizeof(float), SEEK_SET);
				fseeko(fdwave, (is*wnt+wit-1)*nx*nz*sizeof(float), SEEK_SET);
				fread(wave[0], sizeof(float), nz*nx, fwave);
				fread(dwave[0], sizeof(float), nz*nx, fdwave);
		/*		fseeko(fwave, (is*wnt+wit-2)*nx*nz*sizeof(float), SEEK_SET);
				fseeko(fdwave, (is*wnt+wit-2)*nx*nz*sizeof(float), SEEK_SET);
				fread(wave1[0], sizeof(float), nz*nx, fwave);
				fread(dwave1[0], sizeof(float), nz*nx, fdwave);
				fseeko(fwave, (is*wnt+wit-3)*nx*nz*sizeof(float), SEEK_SET);
				fseeko(fdwave, (is*wnt+wit-3)*nx*nz*sizeof(float), SEEK_SET);
				fread(wave2[0], sizeof(float), nz*nx, fwave);
				fread(dwave2[0], sizeof(float), nz*nx, fdwave);*/
				for(ix=0; ix<nx; ix++)
					for(iz=0; iz<nz; iz++)
					{
						// gradient for \tau
						grads1[ix][iz] += K0[ix+nb][iz+nb]*wave[ix][iz]*(-dp2[ix+nb][iz+nb]
				    			         +1./taus[ix+nb][iz+nb]*drp2[ix+nb][iz+nb]); //source side
						gradr1[ix][iz] += (K0[ix+nb][iz+nb])*dwave[ix][iz]*(-p2[ix+nb][iz+nb]
						                 +1./taus[ix+nb][iz+nb]*rp2[ix+nb][iz+nb]); //receiver side

						//gradr1[ix][iz] += K0[ix+nb][iz+nb]*wave[ix][iz]*(-p2[ix+nb][iz+nb]
						//		        +1./taus[ix+nb][iz+nb]*rp2[ix+nb][iz+nb]); // FWI gradient

						// gradient for K0
				//		grads[ix][iz] += -(-2*wave1[ix][iz] + wave[ix][iz] + wave2[ix][iz])*p2[ix+nb][iz+nb];
		//			    grads[ix][iz] += K0[ix+nb][iz+nb]*wave[ix][iz]*(-p2[ix+nb][iz+nb]+1./taus[ix+nb][iz+nb]*rp2[ix+nb][iz+nb]);
	/*					gradr1[ix][iz] += wave[ix][iz]*(-dp2[ix+nb][iz+nb]*(tau[ix+nb][iz+nb]+1.0)
								         +tau[ix+nb][iz+nb]/taus[ix+nb][iz+nb]*drp2[ix+nb][iz+nb]);
						grads1[ix][iz] += dwave[ix][iz]*(-p2[ix+nb][iz+nb]*(tau[ix+nb][iz+nb]+1.0)
								      +tau[ix+nb][iz+nb]/taus[ix+nb][iz+nb]*rp2[ix+nb][iz+nb]);*/

						sillum1[ix][iz] += p2[ix+nb][iz+nb]*p2[ix+nb][iz+nb];
					}
				wit--;
			}

		/*	if(it%interval==0)
			{
				fseeko(fwave1, (is*wnt+wit)*nx*nz*sizeof(float), SEEK_SET);
				fseeko(fdwave1, (is*wnt+wit)*nx*nz*sizeof(float), SEEK_SET);
				for(ix=0; ix<nx; ix++)
					for(iz=0; iz<nz; iz++)
					{
						wave[ix][iz] = p2[ix+nb][iz+nb];
						dwave[ix][iz] = dp2[ix+nb][iz+nb];
					}
				fwrite(wave[0], sizeof(float), nz*nx, fwave1);
				fwrite(dwave[0], sizeof(float), nz*nx, fdwave1);
			}*/

		} // end of time loop

	/*	fseeko(swap, is*nr*nt*sizeof(float), SEEK_SET);
		fwrite(dd[0], sizeof(float), nr*nt, swap);
		fseeko(swap1, is*nr*nt*sizeof(float), SEEK_SET);
		fwrite(dd1[0], sizeof(float), nr*nt, swap1);*/

	}// end of shot loop
	fclose(fwave);
	fclose(fdwave);
	MPI_Barrier(comm);

	/* grads and gradr reduction */
	MPI_Reduce(&gradr1[0][0], &gradr[0][0], nx*nz, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(&gradr[0][0], nx*nz, MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	MPI_Reduce(grads1[0], grads[0], nx*nz, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(grads[0], nx*nz, MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	MPI_Reduce(sillum1[0], sillum[0], nx*nz, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(sillum[0], nx*nz, MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	/* release allocated memory */
	free(*wave); free(wave);
	free(*dwave); free(dwave);
	free(*p2); free(p2); 
	free(*p1); free(p1);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*rp2); free(rp2);
	free(*rp1); free(rp1);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*drpx); free(drpx); free(*drpz); free(drpz);
	free(*phi_vx1); free(phi_vx1);
	free(*phi_vx2); free(phi_vx2);
	free(*phi_vz1); free(phi_vz1);
	free(*phi_vz2); free(phi_vz2);
	free(*phi_px1); free(phi_px1);
	free(*phi_px2); free(phi_px2);
	free(*phi_pz1); free(phi_pz1);
	free(*phi_pz2); free(phi_pz2);
	free(*phi_rpx1); free(phi_rpx1);
	free(*phi_rpx2); free(phi_rpx2);
	free(*phi_rpz1); free(phi_rpz1);
	free(*phi_rpz2); free(phi_rpz2);

	free(*dp2); free(dp2); 
	free(*dp1); free(dp1);
	free(*dvx1); free(dvx1); free(*dvx2); free(dvx2);
	free(*dvz1); free(dvz1); free(*dvz2); free(dvz2);
	free(*drp2); free(drp2);
	free(*drp1); free(drp1);
	free(*ddvx); free(ddvx); free(*ddvz); free(ddvz);
	free(*ddpx); free(ddpx); free(*ddpz); free(ddpz);
	free(*ddrpx); free(ddrpx); free(*ddrpz); free(ddrpz);
	free(*dphi_vx1); free(dphi_vx1);
	free(*dphi_vx2); free(dphi_vx2);
	free(*dphi_vz1); free(dphi_vz1);
	free(*dphi_vz2); free(dphi_vz2);
	free(*dphi_px1); free(dphi_px1);
	free(*dphi_px2); free(dphi_px2);
	free(*dphi_pz1); free(dphi_pz1);
	free(*dphi_pz2); free(dphi_pz2);
	free(*dphi_rpx1); free(dphi_rpx1);
	free(*dphi_rpx2); free(dphi_rpx2);
	free(*dphi_rpz1); free(dphi_rpz1);
	free(*dphi_rpz2); free(dphi_rpz2);

	free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*K0); free(K0); free(*dK); free(dK);
	free(rr); 
	free(*alphax); free(alphax);
	free(*alphaz); free(alphaz);
	free(*dx1); free(dx1);
	free(*ax1); free(ax1);
	free(*bx); free(bx);
	free(*dz1); free(dz1);
	free(*az1); free(az1);
	free(*bz); free(bz);

	free(*gradr1); free(gradr1);
	free(*grads1); free(grads1);
	free(*sillum1); free(sillum1);

	free(*dKp2); free(dKp2);
	free(*dKp2x); free(dKp2x);
	free(*dKp2z); free(dKp2z);
	free(*dKrp2); free(dKrp2); 
	free(*dKrp2x); free(dKrp2x); 
	free(*dKrp2z); free(dKrp2z); 
	free(*phi_dKp2x1); free(phi_dKp2x1);
	free(*phi_dKp2x2); free(phi_dKp2x2);
	free(*phi_dKp2z1); free(phi_dKp2z1);
	free(*phi_dKp2z2); free(phi_dKp2z2);
	free(*phi_dKrp2x1); free(phi_dKrp2x1);
	free(*phi_dKrp2x2); free(phi_dKrp2x2);
	free(*phi_dKrp2z1); free(phi_dKrp2z1);
	free(*phi_dKrp2z2); free(phi_dKrp2z2);
	MPI_Barrier(comm);
}

void forward_modeling_cpml(sf_file Fdat, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic forward modeling with CPML boundary condition >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **tau, **taus,**K0, **dK;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **p1, **p2;
	float **rp1, **rp2;
	float **phi_vx1, **phi_vx2, **phi_vz1, **phi_vz2;
	float **phi_px1, **phi_px2, **phi_pz1, **phi_pz2;
	float **dpx, **dpz, **dvx, **dvz;

	float kax, kaz, alphax_max, alphaz_max, d0x, d0z;
	float **alphax, **dx1, **bx, **ax1;
	float **alphaz, **dz1, **bz, **az1; //CMPL variables

	MPI_Comm comm=MPI_COMM_WORLD;

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

    float ***rcd, ***rcd1;
	rcd = sf_floatalloc3(nt,nr,acpar->ns);
	rcd1 = sf_floatalloc3(nt,nr,acpar->ns);

	vv = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	K0=sf_floatalloc2(padnz, padnx);
	dK=sf_floatalloc2(padnz, padnx);

	p1=sf_floatalloc2(padnz, padnx);
	p2=sf_floatalloc2(padnz, padnx);
	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);
	rp1=sf_floatalloc2(padnz, padnx);
	rp2=sf_floatalloc2(padnz, padnx);

	phi_vx1=sf_floatalloc2(padnz, padnx);
	phi_vx2=sf_floatalloc2(padnz, padnx);
	phi_vz1=sf_floatalloc2(padnz, padnx);
	phi_vz2=sf_floatalloc2(padnz, padnx);
	phi_px1=sf_floatalloc2(padnz, padnx);
	phi_px2=sf_floatalloc2(padnz, padnx);
	phi_pz1=sf_floatalloc2(padnz, padnx);
	phi_pz2=sf_floatalloc2(padnz, padnx);

	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);
	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);
	rr=sf_floatalloc(padnzx);

	/* PML boundary condition*/
	alphax=sf_floatalloc2(padnz, padnx);
	dx1=sf_floatalloc2(padnz, padnx);
	bx=sf_floatalloc2(padnz, padnx);
	ax1=sf_floatalloc2(padnz, padnx);
	alphaz=sf_floatalloc2(padnz, padnx);
	dz1=sf_floatalloc2(padnz, padnx);
	bz=sf_floatalloc2(padnz, padnx);
	az1=sf_floatalloc2(padnz, padnx);
	kax = 1.;
	kaz = 1.;
	alphax_max = SF_PI*20.;
	alphaz_max = SF_PI*20.;
	d0x = -3.8*4.500*log(0.001)/2./nb/dx;
	d0z = -3.8*4.500*log(0.001)/2./nb/dz;
	for(ix=0;ix<padnx;ix++)
		for(iz=0;iz<padnz;iz++)
		{
		   alphax[ix][iz] = 0.0;
		   alphaz[ix][iz] = 0.0;
		   dx1[ix][iz] = 0.0;
		   dz1[ix][iz] = 0.0;
		   bx[ix][iz] = 0.0;
		   bz[ix][iz] = 0.0;
		   ax1[ix][iz] = 0.0;
		   az1[ix][iz] = 0.0;

           if(ix<nb)
           {
               alphax[ix][iz] = alphax_max*ix/nb;
               dx1[ix][iz] = d0x*pow((nb-ix)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.001)*(bx[ix][iz]-1.);
           }
           if(ix>padnx-nb)
           {
               alphax[ix][iz] = alphax_max*(padnx-ix)/nb;
               dx1[ix][iz] = d0x*pow((ix+nb-padnx)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.001)*(bx[ix][iz]-1.);
           }
           if(iz<nb)
           {
               alphaz[ix][iz] = alphaz_max*iz/nb;
               dz1[ix][iz] = d0z*pow((nb-iz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.001)*(bz[ix][iz]-1.);
           }
           if(iz>padnz-nb)
           {
               alphaz[ix][iz] = alphaz_max*(padnz-iz)/nb;
               dz1[ix][iz] = d0x*pow((iz+nb-padnz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.001)*(bz[ix][iz]-1.);
           }
           bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
           bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
      }

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->dvv, dK, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(ix=0; ix<padnx; ix++)
		for(iz=0; iz<padnz; iz++)
		{
			K0[ix][iz] = vv[ix][iz]*vv[ix][iz];
			dK[ix][iz] = 2.0*vv[ix][iz]*dK[ix][iz]+dK[ix][iz]*dK[ix][iz];
		//	if(iz<nb+10)
		//		dK[ix][iz] = 0;
		//	if(iz>148 && iz<153 && ix>padnx/2-3 && ix<padnx/2+3)
		//		dK[ix][iz] = 0.6;
		//	K0[ix][iz] += dK[ix][iz];
		}

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){
		sf_warning("###### is=%d ######", is+1);

		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(rp1[0], 0., padnzx*sizeof(float));
		memset(rp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		memset(phi_vx1[0], 0., padnzx*sizeof(float));
		memset(phi_vx2[0], 0., padnzx*sizeof(float));
		memset(phi_vz1[0], 0., padnzx*sizeof(float));
		memset(phi_vz2[0], 0., padnzx*sizeof(float));
		memset(phi_px1[0], 0., padnzx*sizeof(float));
		memset(phi_px2[0], 0., padnzx*sizeof(float));
		memset(phi_pz1[0], 0., padnzx*sizeof(float));
		memset(phi_pz2[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
			if(verb && it%200==0) sf_warning("Forward modeling is=%d; it=%d;", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				rcd1[is][acpar->r02[is]+ir][it]=p2[rx][rz];
			//	if(it<abs(sx-rx)*dx/vv[rx][sz]/dt+300+300./(0.1*abs(sx-rx)+1))
			//		rcd1[is][acpar->r02[is]+ir][it] = 0;
			}

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					rp2[ix][iz] = -dt/taus[ix][iz]*(rp1[ix][iz]+tau[ix][iz]*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]
								+phi_vz2[ix][iz]+phi_vx2[ix][iz])) + rp1[ix][iz];
					p2[ix][iz] = dt*((tau[ix][iz]+1)*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]+phi_vz2[ix][iz]
								+phi_vx2[ix][iz])+rp2[ix][iz])+p1[ix][iz] + rr[ix*padnz+iz]*array->ww[it];

					if(iz<nb || iz>=padnz-nb)
						phi_vz2[ix][iz] = bz[ix][iz]*phi_vz1[ix][iz] + az1[ix][iz]*dvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_vx2[ix][iz] = bx[ix][iz]*phi_vx1[ix][iz] + ax1[ix][iz]*dvx[ix][iz];
				}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					vz2[ix][iz] = dt*(dpz[ix][iz]+phi_pz2[ix][iz]) +vz1[ix][iz];
					vx2[ix][iz] = dt*(dpx[ix][iz]+phi_px2[ix][iz]) +vx1[ix][iz];
					if(iz<nb || iz>=padnz-nb)
						phi_pz2[ix][iz] = bz[ix][iz]*phi_pz1[ix][iz] + az1[ix][iz]*dpz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_px2[ix][iz] = bx[ix][iz]*phi_px1[ix][iz] + ax1[ix][iz]*dpx[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p1[ix][iz] = p2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					rp1[ix][iz] = rp2[ix][iz];
					phi_px1[ix][iz] = phi_px2[ix][iz];
					phi_pz1[ix][iz] = phi_pz2[ix][iz];
					phi_vx1[ix][iz] = phi_vx2[ix][iz];
					phi_vz1[ix][iz] = phi_vz2[ix][iz];
				}

		} // end of time loop

	}// end of shot loop

	MPI_Barrier(comm);
	MPI_Reduce(&rcd1[0][0][0], &rcd[0][0][0], acpar->ns*nr*nt, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Barrier(comm);
	if(mpipar->cpuid==0){
		sf_floatwrite(&rcd[0][0][0],acpar->ns*nr*nt, Fdat);}
	
	/* release allocated memory */
	free(*p2); free(p2); 
	free(*p1); free(p1);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*rp2); free(rp2);
	free(*rp1); free(rp1);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*phi_vx1); free(phi_vx1);
	free(*phi_vx2); free(phi_vx2);
	free(*phi_vz1); free(phi_vz1);
	free(*phi_vz2); free(phi_vz2);
	free(*phi_px1); free(phi_px1);
	free(*phi_px2); free(phi_px2);
	free(*phi_pz1); free(phi_pz1);
	free(*phi_pz2); free(phi_pz2);
	free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*K0); free(K0); free(*dK); free(dK);
	free(rr); 
	free(**rcd); free(*rcd); free(rcd);
	free(**rcd1); free(*rcd1); free(rcd1);
	free(*alphax); free(alphax);
	free(*alphaz); free(alphaz);
	free(*dx1); free(dx1);
	free(*ax1); free(ax1);
	free(*bx); free(bx);
	free(*dz1); free(dz1);
	free(*az1); free(az1);
	free(*bz); free(bz);
}

void forward_modeling_ls(float ***dd, float *fcost, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic forward modeling for line search >*/
{
	MPI_Comm comm=MPI_COMM_WORLD;
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **tau, **taus,**K0, **dK;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **p1, **p2;
	float **rp1, **rp2;
	float **phi_vx1, **phi_vx2, **phi_vz1, **phi_vz2;
	float **phi_px1, **phi_px2, **phi_pz1, **phi_pz2;
	float **dpx, **dpz, **dvx, **dvz;

	float kax, kaz, alphax_max, alphaz_max, d0x, d0z;
	float **alphax, **dx1, **bx, **ax1;
	float **alphaz, **dz1, **bz, **az1; //CMPL variables

	float fcost1 = 0.0;

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

    float ***rcd1;
	rcd1 = sf_floatalloc3(nt,nr,acpar->ns);

    float ***rcd;
	rcd = sf_floatalloc3(nt,nr,acpar->ns);

	vv = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	K0=sf_floatalloc2(padnz, padnx);
	dK=sf_floatalloc2(padnz, padnx);

	p1=sf_floatalloc2(padnz, padnx);
	p2=sf_floatalloc2(padnz, padnx);
	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);
	rp1=sf_floatalloc2(padnz, padnx);
	rp2=sf_floatalloc2(padnz, padnx);

	phi_vx1=sf_floatalloc2(padnz, padnx);
	phi_vx2=sf_floatalloc2(padnz, padnx);
	phi_vz1=sf_floatalloc2(padnz, padnx);
	phi_vz2=sf_floatalloc2(padnz, padnx);
	phi_px1=sf_floatalloc2(padnz, padnx);
	phi_px2=sf_floatalloc2(padnz, padnx);
	phi_pz1=sf_floatalloc2(padnz, padnx);
	phi_pz2=sf_floatalloc2(padnz, padnx);

	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);
	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);
	rr=sf_floatalloc(padnzx);

	/* PML boundary condition*/
	alphax=sf_floatalloc2(padnz, padnx);
	dx1=sf_floatalloc2(padnz, padnx);
	bx=sf_floatalloc2(padnz, padnx);
	ax1=sf_floatalloc2(padnz, padnx);
	alphaz=sf_floatalloc2(padnz, padnx);
	dz1=sf_floatalloc2(padnz, padnx);
	bz=sf_floatalloc2(padnz, padnx);
	az1=sf_floatalloc2(padnz, padnx);
	kax = 1.;
	kaz = 1.;
	alphax_max = SF_PI*20.;
	alphaz_max = SF_PI*20.;
	d0x = -3.8*4.500*log(0.001)/2./nb/dx;
	d0z = -3.8*4.500*log(0.001)/2./nb/dz;
	for(ix=0;ix<padnx;ix++)
		for(iz=0;iz<padnz;iz++)
		{
		   alphax[ix][iz] = 0.0;
		   alphaz[ix][iz] = 0.0;
		   dx1[ix][iz] = 0.0;
		   dz1[ix][iz] = 0.0;
		   bx[ix][iz] = 0.0;
		   bz[ix][iz] = 0.0;
		   ax1[ix][iz] = 0.0;
		   az1[ix][iz] = 0.0;

           if(ix<nb)
           {
               alphax[ix][iz] = alphax_max*ix/nb;
               dx1[ix][iz] = d0x*pow((nb-ix)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.001)*(bx[ix][iz]-1.);
           }
           if(ix>padnx-nb)
           {
               alphax[ix][iz] = alphax_max*(padnx-ix)/nb;
               dx1[ix][iz] = d0x*pow((ix+nb-padnx)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.001)*(bx[ix][iz]-1.);
           }
           if(iz<nb)
           {
               alphaz[ix][iz] = alphaz_max*iz/nb;
               dz1[ix][iz] = d0z*pow((nb-iz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.001)*(bz[ix][iz]-1.);
           }
           if(iz>padnz-nb)
           {
               alphaz[ix][iz] = alphaz_max*(padnz-iz)/nb;
               dz1[ix][iz] = d0x*pow((iz+nb-padnz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.001)*(bz[ix][iz]-1.);
           }
           bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
           bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
      }

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->dvv, dK, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(ix=0; ix<padnx; ix++)
		for(iz=0; iz<padnz; iz++)
		{
			K0[ix][iz] = vv[ix][iz]*vv[ix][iz];
			dK[ix][iz] = 2.0*vv[ix][iz]*dK[ix][iz]+dK[ix][iz]*dK[ix][iz];
			K0[ix][iz] += dK[ix][iz];
		}

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){
		sf_warning("###### is=%d ######", is+1);

		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(rp1[0], 0., padnzx*sizeof(float));
		memset(rp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		memset(phi_vx1[0], 0., padnzx*sizeof(float));
		memset(phi_vx2[0], 0., padnzx*sizeof(float));
		memset(phi_vz1[0], 0., padnzx*sizeof(float));
		memset(phi_vz2[0], 0., padnzx*sizeof(float));
		memset(phi_px1[0], 0., padnzx*sizeof(float));
		memset(phi_px2[0], 0., padnzx*sizeof(float));
		memset(phi_pz1[0], 0., padnzx*sizeof(float));
		memset(phi_pz2[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
			if(verb && it%200==0) sf_warning("Forward modeling is=%d; it=%d;", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v; 
				rcd1[is][acpar->r02[is]+ir][it]=p2[rx][rz]-dd[is][acpar->r02[is]+ir][it];
				if(it<abs(sx-rx)*dx/vv[rx][sz]/dt+300+300./(0.1*abs(sx-rx)+1))
					rcd1[is][acpar->r02[is]+ir][it] = 0;
				fcost1 += 0.5*rcd1[is][acpar->r02[is]+ir][it]*rcd1[is][acpar->r02[is]+ir][it];
			}

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					rp2[ix][iz] = -dt/taus[ix][iz]*(rp1[ix][iz]+tau[ix][iz]*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]
								+phi_vz2[ix][iz]+phi_vx2[ix][iz])) + rp1[ix][iz];
					p2[ix][iz] = dt*((tau[ix][iz]+1)*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]+phi_vz2[ix][iz]
								+phi_vx2[ix][iz])+rp2[ix][iz])+p1[ix][iz] + rr[ix*padnz+iz]*array->ww[it];

					if(iz<nb || iz>=padnz-nb)
						phi_vz2[ix][iz] = bz[ix][iz]*phi_vz1[ix][iz] + az1[ix][iz]*dvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_vx2[ix][iz] = bx[ix][iz]*phi_vx1[ix][iz] + ax1[ix][iz]*dvx[ix][iz];
				}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					vz2[ix][iz] = dt*(dpz[ix][iz]+phi_pz2[ix][iz]) +vz1[ix][iz];
					vx2[ix][iz] = dt*(dpx[ix][iz]+phi_px2[ix][iz]) +vx1[ix][iz];
					if(iz<nb || iz>=padnz-nb)
						phi_pz2[ix][iz] = bz[ix][iz]*phi_pz1[ix][iz] + az1[ix][iz]*dpz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_px2[ix][iz] = bx[ix][iz]*phi_px1[ix][iz] + ax1[ix][iz]*dpx[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p1[ix][iz] = p2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					rp1[ix][iz] = rp2[ix][iz];
					phi_px1[ix][iz] = phi_px2[ix][iz];
					phi_pz1[ix][iz] = phi_pz2[ix][iz];
					phi_vx1[ix][iz] = phi_vx2[ix][iz];
					phi_vz1[ix][iz] = phi_vz2[ix][iz];
				}

		} // end of time loop

	}// end of shot loop

	MPI_Barrier(comm);
	MPI_Reduce(&rcd1[0][0][0], &rcd[0][0][0], acpar->ns*nr*nt, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(&rcd[0][0][0], acpar->ns*nr*nt, MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	if(mpipar->cpuid==0){
	FILE *fp=fopen("rcd1","wb");
	fwrite(&rcd[0][0][0],sizeof(float),acpar->ns*nt*nr,fp);
	fclose(fp);}

	MPI_Reduce(&fcost1, fcost, 1,  MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(fcost, 1,  MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	/* release allocated memory */
	free(*p2); free(p2); 
	free(*p1); free(p1);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*rp2); free(rp2);
	free(*rp1); free(rp1);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*phi_vx1); free(phi_vx1);
	free(*phi_vx2); free(phi_vx2);
	free(*phi_vz1); free(phi_vz1);
	free(*phi_vz2); free(phi_vz2);
	free(*phi_px1); free(phi_px1);
	free(*phi_px2); free(phi_px2);
	free(*phi_pz1); free(phi_pz1);
	free(*phi_pz2); free(phi_pz2);
	free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*K0); free(K0); free(*dK); free(dK);
	free(rr); 
	free(**rcd1); free(*rcd1); free(rcd1);
	free(*alphax); free(alphax);
	free(*alphaz); free(alphaz);
	free(*dx1); free(dx1);
	free(*ax1); free(ax1);
	free(*bx); free(bx);
	free(*dz1); free(dz1);
	free(*az1); free(az1);
	free(*bz); free(bz);
}

void adjoint_modeling_cpml(sf_file Fdat, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic adjoint modeling with CPML boundary condition >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **tau, **taus, **dd, **K0, **dK;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **p1, **p2;
	float **rp1, **rp2;
	float **phi_vx1, **phi_vx2, **phi_vz1, **phi_vz2;
	float **phi_px1, **phi_px2, **phi_pz1, **phi_pz2;
	float **phi_rpx1, **phi_rpx2, **phi_rpz1, **phi_rpz2;
	float **dpx, **dpz, **dvx, **dvz;
	float **drpx, **drpz;

	float kax, kaz, alphax_max, alphaz_max, d0x, d0z;
	float **alphax, **dx1, **bx, **ax1;
	float **alphaz, **dz1, **bz, **az1; //CMPL variables

	FILE *swap;

	MPI_Comm comm=MPI_COMM_WORLD;

	swap=fopen("temswap.bin", "wb+");

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

	vv = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	K0=sf_floatalloc2(padnz, padnx);
	dK=sf_floatalloc2(padnz, padnx);
	dd=sf_floatalloc2(nt, nr);

	p1=sf_floatalloc2(padnz, padnx);
	p2=sf_floatalloc2(padnz, padnx);
	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);
	rp1=sf_floatalloc2(padnz, padnx);
	rp2=sf_floatalloc2(padnz, padnx);

	phi_vx1=sf_floatalloc2(padnz, padnx);
	phi_vx2=sf_floatalloc2(padnz, padnx);
	phi_vz1=sf_floatalloc2(padnz, padnx);
	phi_vz2=sf_floatalloc2(padnz, padnx);
	phi_px1=sf_floatalloc2(padnz, padnx);
	phi_px2=sf_floatalloc2(padnz, padnx);
	phi_pz1=sf_floatalloc2(padnz, padnx);
	phi_pz2=sf_floatalloc2(padnz, padnx);
	phi_rpx1=sf_floatalloc2(padnz, padnx);
	phi_rpx2=sf_floatalloc2(padnz, padnx);
	phi_rpz1=sf_floatalloc2(padnz, padnx);
	phi_rpz2=sf_floatalloc2(padnz, padnx);

	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);
	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);
	drpx=sf_floatalloc2(padnz, padnx);
	drpz=sf_floatalloc2(padnz, padnx);
	rr=sf_floatalloc(padnzx);

	/* PML boundary condition*/
	alphax=sf_floatalloc2(padnz, padnx);
	dx1=sf_floatalloc2(padnz, padnx);
	bx=sf_floatalloc2(padnz, padnx);
	ax1=sf_floatalloc2(padnz, padnx);
	alphaz=sf_floatalloc2(padnz, padnx);
	dz1=sf_floatalloc2(padnz, padnx);
	bz=sf_floatalloc2(padnz, padnx);
	az1=sf_floatalloc2(padnz, padnx);
	kax = 1.;
	kaz = 1.;
	alphax_max = SF_PI*20.;
	alphaz_max = SF_PI*20.;
	d0x = -3.8*4.500*log(0.001)/2./nb/dx;
	d0z = -3.8*4.500*log(0.001)/2./nb/dz;
	for(ix=0;ix<padnx;ix++)
		for(iz=0;iz<padnz;iz++)
		{
		   alphax[ix][iz] = 0.0;
		   alphaz[ix][iz] = 0.0;
		   dx1[ix][iz] = 0.0;
		   dz1[ix][iz] = 0.0;
		   bx[ix][iz] = 0.0;
		   bz[ix][iz] = 0.0;
		   ax1[ix][iz] = 0.0;
		   az1[ix][iz] = 0.0;

           if(ix<nb)
           {
               alphax[ix][iz] = alphax_max*ix/nb;
               dx1[ix][iz] = d0x*pow((nb-ix)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.001)*(bx[ix][iz]-1.);
           }
           if(ix>padnx-nb)
           {
               alphax[ix][iz] = alphax_max*(padnx-ix)/nb;
               dx1[ix][iz] = d0x*pow((ix+nb-padnx)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.001)*(bx[ix][iz]-1.);
           }
           if(iz<nb)
           {
               alphaz[ix][iz] = alphaz_max*iz/nb;
               dz1[ix][iz] = d0z*pow((nb-iz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.001)*(bz[ix][iz]-1.);
           }
           if(iz>padnz-nb)
           {
               alphaz[ix][iz] = alphaz_max*(padnz-iz)/nb;
               dz1[ix][iz] = d0x*pow((iz+nb-padnz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.001)*(bz[ix][iz]-1.);
           }
           bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
           bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
      }

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->dvv, dK, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(ix=0; ix<padnx; ix++)
		for(iz=0; iz<padnz; iz++)
		{
			K0[ix][iz] = vv[ix][iz]*vv[ix][iz];
			dK[ix][iz] = 2.0*vv[ix][iz]*dK[ix][iz]+dK[ix][iz]*dK[ix][iz];
			if(iz<nb+10)
				dK[ix][iz] = 0;
		}

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){
		sf_warning("###### is=%d ######", is+1);

		memset(dd[0], 0., nr*nt*sizeof(float));
		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(rp1[0], 0., padnzx*sizeof(float));
		memset(rp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		memset(phi_vx1[0], 0., padnzx*sizeof(float));
		memset(phi_vx2[0], 0., padnzx*sizeof(float));
		memset(phi_vz1[0], 0., padnzx*sizeof(float));
		memset(phi_vz2[0], 0., padnzx*sizeof(float));
		memset(phi_px1[0], 0., padnzx*sizeof(float));
		memset(phi_px2[0], 0., padnzx*sizeof(float));
		memset(phi_pz1[0], 0., padnzx*sizeof(float));
		memset(phi_pz2[0], 0., padnzx*sizeof(float));
		memset(phi_rpx1[0], 0., padnzx*sizeof(float));
		memset(phi_rpx2[0], 0., padnzx*sizeof(float));
		memset(phi_rpz1[0], 0., padnzx*sizeof(float));
		memset(phi_rpz2[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
			if(verb && it%200==0) sf_warning("Backward modeling is=%d; it=%d;", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				dd[acpar->r02[is]+ir][it]=p2[rx][rz];
			}

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p2[ix][iz] = dt*(dvz[ix][iz]+dvx[ix][iz]+phi_vz2[ix][iz]
								+phi_vx2[ix][iz])+p1[ix][iz] + rr[ix*padnz+iz]*array->ww[it];

					rp2[ix][iz] = -dt*(rp1[ix][iz]/taus[ix][iz]-p2[ix][iz]) + rp1[ix][iz];

					if(iz<nb || iz>=padnz-nb)
						phi_vz2[ix][iz] = bz[ix][iz]*phi_vz1[ix][iz] + az1[ix][iz]*dvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_vx2[ix][iz] = bx[ix][iz]*phi_vx1[ix][iz] + ax1[ix][iz]*dvx[ix][iz];
				}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			staggerFD1order2Ddx1(rp2,drpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(rp2,drpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					vz2[ix][iz] = dt*(K0[ix][iz]*((tau[ix][iz]+1.0)*(dpz[ix][iz]+phi_pz2[ix][iz])
								-tau[ix][iz]/taus[ix][iz]*(drpz[ix][iz]+phi_rpz2[ix][iz]))) +vz1[ix][iz];

					vx2[ix][iz] = dt*(K0[ix][iz]*((tau[ix][iz]+1.0)*(dpx[ix][iz]+phi_px2[ix][iz])
								-tau[ix][iz]/taus[ix][iz]*(drpx[ix][iz]+phi_rpx2[ix][iz]))) +vx1[ix][iz];

					if(iz<nb || iz>=padnz-nb)
					{
						phi_pz2[ix][iz] = bz[ix][iz]*phi_pz1[ix][iz] + az1[ix][iz]*dpz[ix][iz];
						phi_rpz2[ix][iz] = bz[ix][iz]*phi_rpz1[ix][iz] + az1[ix][iz]*drpz[ix][iz];
					}
					if(ix<nb || ix>=padnx-nb)
					{
						phi_px2[ix][iz] = bx[ix][iz]*phi_px1[ix][iz] + ax1[ix][iz]*dpx[ix][iz];
						phi_rpx2[ix][iz] = bx[ix][iz]*phi_rpx1[ix][iz] + ax1[ix][iz]*drpx[ix][iz];
					}
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p1[ix][iz] = p2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					rp1[ix][iz] = rp2[ix][iz];
					phi_px1[ix][iz] = phi_px2[ix][iz];
					phi_pz1[ix][iz] = phi_pz2[ix][iz];
					phi_vx1[ix][iz] = phi_vx2[ix][iz];
					phi_vz1[ix][iz] = phi_vz2[ix][iz];
					phi_rpx1[ix][iz] = phi_rpx2[ix][iz];
					phi_rpz1[ix][iz] = phi_rpz2[ix][iz];
				}

		} // end of time loop

		fseeko(swap, is*nr*nt*sizeof(float), SEEK_SET);
		fwrite(dd[0], sizeof(float), nr*nt, swap);
	}// end of shot loop
	fclose(swap);
	MPI_Barrier(comm);

	/* transfer data to Fdat */
	if(mpipar->cpuid==0){
		swap=fopen("temswap.bin", "rb");
		for(is=0; is<acpar->ns; is++){
			fseeko(swap, is*nr*nt*sizeof(float), SEEK_SET);
			fread(dd[0], sizeof(float), nr*nt, swap);
			sf_floatwrite(dd[0], nr*nt, Fdat);
		}
		fclose(swap);
		remove("temswap.bin"); 
	}
	MPI_Barrier(comm);
	
	/* release allocated memory */
	free(*p2); free(p2); 
	free(*p1); free(p1);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*rp2); free(rp2);
	free(*rp1); free(rp1);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*drpx); free(drpx); free(*drpz); free(drpz);
	free(*phi_vx1); free(phi_vx1);
	free(*phi_vx2); free(phi_vx2);
	free(*phi_vz1); free(phi_vz1);
	free(*phi_vz2); free(phi_vz2);
	free(*phi_px1); free(phi_px1);
	free(*phi_px2); free(phi_px2);
	free(*phi_pz1); free(phi_pz1);
	free(*phi_pz2); free(phi_pz2);
	free(*phi_rpx1); free(phi_rpx1);
	free(*phi_rpx2); free(phi_rpx2);
	free(*phi_rpz1); free(phi_rpz1);
	free(*phi_rpz2); free(phi_rpz2);
	free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*K0); free(K0); free(*dK); free(dK);
	free(*dd); free(dd); free(rr); 
	free(*alphax); free(alphax);
	free(*alphaz); free(alphaz);
	free(*dx1); free(dx1);
	free(*ax1); free(ax1);
	free(*bx); free(bx);
	free(*dz1); free(dz1);
	free(*az1); free(az1);
	free(*bz); free(bz);
}

void forward_modeling_ls1(float ***dd, float *fcost, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic forward modeling >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **tau, **taus, **K0, **dK;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **p1, **p2;
	float **rp1, **rp2;
	float **phi_vx1, **phi_vx2, **phi_vz1, **phi_vz2;
	float **phi_px1, **phi_px2, **phi_pz1, **phi_pz2;
	float **dpx, **dpz, **dvx, **dvz;

	float kax, kaz, alphax_max, alphaz_max, d0x, d0z;
	float **alphax, **dx1, **bx, **ax1;
	float **alphaz, **dz1, **bz, **az1; //CMPL variables


	float fcost1 = 0.0;

	MPI_Comm comm=MPI_COMM_WORLD;

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	float ***rcd1; // temporary variables for MPI_Reduce
	rcd1=sf_floatalloc3(nt,nr,acpar->ns);

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

	vv = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	K0=sf_floatalloc2(padnz, padnx);
	dK=sf_floatalloc2(padnz, padnx);

	p1=sf_floatalloc2(padnz, padnx);
	p2=sf_floatalloc2(padnz, padnx);
	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);
	rp1=sf_floatalloc2(padnz, padnx);
	rp2=sf_floatalloc2(padnz, padnx);

	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);
	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);

	phi_vx1=sf_floatalloc2(padnz, padnx);
	phi_vx2=sf_floatalloc2(padnz, padnx);
	phi_vz1=sf_floatalloc2(padnz, padnx);
	phi_vz2=sf_floatalloc2(padnz, padnx);
	phi_px1=sf_floatalloc2(padnz, padnx);
	phi_px2=sf_floatalloc2(padnz, padnx);
	phi_pz1=sf_floatalloc2(padnz, padnx);
	phi_pz2=sf_floatalloc2(padnz, padnx);

	rr=sf_floatalloc(padnzx);

	/* PML boundary condition*/
	alphax=sf_floatalloc2(padnz, padnx);
	dx1=sf_floatalloc2(padnz, padnx);
	bx=sf_floatalloc2(padnz, padnx);
	ax1=sf_floatalloc2(padnz, padnx);
	alphaz=sf_floatalloc2(padnz, padnx);
	dz1=sf_floatalloc2(padnz, padnx);
	bz=sf_floatalloc2(padnz, padnx);
	az1=sf_floatalloc2(padnz, padnx);
	kax = 1.;
	kaz = 1.;
	alphax_max = SF_PI*20.;
	alphaz_max = SF_PI*20.;
	d0x = -3.8*4.500*log(0.001)/2./nb/dx;
	d0z = -3.8*4.500*log(0.001)/2./nb/dz;
	for(ix=0;ix<padnx;ix++)
		for(iz=0;iz<padnz;iz++)
		{
		   alphax[ix][iz] = 0.0;
		   alphaz[ix][iz] = 0.0;
		   dx1[ix][iz] = 0.0;
		   dz1[ix][iz] = 0.0;
		   bx[ix][iz] = 0.0;
		   bz[ix][iz] = 0.0;
		   ax1[ix][iz] = 0.0;
		   az1[ix][iz] = 0.0;

           if(ix<nb)
           {
               alphax[ix][iz] = alphax_max*ix/nb;
               dx1[ix][iz] = d0x*pow((nb-ix)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.000001)*(bx[ix][iz]-1.);
           }
           if(ix>padnx-nb)
           {
               alphax[ix][iz] = alphax_max*(padnx-ix)/nb;
               dx1[ix][iz] = d0x*pow((ix+nb-padnx)*1.0/nb,2);
               bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
               ax1[ix][iz] = dx1[ix][iz]/kax/(dx1[ix][iz]+kax*alphax[ix][iz]+0.000001)*(bx[ix][iz]-1.);
           }
           if(iz<nb)
           {
               alphaz[ix][iz] = alphaz_max*iz/nb;
               dz1[ix][iz] = d0z*pow((nb-iz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.000001)*(bz[ix][iz]-1.);
           }
           if(iz>padnz-nb)
           {
               alphaz[ix][iz] = alphaz_max*(padnz-iz)/nb;
               dz1[ix][iz] = d0x*pow((iz+nb-padnz)*1.0/nb,2);
               bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
               az1[ix][iz] = dz1[ix][iz]/kaz/(dz1[ix][iz]+kaz*alphaz[ix][iz]+0.000001)*(bz[ix][iz]-1.);
           }
           bx[ix][iz] = exp(-(dx1[ix][iz]/kax+alphax[ix][iz])*dt);
           bz[ix][iz] = exp(-(dz1[ix][iz]/kaz+alphaz[ix][iz])*dt);
      }

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->dvv, dK, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(ix=0; ix<padnx; ix++)
		for(iz=0; iz<padnz; iz++)
		{
			K0[ix][iz] = vv[ix][iz]*vv[ix][iz];
			dK[ix][iz] = 2.0*vv[ix][iz]*dK[ix][iz]+dK[ix][iz]*dK[ix][iz];
			K0[ix][iz] += dK[ix][iz];
		}

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){
		sf_warning("###### is=%d ######", is+1);

		memset(p1[0], 0., padnzx*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(rp1[0], 0., padnzx*sizeof(float));
		memset(rp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		memset(phi_vx1[0], 0., padnzx*sizeof(float));
		memset(phi_vx2[0], 0., padnzx*sizeof(float));
		memset(phi_vz1[0], 0., padnzx*sizeof(float));
		memset(phi_vz2[0], 0., padnzx*sizeof(float));
		memset(phi_px1[0], 0., padnzx*sizeof(float));
		memset(phi_px2[0], 0., padnzx*sizeof(float));
		memset(phi_pz1[0], 0., padnzx*sizeof(float));
		memset(phi_pz2[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
			if(verb && it%200==0) sf_warning("Forward modeling is=%d; it=%d;", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				rcd1[is][acpar->r02[is]+ir][it]=p2[rx][rz]-dd[is][acpar->r02[is]+ir][it];
				if(it<abs(sx-rx)*dx/vv[rx][sz]/dt+300/*+300./(0.1*abs(sx-rx)+1)*/)
					rcd1[is][acpar->r02[is]+ir][it] = 0;
				fcost1 += 0.5*rcd1[is][acpar->r02[is]+ir][it]*rcd1[is][acpar->r02[is]+ir][it];
			}

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					rp2[ix][iz] = -dt/taus[ix][iz]*(rp1[ix][iz]+tau[ix][iz]*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]
								+phi_vz2[ix][iz]+phi_vx2[ix][iz])) + rp1[ix][iz];
					p2[ix][iz] = dt*((tau[ix][iz]+1)*K0[ix][iz]*(dvz[ix][iz]+dvx[ix][iz]+phi_vz2[ix][iz]
								+phi_vx2[ix][iz])+rp2[ix][iz])+p1[ix][iz] + rr[ix*padnz+iz]*array->ww[it];

					if(iz<nb || iz>=padnz-nb)
						phi_vz2[ix][iz] = bz[ix][iz]*phi_vz1[ix][iz] + az1[ix][iz]*dvz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_vx2[ix][iz] = bx[ix][iz]*phi_vx1[ix][iz] + ax1[ix][iz]*dvx[ix][iz];
				}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					vz2[ix][iz] = dt*(dpz[ix][iz]+phi_pz2[ix][iz]) +vz1[ix][iz];
					vx2[ix][iz] = dt*(dpx[ix][iz]+phi_px2[ix][iz]) +vx1[ix][iz];
					if(iz<nb || iz>=padnz-nb)
						phi_pz2[ix][iz] = bz[ix][iz]*phi_pz1[ix][iz] + az1[ix][iz]*dpz[ix][iz];
					if(ix<nb || ix>=padnx-nb)
						phi_px2[ix][iz] = bx[ix][iz]*phi_px1[ix][iz] + ax1[ix][iz]*dpx[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					p1[ix][iz] = p2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					rp1[ix][iz] = rp2[ix][iz];
					phi_px1[ix][iz] = phi_px2[ix][iz];
					phi_pz1[ix][iz] = phi_pz2[ix][iz];
					phi_vx1[ix][iz] = phi_vx2[ix][iz];
					phi_vz1[ix][iz] = phi_vz2[ix][iz];
				}

		} // end of time loop

	}// end of shot loop

	MPI_Barrier(comm);

	/* record and sillum reduction */
/*	MPI_Reduce(&rcd1[0][0][0], &rcd[0][0][0], acpar->ns*nr*nt, MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(&rcd[0][0][0], acpar->ns*nr*nt, MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);
*/
	MPI_Reduce(&fcost1, fcost, 1,  MPI_FLOAT, MPI_SUM, 0, comm);
	MPI_Bcast(fcost, 1,  MPI_FLOAT, 0, comm);
	MPI_Barrier(comm);

	sf_warning("fcost2=%f",fcost1);

	/* release allocated memory */
	free(*p2); free(p2); 
	free(*p1); free(p1);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*rp2); free(rp2);
	free(*rp1); free(rp1);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*phi_vx1); free(phi_vx1);
	free(*phi_vx2); free(phi_vx2);
	free(*phi_vz1); free(phi_vz1);
	free(*phi_vz2); free(phi_vz2);
	free(*phi_px1); free(phi_px1);
	free(*phi_px2); free(phi_px2);
	free(*phi_pz1); free(phi_pz1);
	free(*phi_pz2); free(phi_pz2);
	free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*K0); free(K0); free(*dK); free(dK);
	free(rr); 
	free(*alphax); free(alphax);
	free(*alphaz); free(alphaz);
	free(*dx1); free(dx1);
	free(*ax1); free(ax1);
	free(*bx); free(bx);
	free(*dz1); free(dz1);
	free(*az1); free(az1);
	free(*bz); free(bz);
	free(**rcd1); free(*rcd1); free(rcd1);
	MPI_Barrier(comm);
}

void forward_modeling_pml(sf_file Fdat, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic forward modeling with PML boundary condition>*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **tau, **taus, **dd;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **xp1, **xp2, **zp1, **zp2, **p2;
	float **xrp1, **xrp2, **zrp1, **zrp2;
	float **dpx, **dpz, **dvx, **dvz;

	float vmax;
	float *alpha; //PML boundary condition

	FILE *swap;

	MPI_Comm comm=MPI_COMM_WORLD;

	swap=fopen("temswap.bin", "wb+");

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

	vv = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	dd=sf_floatalloc2(nt, nr);

	p2=sf_floatalloc2(padnz, padnx);
	xp1=sf_floatalloc2(padnz, padnx);
	xp2=sf_floatalloc2(padnz, padnx);
	zp1=sf_floatalloc2(padnz, padnx);
	zp2=sf_floatalloc2(padnz, padnx);

	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);

	xrp1=sf_floatalloc2(padnz, padnx);
	xrp2=sf_floatalloc2(padnz, padnx);
	zrp1=sf_floatalloc2(padnz, padnx);
	zrp2=sf_floatalloc2(padnz, padnx);

	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);
	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);

	rr=sf_floatalloc(padnzx);

	/* PML boundary condition*/
	alpha=sf_floatalloc(nb);
	vmax = 4.5;
	for(ix=0;ix<nb;ix++)
		alpha[ix] = 3.5*vmax/dx*pow(ix*1.0/nb,4);

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){
		sf_warning("###### is=%d ######", is+1);

		memset(dd[0], 0., nr*nt*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(xp1[0], 0., padnzx*sizeof(float));
		memset(xp2[0], 0., padnzx*sizeof(float));
		memset(zp1[0], 0., padnzx*sizeof(float));
		memset(zp2[0], 0., padnzx*sizeof(float));
		memset(xrp1[0], 0., padnzx*sizeof(float));
		memset(xrp2[0], 0., padnzx*sizeof(float));
		memset(zrp1[0], 0., padnzx*sizeof(float));
		memset(zrp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		
		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
			if(verb) sf_warning("Modeling is=%d; it=%d;", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				dd[acpar->r02[is]+ir][it]=p2[rx][rz];
			}

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					if(iz<=nb)
						zrp2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(-dt/taus[ix][iz]*(zrp1[ix][iz]+tau[ix][iz]*dvz[ix][iz]*vv[ix][iz]*vv[ix][iz]) + exp(-alpha[nb-iz-1]*dt/2.)*zrp1[ix][iz]);
					else if(iz>=padnz-nb)
						zrp2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(-dt/taus[ix][iz]*(zrp1[ix][iz]+tau[ix][iz]*dvz[ix][iz]*vv[ix][iz]*vv[ix][iz]) + exp(-alpha[nb+iz-padnz]*dt/2.)*zrp1[ix][iz]);
					else
						zrp2[ix][iz] = -dt/taus[ix][iz]*(zrp1[ix][iz]+tau[ix][iz]*dvz[ix][iz]*vv[ix][iz]*vv[ix][iz]) + zrp1[ix][iz];

					if(ix<=nb)
						xrp2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(-dt/taus[ix][iz]*(xrp1[ix][iz]+tau[ix][iz]*dvx[ix][iz]*vv[ix][iz]*vv[ix][iz]) + exp(-alpha[nb-ix-1]*dt/2.)*xrp1[ix][iz]);
					else if(ix>=padnx-nb)
						xrp2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(-dt/taus[ix][iz]*(xrp1[ix][iz]+tau[ix][iz]*dvx[ix][iz]*vv[ix][iz]*vv[ix][iz]) + exp(-alpha[nb+ix-padnx]*dt/2.)*xrp1[ix][iz]);
					else
						xrp2[ix][iz] = -dt/taus[ix][iz]*(xrp1[ix][iz]+tau[ix][iz]*dvx[ix][iz]*vv[ix][iz]*vv[ix][iz]) + xrp1[ix][iz];

					if(iz<=nb)
						zp2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(dt*((tau[ix][iz]+1)*dvz[ix][iz]*vv[ix][iz]*vv[ix][iz]+zrp2[ix][iz])+exp(-alpha[nb-iz-1]*dt/2.)*zp1[ix][iz]);
					else if(iz>=padnz-nb)
						zp2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(dt*((tau[ix][iz]+1)*dvz[ix][iz]*vv[ix][iz]*vv[ix][iz]+zrp2[ix][iz])+exp(-alpha[nb+iz-padnz]*dt/2.)*zp1[ix][iz]);
					else
						zp2[ix][iz] = dt*((tau[ix][iz]+1)*dvz[ix][iz]*vv[ix][iz]*vv[ix][iz]+zrp2[ix][iz])+zp1[ix][iz];

					if(ix<=nb)
						xp2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(dt*((tau[ix][iz]+1)*dvx[ix][iz]*vv[ix][iz]*vv[ix][iz]+xrp2[ix][iz])+exp(-alpha[nb-ix-1]*dt/2.)*xp1[ix][iz]);
					else if(ix>=padnx-nb)
						xp2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(dt*((tau[ix][iz]+1)*dvx[ix][iz]*vv[ix][iz]*vv[ix][iz]+xrp2[ix][iz])+exp(-alpha[nb+ix-padnx]*dt/2.)*xp1[ix][iz]);
					else
						xp2[ix][iz] = dt*((tau[ix][iz]+1)*dvx[ix][iz]*vv[ix][iz]*vv[ix][iz]+xrp2[ix][iz])+xp1[ix][iz];

					p2[ix][iz] = xp2[ix][iz] + zp2[ix][iz] + rr[ix*padnz+iz]*array->ww[it];
				}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					if(iz<=nb)
						vz2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(dt*dpz[ix][iz] +exp(-alpha[nb-iz-1]*dt/2.)*vz1[ix][iz]);
					else if(iz>=padnz-nb)
						vz2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(dt*dpz[ix][iz] +exp(-alpha[nb+iz-padnz]*dt/2.)*vz1[ix][iz]);
					else
						vz2[ix][iz] = dt*dpz[ix][iz] +vz1[ix][iz];

					if(ix<=nb)
						vx2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(dt*dpx[ix][iz] +exp(-alpha[nb-ix-1]*dt/2.)*vx1[ix][iz]);
					else if(ix>=padnx-nb)
						vx2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(dt*dpx[ix][iz] +exp(-alpha[nb+ix-padnx]*dt/2.)*vx1[ix][iz]);
					else
						vx2[ix][iz] = dt*dpx[ix][iz] +vx1[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					xp1[ix][iz] = xp2[ix][iz];
					zp1[ix][iz] = zp2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					xrp1[ix][iz] = xrp2[ix][iz];
					zrp1[ix][iz] = zrp2[ix][iz];
				}
		} // end of time loop

		fseeko(swap, is*nr*nt*sizeof(float), SEEK_SET);
		fwrite(dd[0], sizeof(float), nr*nt, swap);
	}// end of shot loop
	fclose(swap);
	MPI_Barrier(comm);

	/* transfer data to Fdat */
	if(mpipar->cpuid==0){
		swap=fopen("temswap.bin", "rb");
		for(is=0; is<acpar->ns; is++){
			fseeko(swap, is*nr*nt*sizeof(float), SEEK_SET);
			fread(dd[0], sizeof(float), nr*nt, swap);
			sf_floatwrite(dd[0], nr*nt, Fdat);
		}
		fclose(swap);
		remove("temswap.bin");
	}
	MPI_Barrier(comm);
	
	/* release allocated memory */
	free(*p2); free(p2); 
	free(*xp1); free(xp1);
	free(*xp2); free(xp2);
	free(*zp1);free(zp1);
	free(*zp2); free(zp2);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*xrp2); free(xrp2);
	free(*xrp1); free(xrp1); free(*zrp2); free(zrp2);
	free(*zrp1); free(zrp1);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*vv); free(vv);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*dd); free(dd); free(rr); 
}

void forward_pml(sf_file Fdat, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< visco-acoustic forward progation for gradient calculation >*/
{
	int ix, iz, is, ir, it;
	int sx, rx, sz, rz, rectx, rectz;
	int nz, nx, padnz, padnx, padnzx, nt, nr, nb;

	float dx, dz, dt, idt;
	float **vv, **K0, **dK, **tau, **taus, **dd;
	float *rr;
	float **vx1, **vx2, **vz1, **vz2;
	float **xp1, **xp2, **zp1, **zp2, **p2;
	float **xrp1, **xrp2, **zrp1, **zrp2;
	float **dpx, **dpz, **dvx, **dvz;

	float **dvx1, **dvx2, **dvz1, **dvz2;
	float **dxp1, **dxp2, **dzp1, **dzp2, **dp2;
	float **dxrp1, **dxrp2, **dzrp1, **dzrp2; //delta_u
	float **ddvx, **ddvz;

	float vmax;
	float *alpha; //PML boundary condition

	FILE *swap;

	MPI_Comm comm=MPI_COMM_WORLD;

	swap=fopen("temswap.bin", "wb+");

	padnz=acpar->padnz;
	padnx=acpar->padnx;
	padnzx=padnz*padnx;
	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	nr=acpar->nr;
	nb=acpar->nb;
	sz=acpar->sz;
	rz=acpar->rz;
	rectx=soupar->rectx;
	rectz=soupar->rectz;

	dx=acpar->dx;
	dz=acpar->dz;
	dt=acpar->dt;
	idt=1./acpar->dt;

	vv = sf_floatalloc2(padnz, padnx);
	K0 = sf_floatalloc2(padnz, padnx);
	dK = sf_floatalloc2(padnz, padnx);
	tau= sf_floatalloc2(padnz, padnx);
	taus=sf_floatalloc2(padnz, padnx);
	dd=sf_floatalloc2(nt, nr);

	p2=sf_floatalloc2(padnz, padnx);
	xp1=sf_floatalloc2(padnz, padnx);
	xp2=sf_floatalloc2(padnz, padnx);
	zp1=sf_floatalloc2(padnz, padnx);
	zp2=sf_floatalloc2(padnz, padnx);

	vx1=sf_floatalloc2(padnz, padnx);
	vx2=sf_floatalloc2(padnz, padnx);
	vz1=sf_floatalloc2(padnz, padnx);
	vz2=sf_floatalloc2(padnz, padnx);

	xrp1=sf_floatalloc2(padnz, padnx);
	xrp2=sf_floatalloc2(padnz, padnx);
	zrp1=sf_floatalloc2(padnz, padnx);
	zrp2=sf_floatalloc2(padnz, padnx);

	dpx=sf_floatalloc2(padnz, padnx);
	dpz=sf_floatalloc2(padnz, padnx);
	dvx=sf_floatalloc2(padnz, padnx);
	dvz=sf_floatalloc2(padnz, padnx);
	ddvx=sf_floatalloc2(padnz, padnx);
	ddvz=sf_floatalloc2(padnz, padnx);

	dp2=sf_floatalloc2(padnz, padnx);
	dxp1=sf_floatalloc2(padnz, padnx);
	dxp2=sf_floatalloc2(padnz, padnx);
	dzp1=sf_floatalloc2(padnz, padnx);
	dzp2=sf_floatalloc2(padnz, padnx);

	dvx1=sf_floatalloc2(padnz, padnx);
	dvx2=sf_floatalloc2(padnz, padnx);
	dvz1=sf_floatalloc2(padnz, padnx);
	dvz2=sf_floatalloc2(padnz, padnx);

	dxrp1=sf_floatalloc2(padnz, padnx);
	dxrp2=sf_floatalloc2(padnz, padnx);
	dzrp1=sf_floatalloc2(padnz, padnx);
	dzrp2=sf_floatalloc2(padnz, padnx);

	rr=sf_floatalloc(padnzx);

	/* PML boundary condition*/
	alpha=sf_floatalloc(nb);
	vmax = 4.500;
	for(ix=0;ix<nb;ix++)
		alpha[ix] = 4.0*vmax/dx*pow(ix*1.0/nb,4);

	/* padding and convert vector to 2-d array */
	pad2d(array->vv, vv, nz, nx, nb);
	pad2d(array->dvv, dK, nz, nx, nb);
	pad2d(array->tau, tau, nz, nx, nb);
	pad2d(array->taus, taus, nz, nx, nb);

	for(ix=0; ix<padnx; ix++)
		for(iz=0; iz<padnz; iz++)
		{
			K0[ix][iz] = vv[ix][iz]*vv[ix][iz];
			dK[ix][iz] = 2.0*vv[ix][iz]*dK[ix][iz]+dK[ix][iz]*dK[ix][iz];
			if(iz<nb+10)
				dK[ix][iz] = 0;
		}

	for(is=mpipar->cpuid; is<acpar->ns; is+=mpipar->numprocs){
		sf_warning("###### is=%d ######", is+1);

		memset(dd[0], 0., nr*nt*sizeof(float));
		memset(p2[0], 0., padnzx*sizeof(float));
		memset(xp1[0], 0., padnzx*sizeof(float));
		memset(xp2[0], 0., padnzx*sizeof(float));
		memset(zp1[0], 0., padnzx*sizeof(float));
		memset(zp2[0], 0., padnzx*sizeof(float));
		memset(xrp1[0], 0., padnzx*sizeof(float));
		memset(xrp2[0], 0., padnzx*sizeof(float));
		memset(zrp1[0], 0., padnzx*sizeof(float));
		memset(zrp2[0], 0., padnzx*sizeof(float));
		memset(vx1[0], 0., padnzx*sizeof(float));
		memset(vx2[0], 0., padnzx*sizeof(float));
		memset(vz1[0], 0., padnzx*sizeof(float));
		memset(vz2[0], 0., padnzx*sizeof(float));
		
		memset(dp2[0], 0., padnzx*sizeof(float));
		memset(dxp1[0], 0., padnzx*sizeof(float));
		memset(dxp2[0], 0., padnzx*sizeof(float));
		memset(dzp1[0], 0., padnzx*sizeof(float));
		memset(dzp2[0], 0., padnzx*sizeof(float));
		memset(dxrp1[0], 0., padnzx*sizeof(float));
		memset(dxrp2[0], 0., padnzx*sizeof(float));
		memset(dzrp1[0], 0., padnzx*sizeof(float));
		memset(dzrp2[0], 0., padnzx*sizeof(float));
		memset(dvx1[0], 0., padnzx*sizeof(float));
		memset(dvx2[0], 0., padnzx*sizeof(float));
		memset(dvz1[0], 0., padnzx*sizeof(float));
		memset(dvz2[0], 0., padnzx*sizeof(float));

		sx=acpar->s0_v+is*acpar->ds_v;
		source_map(sx, sz, rectx, rectz, padnx, padnz, padnzx, rr);

		for(it=0; it<nt; it++){
			if(verb && it%200==0) sf_warning("Modeling is=%d; it=%d;", is+1, it);

			/* output data */
			for(ir=0; ir<acpar->nr2[is]; ir++){
				rx=acpar->r0_v[is]+ir*acpar->dr_v;
				dd[acpar->r02[is]+ir][it]=dp2[rx][rz];
			}

			staggerFD1order2Ddx1(vx2,dvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(vz2,dvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					if(iz<=nb)
						zrp2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(-dt/taus[ix][iz]*(zrp1[ix][iz]+tau[ix][iz]*dvz[ix][iz]*K0[ix][iz]) + exp(-alpha[nb-iz-1]*dt/2.)*zrp1[ix][iz]);
					else if(iz>=padnz-nb)
						zrp2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(-dt/taus[ix][iz]*(zrp1[ix][iz]+tau[ix][iz]*dvz[ix][iz]*K0[ix][iz]) + exp(-alpha[nb+iz-padnz]*dt/2.)*zrp1[ix][iz]);
					else
						zrp2[ix][iz] = -dt/taus[ix][iz]*(zrp1[ix][iz]+tau[ix][iz]*dvz[ix][iz]*K0[ix][iz]) + zrp1[ix][iz];

					if(ix<=nb)
						xrp2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(-dt/taus[ix][iz]*(xrp1[ix][iz]+tau[ix][iz]*dvx[ix][iz]*K0[ix][iz]) + exp(-alpha[nb-ix-1]*dt/2.)*xrp1[ix][iz]);
					else if(ix>=padnx-nb)
						xrp2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(-dt/taus[ix][iz]*(xrp1[ix][iz]+tau[ix][iz]*dvx[ix][iz]*K0[ix][iz]) + exp(-alpha[nb+ix-padnx]*dt/2.)*xrp1[ix][iz]);
					else
						xrp2[ix][iz] = -dt/taus[ix][iz]*(xrp1[ix][iz]+tau[ix][iz]*dvx[ix][iz]*K0[ix][iz]) + xrp1[ix][iz];

					if(iz<=nb)
						zp2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(dt*((tau[ix][iz]+1)*dvz[ix][iz]*K0[ix][iz]+zrp2[ix][iz])+exp(-alpha[nb-iz-1]*dt/2.)*zp1[ix][iz]);
					else if(iz>=padnz-nb)
						zp2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(dt*((tau[ix][iz]+1)*dvz[ix][iz]*K0[ix][iz]+zrp2[ix][iz])+exp(-alpha[nb+iz-padnz]*dt/2.)*zp1[ix][iz]);
					else
						zp2[ix][iz] = dt*((tau[ix][iz]+1)*dvz[ix][iz]*K0[ix][iz]+zrp2[ix][iz])+zp1[ix][iz];

					if(ix<=nb)
						xp2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(dt*((tau[ix][iz]+1)*dvx[ix][iz]*K0[ix][iz]+xrp2[ix][iz])+exp(-alpha[nb-ix-1]*dt/2.)*xp1[ix][iz]);
					else if(ix>=padnx-nb)
						xp2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(dt*((tau[ix][iz]+1)*dvx[ix][iz]*K0[ix][iz]+xrp2[ix][iz])+exp(-alpha[nb+ix-padnx]*dt/2.)*xp1[ix][iz]);
					else
						xp2[ix][iz] = dt*((tau[ix][iz]+1)*dvx[ix][iz]*K0[ix][iz]+xrp2[ix][iz])+xp1[ix][iz];

					p2[ix][iz] = xp2[ix][iz] + zp2[ix][iz] + rr[ix*padnz+iz]*array->ww[it];
				}

			staggerFD1order2Ddx1(p2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(p2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					if(iz<=nb)
						vz2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(dt*dpz[ix][iz] +exp(-alpha[nb-iz-1]*dt/2.)*vz1[ix][iz]);
					else if(iz>=padnz-nb)
						vz2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(dt*dpz[ix][iz] +exp(-alpha[nb+iz-padnz]*dt/2.)*vz1[ix][iz]);
					else
						vz2[ix][iz] = dt*dpz[ix][iz] +vz1[ix][iz];

					if(ix<=nb)
						vx2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(dt*dpx[ix][iz] +exp(-alpha[nb-ix-1]*dt/2.)*vx1[ix][iz]);
					else if(ix>=padnx-nb)
						vx2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(dt*dpx[ix][iz] +exp(-alpha[nb+ix-padnx]*dt/2.)*vx1[ix][iz]);
					else
						vx2[ix][iz] = dt*dpx[ix][iz] +vx1[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					xp1[ix][iz] = xp2[ix][iz];
					zp1[ix][iz] = zp2[ix][iz];
					vx1[ix][iz] = vx2[ix][iz];
					vz1[ix][iz] = vz2[ix][iz];
					xrp1[ix][iz] = xrp2[ix][iz];
					zrp1[ix][iz] = zrp2[ix][iz];
				}

		// begin Born modelling
			staggerFD1order2Ddx1(dvx2,ddvx,padnx,padnz,dx,dz,5,0,1);//dvx/dx forward_stagger
			staggerFD1order2Ddx1(dvz2,ddvz,padnx,padnz,dx,dz,5,1,1); //dvz/dz forward_stagger
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					if(iz<=nb)
						dzrp2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(-dt/taus[ix][iz]*(dzrp1[ix][iz]+tau[ix][iz]*(ddvz[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvz[ix][iz]*dK[ix][iz])) + exp(-alpha[nb-iz-1]*dt/2.)*dzrp1[ix][iz]);
					else if(iz>=padnz-nb)
						dzrp2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(-dt/taus[ix][iz]*(dzrp1[ix][iz]+tau[ix][iz]*(ddvz[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvz[ix][iz]*dK[ix][iz])) + exp(-alpha[nb+iz-padnz]*dt/2.)*dzrp1[ix][iz]);
					else
						dzrp2[ix][iz] = -dt/taus[ix][iz]*(dzrp1[ix][iz]+tau[ix][iz]*(ddvz[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvz[ix][iz]*dK[ix][iz])) + dzrp1[ix][iz];

					if(ix<=nb)
						dxrp2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(-dt/taus[ix][iz]*(dxrp1[ix][iz]+tau[ix][iz]*(ddvx[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvx[ix][iz]*dK[ix][iz])) + exp(-alpha[nb-ix-1]*dt/2.)*dxrp1[ix][iz]);
					else if(ix>=padnx-nb)
						dxrp2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(-dt/taus[ix][iz]*(dxrp1[ix][iz]+tau[ix][iz]*(ddvx[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvx[ix][iz]*dK[ix][iz])) + exp(-alpha[nb+ix-padnx]*dt/2.)*dxrp1[ix][iz]);
					else
						dxrp2[ix][iz] = -dt/taus[ix][iz]*(dxrp1[ix][iz]+tau[ix][iz]*(ddvx[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvx[ix][iz]*dK[ix][iz])) + dxrp1[ix][iz];

					if(iz<=nb)
						dzp2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(dt*((tau[ix][iz]+1)*(ddvz[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvz[ix][iz]*dK[ix][iz])+dzrp2[ix][iz])+exp(-alpha[nb-iz-1]*dt/2.)*dzp1[ix][iz]);
					else if(iz>=padnz-nb)
						dzp2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(dt*((tau[ix][iz]+1)*(ddvz[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvz[ix][iz]*dK[ix][iz])+dzrp2[ix][iz])+exp(-alpha[nb+iz-padnz]*dt/2.)*dzp1[ix][iz]);
					else
						dzp2[ix][iz] = dt*((tau[ix][iz]+1)*(ddvz[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvz[ix][iz]*dK[ix][iz])+dzrp2[ix][iz])+dzp1[ix][iz];

					if(ix<=nb)
						dxp2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(dt*((tau[ix][iz]+1)*(ddvx[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvx[ix][iz]*dK[ix][iz])+dxrp2[ix][iz])+exp(-alpha[nb-ix-1]*dt/2.)*dxp1[ix][iz]);
					else if(ix>=padnx-nb)
						dxp2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(dt*((tau[ix][iz]+1)*(ddvx[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvx[ix][iz]*dK[ix][iz])+dxrp2[ix][iz])+exp(-alpha[nb+ix-padnx]*dt/2.)*dxp1[ix][iz]);
					else
						dxp2[ix][iz] = dt*((tau[ix][iz]+1)*(ddvx[ix][iz]*(K0[ix][iz]+dK[ix][iz])+dvx[ix][iz]*dK[ix][iz])+dxrp2[ix][iz])+dxp1[ix][iz];

					dp2[ix][iz] = dxp2[ix][iz] + dzp2[ix][iz];
				}

			staggerFD1order2Ddx1(dp2,dpx,padnx,padnz,dx,dz,5,0,0);
			staggerFD1order2Ddx1(dp2,dpz,padnx,padnz,dx,dz,5,1,0);
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					if(iz<=nb)
						dvz2[ix][iz] = exp(-alpha[nb-iz-1]*dt/2.)*(dt*dpz[ix][iz] +exp(-alpha[nb-iz-1]*dt/2.)*dvz1[ix][iz]);
					else if(iz>=padnz-nb)
						dvz2[ix][iz] = exp(-alpha[nb+iz-padnz]*dt/2.)*(dt*dpz[ix][iz] +exp(-alpha[nb+iz-padnz]*dt/2.)*dvz1[ix][iz]);
					else
						dvz2[ix][iz] = dt*dpz[ix][iz] + dvz1[ix][iz];

					if(ix<=nb)
						dvx2[ix][iz] = exp(-alpha[nb-ix-1]*dt/2.)*(dt*dpx[ix][iz] +exp(-alpha[nb-ix-1]*dt/2.)*dvx1[ix][iz]);
					else if(ix>=padnx-nb)
						dvx2[ix][iz] = exp(-alpha[nb+ix-padnx]*dt/2.)*(dt*dpx[ix][iz] +exp(-alpha[nb+ix-padnx]*dt/2.)*dvx1[ix][iz]);
					else
						dvx2[ix][iz] = dt*dpx[ix][iz] + dvx1[ix][iz];
				}

			/* swap wavefield pointer of different time steps */
			for(ix=0; ix<padnx; ix++)
				for(iz=0; iz<padnz; iz++)
				{
					dxp1[ix][iz] = dxp2[ix][iz];
					dzp1[ix][iz] = dzp2[ix][iz];
					dvx1[ix][iz] = dvx2[ix][iz];
					dvz1[ix][iz] = dvz2[ix][iz];
					dxrp1[ix][iz] = dxrp2[ix][iz];
					dzrp1[ix][iz] = dzrp2[ix][iz];
				}

		} // end of time loop

		fseeko(swap, is*nr*nt*sizeof(float), SEEK_SET);
		fwrite(dd[0], sizeof(float), nr*nt, swap);
	}// end of shot loop
	fclose(swap);
	MPI_Barrier(comm);

	/* transfer data to Fdat */
	if(mpipar->cpuid==0){
		swap=fopen("temswap.bin", "rb");
		for(is=0; is<acpar->ns; is++){
			fseeko(swap, is*nr*nt*sizeof(float), SEEK_SET);
			fread(dd[0], sizeof(float), nr*nt, swap);
			sf_floatwrite(dd[0], nr*nt, Fdat);
		}
		fclose(swap);
		remove("temswap.bin");
	}
	MPI_Barrier(comm);
	
	/* release allocated memory */
	free(*p2); free(p2); 
	free(*xp1); free(xp1);
	free(*xp2); free(xp2);
	free(*zp1);free(zp1);
	free(*zp2); free(zp2);
	free(*vx1); free(vx1); free(*vx2); free(vx2);
	free(*vz1); free(vz1); free(*vz2); free(vz2);
	free(*xrp2); free(xrp2);
	free(*xrp1); free(xrp1); free(*zrp2); free(zrp2);
	free(*zrp1); free(zrp1);
	free(*dp2); free(dp2); 
	free(*dxp1); free(dxp1);
	free(*dxp2); free(dxp2);
	free(*dzp1);free(dzp1);
	free(*dzp2); free(dzp2);
	free(*dvx1); free(dvx1); free(*dvx2); free(dvx2);
	free(*dvz1); free(dvz1); free(*dvz2); free(dvz2);
	free(*dxrp2); free(dxrp2);
	free(*dxrp1); free(dxrp1); free(*dzrp2); free(dzrp2);
	free(*dzrp1); free(dzrp1);
	free(*dpx); free(dpx); free(*dpz); free(dpz);
	free(*dvx); free(dvx); free(*dvz); free(dvz);
	free(*ddvx); free(ddvx); free(*ddvz); free(ddvz);
	free(*vv); free(vv);
	free(*K0); free(K0);
	free(*dK); free(dK);
	free(*tau); free(tau); free(*taus); free(taus);
	free(*dd); free(dd); free(rr); 
}

