/* Visco-acoustic FWI */
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
#include <omp.h>
#include <mpi.h>
#include "Qfwi_commons.h"
#include "Qfwi_gradient.h"
#include "Qfwi_modeling.h"
/*^*/

void line_search(float **, float * , float ***, sf_mpi *, sf_sou , sf_acqui, sf_vec , sf_fwi , sf_optim , int* , bool);

void qfwi(sf_file Fdat, sf_file Finv, sf_file Fgrad, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, sf_fwi fwipar, sf_optim optpar, bool verb, int media)
/*< fwi >*/
{
	int j, ix, iz, nx, nz, nzx;
	int nt, nr, ns;
	nx=acpar->nx;
	nz=acpar->nz;
	nzx=nx*nz;
	nt=acpar->nt;
	ns=acpar->ns;
	nr=acpar->nr;

	float fcost=0.0;

	float ***dd;
	dd = sf_floatalloc3(nt,nr,ns);
	sf_floatread(&dd[0][0][0],ns*nt*nr,Fdat); //Read the observe data

	float **grad;
	grad = sf_floatalloc2(nz,nx);

	for(int ii=0; ii<3;ii++){
	MPI_Barrier(MPI_COMM_WORLD);
	gradient(grad, &fcost, dd, mpipar, soupar, acpar, array, verb);
	if (mpipar->cpuid==0)
		sf_warning("fcost=%10.9f\n",fcost);
	if(mpipar->cpuid==0)
		sf_floatwrite(&grad[0][0],nzx,Fgrad);
	//sf_fileclose(Fgrad);

	MPI_Barrier(MPI_COMM_WORLD);
	optpar->fk = fcost;
	
	float alpha;
	int flag;
//	line_search(grad, &alpha, dd, mpipar, soupar, acpar, array, fwipar, optpar, &flag, verb);
//	if(mpipar->cpuid==0)
//		sf_warning("alpha=%f",alpha);


/*	if(ii==0)
		alpha=0.0;
	if(ii==1)
		alpha=0.0;
	if(ii==2)
		alpha=0.0;
	if(ii==3)
		alpha=0.0;
	if(ii==4)
		alpha=0.0;

	j = 0;
	for(ix=0;ix<nx;ix++)
		for(iz=0;iz<nz;iz++)
		{
			array->tau[j] = array->tau[j] + 2.*alpha*grad[ix][iz];  //update model
				if(array->tau[j] > 0.0689) //Q=30
					array->tau[j] = 0.0689;
				if(array->tau[j] < 0.002) //Q=1000
					array->tau[j] = 0.002;
			array->taus[j] = 1./(2.*SF_PI*acpar->f0*sqrtf(1.+array->tau[j]));
			array->qq[j] = 2.0*sqrt(array->tau[j]+1.)/array->tau[j];
			j++;
		}*/
	}

	if(mpipar->cpuid==0)
		sf_floatwrite(&array->qq[0],nzx,Finv);
	sf_fileclose(Finv);

	free(**dd); free(*dd); free(dd);
	free(*grad); free(grad);
}

void line_search(float **direction, float *alpha, float ***dd, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, sf_fwi fwipar, sf_optim optpar, int *flag, bool verb)
{
	int i, j, ix, iz, nx, nz, nt, nzx;
	float alpha1=0., alpha2=0.;
	nx=acpar->nx;
	nz=acpar->nz;
	nt=acpar->nt;
    nzx=acpar->nz*acpar->nx;
	float fcost=0.0;

    sf_vec array1;
	array1=(sf_vec)sf_alloc(1, sizeof(*array1));
    array1->vv=sf_floatalloc(nzx);
    array1->dvv=sf_floatalloc(nzx);
    array1->qq=sf_floatalloc(nzx);
    array1->tau=sf_floatalloc(nzx);
    array1->taus=sf_floatalloc(nzx);
    array1->ww=sf_floatalloc(nt);

	for(i=0;i<nzx;i++)
	{
		array1->vv[i] = array->vv[i];
		array1->dvv[i] = array->dvv[i];
		array1->qq[i] = array->qq[i];
	}

	for(i=0;i<nt;i++)
		array1->ww[i] = array->ww[i];

	MPI_Barrier(MPI_COMM_WORLD);
	*alpha = 0.1;//optpar->alpha;
	for(i=0; i<optpar->nls; i++)
	{
		j = 0;
		for(ix=0;ix<nx;ix++)
			for(iz=0;iz<nz;iz++)
			{
				array1->tau[j] = array->tau[j] + *alpha*direction[ix][iz];
				if(array1->tau[j] > 0.0689) //Q=30
					array1->tau[j] = 0.0689;
				if(array1->tau[j] < 0.002) //Q=1000
					array1->tau[j] = 0.002;
				array1->taus[j] = 1./(2.*SF_PI*acpar->f0*sqrtf(1.+array1->tau[j]));
				j++;
			}

		fcost = 0.0;
		MPI_Barrier(MPI_COMM_WORLD);
		forward_modeling_ls(dd, &fcost, mpipar, soupar, acpar, array1, verb);
		MPI_Barrier(MPI_COMM_WORLD);
		if (mpipar->cpuid==0)
			sf_warning("fcost1111=%f\talpha=%f\n",fcost,*alpha);

		if(fcost > optpar->fk)
		{
			alpha2=*alpha;
			*alpha=0.5*(alpha1+alpha2);
		}
		else
		{
			alpha1=*alpha;
			if(alpha2 == 0.)
				*alpha *= optpar->factor;
			else
				*alpha = 0.5*(alpha1+alpha2);
		}
	}

	if(i==optpar->nls)
	{
		if(fcost <= optpar->fk)
			*flag=1;
		else
			*flag=2;
	}

	free(array1->vv);
	free(array1->dvv);
	free(array1->qq);
	free(array1->tau);
	free(array1->taus);
	free(array1->ww);

}


