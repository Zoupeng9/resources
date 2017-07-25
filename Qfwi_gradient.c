/* Visco-acoustic gradient */
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
#include "Qfwi_commons.h"
#include "Qfwi_modeling.h"
#include "smooth2d.h"
#include "triutil.h"
#include <time.h>
#include <stdlib.h>


MPI_Comm comm=MPI_COMM_WORLD;

void gradient_init(sf_file Fdat, sf_mpi *mpipar, sf_acqui acpar)
/*< initialize >*/
{
	return;
}

void gradient(float **grad, float *fcost, float ***dd, sf_mpi *mpipar, sf_sou soupar, sf_acqui acpar, sf_vec array, bool verb)
/*< viscoacoustic \tau gradient >*/
{
	int ix, iz;
	int nx,nz,nt,ns,nr;
	float ***rcd, **grads, **gradr, **sillums, **sillumr;

	nz=acpar->nz;
	nx=acpar->nx;
	nt=acpar->nt;
	ns=acpar->ns;
	nr=acpar->nr;

    rcd = sf_floatalloc3(nt,nr,ns);
    grads=sf_floatalloc2(nz,nx);
    gradr=sf_floatalloc2(nz,nx);
    memset(grads[0],0,nx*nz*sizeof(float));
    memset(gradr[0],0,nx*nz*sizeof(float));

    sillums=sf_floatalloc2(nz,nx);
    sillumr=sf_floatalloc2(nz,nx);
    memset(sillums[0],0,nx*nz*sizeof(float));
    memset(sillumr[0],0,nx*nz*sizeof(float));

   forward_cpml(rcd,dd,fcost,sillums,mpipar,soupar,acpar,array,verb);

    if(mpipar->cpuid==0)
    {
    FILE *fp=fopen("rcd","wb");
    fwrite(&rcd[0][0][0],sizeof(float),ns*nt*nr,fp);
    fclose(fp);}

    MPI_Barrier(comm);
    adjoint_cpml(grads,gradr,rcd, sillumr, mpipar,soupar,acpar,array,verb);

	float maxgrad = 0.0;
    for(ix=0;ix<nx;ix++)
        for(iz=0;iz<nz;iz++)
		{
          grads[ix][iz] = grads[ix][iz]/(sillums[ix][iz]+sillumr[ix][iz]);
          gradr[ix][iz] = gradr[ix][iz]/(sillums[ix][iz]+sillumr[ix][iz]);
			grad[ix][iz] = grads[ix][iz]+gradr[ix][iz];
			if(iz>95 || iz<8)
				grad[ix][iz] = 0.;
			if(fabs(grad[ix][iz]) > maxgrad)
				maxgrad = fabs(grad[ix][iz]);
        }

	float maxtau = 0.0;
	for(ix=0;ix<nx*nz;ix++)
		if(fabs(array->tau[ix])>maxtau)
			maxtau = fabs(array->tau[ix]);

	if(mpipar->cpuid==0)
	{
		sf_warning("maxtau=%f",maxtau);
		sf_warning("maxgrad=%f",maxgrad);
	}

	for(ix=0;ix<nx;ix++)
		for(iz=0;iz<nz;iz++)
		grad[ix][iz] = grad[ix][iz]/maxgrad*maxtau;

	smooth2d(grad,nz,nx,10.,10.,3.);

    MPI_Barrier(comm);

	free(**rcd); free(*rcd); free(rcd);
	free(*gradr); free(gradr);
	free(*grads); free(grads);
	free(*sillums); free(sillums);
	free(*sillumr); free(sillumr);
}
