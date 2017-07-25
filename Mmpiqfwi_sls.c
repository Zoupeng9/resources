/* Visco-acoustic Forward Modeling and FWI based on SLS model */
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
#include "Qfwi_modeling.h"
#include "Qfwi_fwi.h"

int main(int argc, char* argv[])
{
	int function, media;
	bool verb;

	sf_mpi mpipar;
	sf_sou soupar;
	sf_acqui acpar;
	sf_vec array;
	sf_fwi fwipar;
	sf_optim optpar=NULL;

	MPI_Comm comm=MPI_COMM_WORLD;

	sf_file Fv, Fdv, Fq, Fw, Fdat, Finv=NULL, Fgrad;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(comm, &mpipar.cpuid);
	MPI_Comm_size(comm, &mpipar.numprocs);

	sf_init(argc, argv);


	Fv=sf_input("Fvel"); /* background velocity model */
	Fdv = sf_input("Fdvel"); /* pertubation velocity model */
	Fq=sf_input("Fq"); /* quality factor */
	Fw=sf_input("Fwavelet"); /* wavelet */

	soupar=(sf_sou)sf_alloc(1, sizeof(*soupar));
	acpar=(sf_acqui)sf_alloc(1, sizeof(*acpar));
	array=(sf_vec)sf_alloc(1, sizeof(*array));

	/* parameters I/O */
	if(!sf_getint("media", &media)) media=1;
	/* if 1, acoustic media; if 2, visco-acoustic media */
	if(!sf_getint("function", &function)) function=2;
	/* if 1, forward modeling; if 2, FWI; */

	if(!sf_histint(Fv, "n1", &acpar->nz)) sf_error("No n1= in Fv");
	if(!sf_histint(Fv, "n2", &acpar->nx)) sf_error("No n2= in Fv");
	if(!sf_histfloat(Fv, "d1", &acpar->dz)) sf_error("No d1= in Fv");
	if(!sf_histfloat(Fv, "d2", &acpar->dx)) sf_error("No d2= in Fv");
	if(!sf_histfloat(Fv, "o1", &acpar->z0)) sf_error("No o1= in Fv");
	if(!sf_histfloat(Fv, "o2", &acpar->x0)) sf_error("No o2= in Fv");
	if(!sf_histint(Fw, "n1", &acpar->nt)) sf_error("No n1= in Fw");
	if(!sf_histfloat(Fw, "d1", &acpar->dt)) sf_error("No d1= in Fw");
	if(!sf_histfloat(Fw, "o1", &acpar->t0)) sf_error("No o1= in Fw");

	if(!sf_getbool("verb", &verb)) verb=false; /* verbosity flag */
	if(!sf_getint("nb", &acpar->nb)) acpar->nb=30; /* boundary width */

	if(!sf_getint("acqui_type", &acpar->acqui_type)) acpar->acqui_type=1;
	/* if 1, fixed acquisition; if 2, symmetric acquisition */
	if(!sf_getint("ns", &acpar->ns)) sf_error("shot number required"); /* shot number */
	if(!sf_getfloat("ds", &acpar->ds)) sf_error("shot interval required"); /* shot interval */
	if(!sf_getfloat("s0", &acpar->s0)) sf_error("shot origin required"); /* shot origin */
	if(!sf_getint("sz", &acpar->sz)) acpar->sz=5; /* source depth */
	if(!sf_getint("nr", &acpar->nr)) acpar->nr=acpar->nx; /* number of receiver */
	if(!sf_getfloat("dr", &acpar->dr)) acpar->dr=acpar->dx; /* receiver interval */
	if(!sf_getfloat("r0", &acpar->r0)) acpar->r0=acpar->x0; /* receiver origin */
	if(!sf_getint("rz", &acpar->rz)) acpar->rz=1; /* receiver depth */

	if(!sf_getfloat("f0", &acpar->f0)) sf_error("reference frequency required"); /* reference frequency */
	if(!sf_getint("interval", &acpar->interval)) acpar->interval=1; /* wavefield storing interval */

	if(!sf_getfloat("fhi", &soupar->fhi)) soupar->fhi=0.5/acpar->dt; 
	if(!sf_getfloat("flo", &soupar->flo)) soupar->flo=0.; 
	soupar->rectx=1; 
	soupar->rectz=1; 

	/* get prepared */
	preparation(Fv, Fdv, Fq, Fw, acpar, soupar, array);

        switch (function) {

            case 1: /* Modeling */

		Fdat=sf_output("output"); /* shot data */

		/* dimension set up */
		sf_putint(Fdat, "n1", acpar->nt);
		sf_putfloat(Fdat, "d1", acpar->dt);
		sf_putfloat(Fdat, "o1", acpar->t0);
		sf_putstring(Fdat, "label1", "Time");
		sf_putstring(Fdat, "unit1", "s");
		sf_putint(Fdat, "n2", acpar->nr);
		sf_putfloat(Fdat, "d2", acpar->dr);
		sf_putfloat(Fdat, "o2", acpar->r0);
		sf_putstring(Fdat, "label2", "Receiver");
		sf_putstring(Fdat, "unit2", "km");
		sf_putint(Fdat, "n3", acpar->ns);
		sf_putfloat(Fdat, "d3", acpar->ds);
		sf_putfloat(Fdat, "o3", acpar->s0);
		sf_putstring(Fdat, "label3", "Shot");
		sf_putstring(Fdat, "unit3", "km");

		forward_modeling_cpml(Fdat, &mpipar, soupar, acpar, array, verb);

		sf_fileclose(Fdat);

                break;

            case 2: /* FWI */

		fwipar=(sf_fwi)sf_alloc(1, sizeof(*fwipar));

		Fdat=sf_input("Fdat"); /* input data */
		Finv=sf_output("output"); /* FWI result */
		Fgrad=sf_output("Fgrad"); /* FWI gradient at first iteration */

		/* dimension set up */
			sf_putint(Finv, "n1", acpar->nz);
			sf_putfloat(Finv, "d1", acpar->dz);
			sf_putfloat(Finv, "o1", acpar->z0);
			sf_putstring(Finv, "label1", "Depth");
			sf_putstring(Finv, "unit1", "km");
			sf_putint(Finv, "n2", acpar->nx);
			sf_putfloat(Finv, "d2", acpar->dx);
			sf_putfloat(Finv, "o2", acpar->x0);
			sf_putstring(Finv, "label2", "Distance");
			sf_putstring(Finv, "unit2", "km");
			if(fwipar->grad_type==3) sf_putint(Finv, "n3", 2);
		sf_putint(Fgrad, "n1", acpar->nz);
		sf_putfloat(Fgrad, "d1", acpar->dz);
		sf_putfloat(Fgrad, "o1", acpar->z0);
		sf_putstring(Fgrad, "label1", "Depth");
		sf_putstring(Fgrad, "unit1", "km");
		sf_putint(Fgrad, "n2", acpar->nx);
		sf_putfloat(Fgrad, "d2", acpar->dx);
		sf_putfloat(Fgrad, "o2", acpar->x0);
		sf_putstring(Fgrad, "label2", "Distance");
		sf_putstring(Fgrad, "unit2", "km");
		if(fwipar->grad_type==3) sf_putint(Fgrad, "n3", 2);

			optpar=(sf_optim)sf_alloc(1, sizeof(*optpar));
			if(!sf_getint("niter", &optpar->niter)) sf_error("iteration number required"); /* iteration number */
			if(!sf_getfloat("conv_error", &optpar->conv_error)) sf_error("convergence error required"); /* final convergence error */
			optpar->nls=1; /* line search number */
			optpar->factor=2;

		qfwi(Fdat, Finv, Fgrad, &mpipar, soupar, acpar, array, fwipar, optpar, verb, media);

		sf_fileclose(Finv);
		sf_fileclose(Fgrad);

                break;

            default:
                sf_warning("Please specify a valid function");

	} /* switch */

	MPI_Finalize();
	exit(0);
}
