/* 2-D two-components wavefield modeling visco-acoustic media.
   Copyright (C) 2015 Tongji University, Shanghai, China 
   Authors: Peng Zou
     
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

/* prepared head files by myself */
#include "_cjb.h"
#include <alloc.h>

/* calculate phase, group velocity and angle  */
#include "puthead.h"

int main(int argc, char* argv[])
{

        int    nx, nz;
        float  dx, dz; 
        float  Qp_1, density_1, Vp_1;
        float  Qp_2, density_2, Vp_2;
        float  **w;

        sf_init(argc,argv);

        if (!sf_getint("nx",&nx)) nx=201;
        if (!sf_getint("nz",&nz)) nz=201;
        if (!sf_getfloat("dx",&dx)) dx=0.01;
        if (!sf_getfloat("dz",&dz)) dz=0.01;
        if (!sf_getfloat("Qp_1",&Qp_1)) Qp_1=20.0;
        if (!sf_getfloat("Qp_2",&Qp_2)) Qp_1=20.0;
        if (!sf_getfloat("density_1",&density_1)) density_1=2500.0;
        if (!sf_getfloat("density_2",&density_2)) density_2=2500.0;
        if (!sf_getfloat("Vp_1",&Vp_1)) Vp_1=3000.0;
        if (!sf_getfloat("Vp_2",&Vp_2)) Vp_2=3000.0;

        sf_warning("nx= %d nz= %d",nx,nz);
        sf_warning("dx= %f dz= %f",dx,dz);
        sf_warning("Qp_1= %f density_1= %f Vp_1= %f",Qp_1,density_1,Vp_1);
        sf_warning("Qp_2= %f density_2= %f Vp_2= %f",Qp_2,density_2,Vp_2);

        w = sf_floatalloc2(nz,nx);

	zero2float(w,nz,nx);

	int   i,j;

        /* setup I/O files */
        sf_file Fo1, Fo2, Fo3;
        Fo1 = sf_output("out"); // Qp
        Fo2 = sf_output("Vp"); // Vp
        Fo3 = sf_output("density"); // density

        puthead2(Fo1, nz, nx, dz, 0.0, dx, 0.0);
        puthead2(Fo2, nz, nx, dz, 0.0, dx, 0.0);
        puthead2(Fo3, nz, nx, dz, 0.0, dx, 0.0);

        int nnz=nz*5./10;

        //Qp 
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[i][j] = Qp_1;
           for(j=nnz;j<nz;j++)
              w[i][j] = Qp_2;
           
           sf_floatwrite(w[i], nz, Fo1);
        }
        // Vp
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[i][j] = Vp_1;
           for(j=nnz;j<nz;j++)
              w[i][j] = Vp_2;
           
           sf_floatwrite(w[i], nz, Fo2);
        }
        // density
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[i][j] = density_1;
           for(j=nnz;j<nz;j++)
              w[i][j] = density_2;
           
           sf_floatwrite(w[i], nz, Fo3);
        }
        free(*w);
}
