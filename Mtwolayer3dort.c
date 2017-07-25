/* 2-D two-components wavefield modeling using pseudo-pure mode P-wave equation in VTI media.
   Copyright (C) 2012 Tongji University, Shanghai, China 
   Authors: Jiubing Cheng and Wei Kang
     
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

/* calculate phase, group velocity and angle  */
#include "zero.h"
#include "puthead.h"

int main(int argc, char* argv[])
{

        int    nx, ny, nz;
        float  dx, dy, dz; 
        float  vp0_1, vs0_1, eps1_1, eps2_1, del1_1, del2_1, del3_1, gam1_1, gam2_1, the_1, phi_1;
        float  vp0_2, vs0_2, eps1_2, eps2_2, del1_2, del2_2, del3_2, gam1_2, gam2_2, the_2, phi_2;
        float  ***w;

        sf_init(argc,argv);

        if (!sf_getint("ny",&ny)) ny=201;
        if (!sf_getint("nx",&nx)) nx=201;
        if (!sf_getint("nz",&nz)) nz=201;
        if (!sf_getfloat("dy",&dy)) dy=0.008;
        if (!sf_getfloat("dx",&dx)) dx=0.008;
        if (!sf_getfloat("dz",&dz)) dz=0.008;
        if (!sf_getfloat("vp0_1",&vp0_1)) vp0_1=2436.7;
        if (!sf_getfloat("vs0_1",&vs0_1)) vs0_1=1264.9;
        if (!sf_getfloat("eps1_1",&eps1_1)) eps1_1=0.3286;
        if (!sf_getfloat("eps2_1",&eps2_1)) eps2_1=0.2579;
        if (!sf_getfloat("del1_1",&del1_1)) del1_1=0.0825;
        if (!sf_getfloat("del2_1",&del2_1)) del2_1=-0.0775;
        if (!sf_getfloat("del3_1",&del3_1)) del3_1=-0.1064;
        if (!sf_getfloat("gam1_1",&gam1_1)) gam1_1=0.1819;
        if (!sf_getfloat("gam2_1",&gam2_1)) gam2_1=0.0455;
		if (!sf_getfloat("the_1",&the_1)) the_1=0.0;
		if (!sf_getfloat("phi_1",&phi_1)) phi_1=0.0;
        if (!sf_getfloat("vp0_2",&vp0_2)) vp0_2=2436.7;
        if (!sf_getfloat("vs0_2",&vs0_2)) vs0_2=1264.9;
        if (!sf_getfloat("eps1_2",&eps1_2)) eps1_2=0.3286;
        if (!sf_getfloat("eps2_2",&eps2_2)) eps2_2=0.2579;
        if (!sf_getfloat("del1_2",&del1_2)) del1_2=0.0825;
        if (!sf_getfloat("del2_2",&del2_2)) del2_2=-0.0775;
        if (!sf_getfloat("del3_2",&del3_2)) del3_2=-0.1064;
        if (!sf_getfloat("gam1_2",&gam1_2)) gam1_2=0.1819;
        if (!sf_getfloat("gam2_2",&gam2_2)) gam2_2=0.0455;
		if (!sf_getfloat("the_2",&the_2)) the_2=0.0;
		if (!sf_getfloat("phi_2",&phi_2)) phi_2=0.0;
		// the default parameters from Shang yingxia's master thesis 

        sf_warning("nx= %d ny=%d nz= %d",nx,ny,nz);
        sf_warning("dx= %f dy=%f dz= %f",dx,dy,dz);
        sf_warning("vp0_1= %f vs0_1= %f eps1_1= %f del1_1=%f gam1_1=%f eps2_1=%f del2_1=%f gam2_1=%f del3_1=%f",vp0_1,vs0_1,eps1_1,del1_1,gam1_1,eps2_1,del2_1,gam2_1,del3_1);
        sf_warning("vp0_2= %f vs0_2= %f eps1_2= %f del1_2=%f gam1_2=%f eps2_2=%f del2_2=%f gam2_2=%f del3_1=%f",vp0_2,vs0_2,eps1_2,del1_2,gam1_2,eps2_2,del2_2,gam2_2,del3_2);

        w = sf_floatalloc3(nz,nx, ny);

	zero3float(w,nz,nx,ny);

	int   i,j,k;

        /* setup I/O files */
        sf_file Fo1, Fo2, Fo3, Fo4, Fo5, Fo6, Fo7, Fo8, Fo9, Fo10, Fo11;
        Fo1 = sf_output("out"); // Vp0
        Fo2 = sf_output("vs0"); // Vs0
        Fo3 = sf_output("epsi1"); // Epsilon1
        Fo4 = sf_output("epsi2"); // Epsilon2
        Fo5 = sf_output("del1"); // Delta1
        Fo6 = sf_output("del2"); // Delta2
        Fo7 = sf_output("del3"); // Delta3
        Fo8 = sf_output("gam1"); // gama1
        Fo9 = sf_output("gam2"); // gama2
		Fo10 = sf_output("the"); //the
		Fo11 = sf_output("phi"); //phi

        puthead3x(Fo1, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo2, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo3, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo4, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo5, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo6, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo7, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo8, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo9, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo10, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);
        puthead3x(Fo11, nz, nx, ny, dz*0.001, dx*0.001, dy*0.001, 0.0, 0.0, 0.0);

        int nnz=nz*7/12;

        // Vp0
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = vp0_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = vp0_2;
           
           sf_floatwrite(w[k][i], nz, Fo1);
        }
        // Vs0
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = vs0_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = vs0_2;
           
           sf_floatwrite(w[k][i], nz, Fo2);
        }
        // Epsi1
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = eps1_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = eps1_2;
           
           sf_floatwrite(w[k][i], nz, Fo3);
        }
        // Epsi2
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = eps2_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = eps2_2;
           
           sf_floatwrite(w[k][i], nz, Fo4);
        }
        // Delta1 
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = del1_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = del1_2;
           
           sf_floatwrite(w[k][i], nz, Fo5);
        }
        // Delta2 
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = del2_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = del2_2;
           
           sf_floatwrite(w[k][i], nz, Fo6);
        }
        // Delta3 
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = del3_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = del3_2;
           
           sf_floatwrite(w[k][i], nz, Fo7);
        }
        //  Gama1
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = gam1_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = gam1_2;
           
           sf_floatwrite(w[k][i], nz, Fo8);
        }
        //  Gama2
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = gam2_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = gam2_2;
           
           sf_floatwrite(w[k][i], nz, Fo9);
        }
        // the 
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = the_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = the_2;
           
           sf_floatwrite(w[k][i], nz, Fo10);
		}
        // phi 
        for(k=0;k<ny;k++)
        for(i=0;i<nx;i++)
        {
           for(j=0;j<nnz;j++)
              w[k][i][j] = phi_1;
           for(j=nnz;j<nz;j++)
              w[k][i][j] = phi_2;
           
           sf_floatwrite(w[k][i], nz, Fo11);
		}

        free(**w);
}
