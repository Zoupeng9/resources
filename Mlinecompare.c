/*
 * =====================================================================================
 *
 *       Filename:  Mlinecompare.c
 *
 *    Description:  window a line and make a comparison
 *
 *        Version:  1.0
 *        Created:  31. aug. 2016 kl. 12.55 +0800
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Chenlong Wang (chenlonw), clwang88@gmail.com
 *   Organization:  Tongji University
 *
 * =====================================================================================
 */

#include <rsf.h>
#include "utils.h"

int main ( int argc, char *argv[] )
{
	bool verb;                                    /* verbose */
	int which;
	int f;
	bool display;
	bool rm;
	char *Ffile1;
	char *Ffile2;
	char *Ofile;

	/* initialize rsf */
	sf_init(argc,argv);

	/* Read arguments from command lines */
	if ( !sf_getbool( "verb" , &verb ) ) verb = 1;
	if ( !sf_getint( "which" , &which ) ) which = 1;
	if ( !sf_getint( "f" , &f ) ) f = 1;
	if ( !sf_getbool( "display" , &display ) ) display = false;
	if ( !sf_getbool( "rm" , &rm ) ) rm = true;
	Ffile1 = sf_getstring("f1");
	Ffile2 = sf_getstring("f2");
	Ofile = sf_getstring("fout");
	if (Ofile==NULL)
		Ofile="chlwcompare";
	
	char *command1 = sf_charalloc(1024);

	char *cf = sf_charalloc(16);
	snprintf(cf,16,"%d",f);

	char *path = get_env("RSFROOT");

	char *cwhich = sf_charalloc(16);
	snprintf(cwhich,16,"%d",which);

	if (path==NULL)
	{
		path=sf_getstring("path");
	}
	strcat(path, "/bin/");

	strcpy(command1, path);
	strcat(command1, "sfwindow< ");
	strcat(command1, Ffile1);
	strcat(command1, " n");
	strcat(command1, cwhich);
	strcat(command1, "=");
	strcat(command1, "1");
	strcat(command1, " f");
	strcat(command1, cwhich);
	strcat(command1, "=");
	strcat(command1, cf);
	strcat(command1, " > chlwtmp1.rsf; ");
	strcat(command1, path);
	strcat(command1, "sfwindow< ");
	strcat(command1, Ffile2);
	strcat(command1, " n");
	strcat(command1, cwhich);
	strcat(command1, "=");
	strcat(command1, "1");
	strcat(command1, " f");
	strcat(command1, cwhich);
	strcat(command1, "=");
	strcat(command1, cf);
	strcat(command1, " > chlwtmp2.rsf; ");
	strcat(command1, path);
	strcat(command1, "sfcat <chlwtmp1.rsf chlwtmp2.rsf axis=2");
	strcat(command1, ">chlwall.rsf; ");
	strcat(command1, path);
	strcat(command1, "sfgraph title=\" \" <chlwall.rsf > ");
	strcat(command1, Ofile);
	strcat(command1, ".vpl; ");
	if (rm) {
		strcat(command1, path);
		strcat(command1, "sfrm chlwtmp1.rsf chlwtmp2.rsf chlwall.rsf; ");
	}
	if (display) {
		strcat(command1, path);
		strcat(command1, "sfpen ");
		strcat(command1, Ofile);
		strcat(command1, ".vpl");
	}

	if (verb) {
		fprintf(stderr, "%s\n", command1);
	}

	system(command1);

	sf_close();
	exit(0);
}
