/*
 * =====================================================================================
 *
 *       Filename:  Mmpidemo.c
 *
 *    Description:  This is a simple demo, which provides a general way to complete 
 *                  your code in parallel by MPI in Madagascar. Instead of parallelizing 
 *                  your complex code for different methods, it generates commands and then 
 *                  executes them in parallel. In other words, it modularizes your code 
 *                  into different commands;
 *                  This demo uses the simple command, sfspike, in Madagascar software to 
 *                  generate different spikes with different depths and then combines (cat) 
 *                  them together. You can replace this command in your cases.
 *
 *        Version:  1.0
 *        Created:  04. mai 2016 kl. 10.18 +0800
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Chenlong Wang (chenlonw), clwang88@gmail.com. Based on Wiktor's work
 *   Organization:  Tongji University
 *      Copyright:  Copyright (c) 2016, Tongji University
 * =====================================================================================
 */
 
/* Including libraries */ 
#include <rsf.h>
#include <stdio.h>
#include <stddef.h>
#include <mpi.h>
#include <time.h>
#include <sys/stat.h>

#define STRING_LENGTH 4096
#define JOB_NOT_STARTED 0
#define JOB_RUNNING 1
#define JOB_FINISHED 2
#define JOB_FAILED 3
//#define DEBUG

#define TAGWORK 1
#define TAGDIE 2
#define TAGNOWORK 3

#define LOGFILE "status.txt"
#define OUTPUTLOG "outputlist.txt"

/* Structs */
struct work
{
	char call1[STRING_LENGTH];
	int status;
	int id;
	int shotid;
};
typedef struct work Work;

struct result
{
	int id;
	int shotid;
	int status;
};
typedef struct result Result;

struct filename
{
    char *out;                                  /* Filenames to the output files. */
};
typedef struct filename *Files;

struct paths
{
    char *program;                              /* Path to program folder, usually RSFROOT */
    char *local;                                /* Local path on each cluster node */
    char *work;                                 /* Work path on global file system on HPC */
    char *temp;                                 /* Temporary path on global file system on HPC */
    char *data;                                 /* Path where the splitted data files are stored */
};
typedef struct paths *Paths;

struct param
{
    bool verb;                                  /* Verbose */
    bool save;                                  /* If save temp files */
	int nx;                                     /* Spike dimension */
    int nz;                                     /* Spike dimension */
	int depth0;                                 /* Initial depth */
	int Ddepth;                                 /* Depth increment */
	float dx;                                   /* Spatial increment in x-direction */
	float dz;                                   /* Spatial increment in z-direction */
    char c_nx[16],c_nz[16];                     /* Variable to hold string conversions from variables */
    char c_dx[16],c_dz[16];                     /* Variable to hold string conversions from variables */
};
typedef struct param Param;

struct programs
{
    char *command;                              /* Program name for parallelizing, here is sfspike */
};
typedef struct programs *Programs;

struct log
{
    int n;                                      /* Number of jobs */
    time_t *start;                              /* Start time stamp for each job */
    time_t *end;                                /* End time stamp for each job */
    int ncpu;                                   /* Number of CPUs */
    int jobs_left;                              /* Number of jobs left */
};
typedef struct log Log;

/* Function declarations */
Files files_setup();
Paths paths_setup();
Programs programs_setup();

void check_input(Param param, Paths path, Files file, Programs program);
void param_var2str(Param *param);
bool dir_exists(char *dirname);
char *get_env(const char *var);
void log_setup(Log *log, const int nworks);
void create_work_array(Work *works, const int nworks, Param param, Paths path, Files filename, Programs program);
void create_list_of_files(const char *Precfile, const int nworks, Paths path, Files filename);
Work* get_next_work_item(Work *works, const int nworks);
void send_work(Work *work, Log *log, const int rank, MPI_Datatype mpiWork);
void log_filewrite(FILE *file, Log *log, Work *works);
unsigned long count_lines_of_file(const char *file_patch);
void operation(const char *list, const char *outname, const Param param, const Paths path);

/* Main loop */
int main(int argc, char* argv[]) {

    bool verb;                                 /* Verbose */
    bool save;                                 
	int nx, nz;
	float dx, dz;
	int depth0, Ddepth;
	int i, myrank, nrank;
	int nspikes;                                /* How many spikes you want to create */
	int nworks;
	Work *works;
    Paths paths;                                /* Paths struct */

	/* Initializing MPI */
	MPI_Init(&argc,&argv);

	/* Initializing RSF */
	sf_init(argc,argv);

	/* Initializing structs */
	paths = paths_setup();

	/* Create MPI structs */
	int count=4;
	int lengths[4] = {STRING_LENGTH,1,1,1};
	MPI_Datatype types[4] = {MPI_CHAR,MPI_INT,MPI_INT,MPI_INT};
	MPI_Datatype mpiWork;
	MPI_Aint offsets[4];
	offsets[0] = offsetof(Work,call1);
	offsets[1] = offsetof(Work,status);
	offsets[2] = offsetof(Work,id);
	offsets[3] = offsetof(Work,shotid);
	MPI_Type_create_struct(count,lengths,offsets,types,&mpiWork);
	MPI_Type_commit(&mpiWork);

	int count2=4;
	int lengths2[4] = {1,1,1,1};
	MPI_Aint offsets2[4];
	offsets2[0] = offsetof(Result,id);
	offsets2[1] = offsetof(Result,status);
	offsets2[2] = 0;
	offsets2[3] = offsetof(Result,shotid);
	MPI_Datatype types2[4] = {MPI_INT,MPI_INT,MPI_FLOAT,MPI_INT};
	MPI_Datatype mpiResult;
	MPI_Type_create_struct(count2,lengths2,offsets2,types2,&mpiResult);
	MPI_Type_commit(&mpiResult);

	/* Get command line parameters */
	// Paths
	paths->work = sf_getstring("workpath");		/* Working path for saving data */
    paths->local = sf_getstring("localpath");   /* Path to local storage area on cluster nodes */
    paths->temp = sf_getstring("temppath");     /* Path to temporary directory */

	// Find out rank (ID)
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

	if(myrank == 0) {
		/* MASTER */

		/* Variables */
        Files filenames;                        /* Filenames */
        Param params;                           /* Parameters for performing modeling */
        Programs programs;                      /* Programs struct */
		Work *work;
		MPI_Status status;
		Result result;
		Log log;
		FILE *logfile;

		/* MPI setup */
		MPI_Comm_size(MPI_COMM_WORLD,&nrank);

		/* Initializing structs */
		filenames = files_setup();
		programs = programs_setup();

		/* Command line parameter */
		// Misc
        if(!sf_getbool("verb",&verb)) verb=0;   /* Verbose flag */
        if(!sf_getbool("save",&save)) save=0;   /* Verbose flag */
		if(!sf_getint("nx",&nx)) nx = 51;                           
		if(!sf_getint("nz",&nz)) nz = 51;                           
		if(!sf_getfloat("dx",&dx)) dx = 10;                           
		if(!sf_getfloat("dz",&dz)) dz = 10;                           
		if(!sf_getint("depth0",&depth0)) depth0 = 10;                           
		if(!sf_getint("Ddepth",&Ddepth)) Ddepth = 10;                           
		if(!sf_getint("nspikes",&nspikes)) nspikes = 2;                           
        filenames->out = sf_getstring("out");   /* Filename to the output spikes */

		// Program names
		programs->command = sf_getstring("commd");				/* Program name for the modeling for one shot */

		/* Saving to parameter struct */
		params.nx = nx;
		params.nz = nz;
		params.dx = dx;
		params.dz = dz;
		params.depth0 = depth0;
		params.Ddepth = Ddepth;
		params.save = save;

		/* Getting program paths */
		paths->program = get_env("RSFROOT");
		if(paths->program == NULL) {
			sf_error("ERROR: Program not able to find the RSFROOT environment.\n");
		}

		/* Checking input and structs */
		check_input(params,paths,filenames,programs);

		// Converting to strings
		param_var2str(&params);

		/* Setting up works */
        nworks = nspikes;                                                /* Setup of number of works to be done */

		/* Work creation */
        works = malloc(sizeof(*works)*nworks);                           /* Defined in the beginning */
		create_work_array(works,nworks,params,paths,filenames,programs); /* Construct the command lines for all different spikes */
		create_list_of_files(OUTPUTLOG, nworks, paths, filenames);

		/* Log system startup */
		log_setup(&log,nworks);                 /* Alloc space include the running time for each work */
		log.ncpu = nrank;

		// Opening logfile
		if((logfile = fopen(LOGFILE,"w")) == NULL) {
			sf_warning("Unable to open logfile");
		}

		/* MPI STARTUP */
		/* Sending to all slaves --first loop*/
		for(i=1; i<nrank; ++i) {
			// Get work
			work = get_next_work_item(works,nworks); /* Only defined for Master rank */

			// If we have more ranks than work, we need to do something
			if(work == NULL) {
#ifdef DEBUG
				fprintf(stderr,"Master: sending slave %d NOWORK tag.\n",i);
#endif

				MPI_Send(0,0,MPI_INT,i,TAGNOWORK,MPI_COMM_WORLD);
			}
			else {
#ifdef DEBUG
				fprintf(stderr,"Master: sending slave %d work id %d.\n",i,work->id);
#endif

				// Send work
				send_work(work,&log,i,mpiWork);
			}		
		}

		// Writing logfile
		log_filewrite(logfile,&log,works);

		/* Looping over remaining jobs */
		work = get_next_work_item(works,nworks);
		while(work != NULL) {
			// Receive results from a slave
			MPI_Recv(&result,1,mpiResult,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

			// Checking received tag
			if(status.MPI_TAG == TAGNOWORK) {
#ifdef DEBUG
				fprintf(stderr,"Master received NOWORK tag from slave %d.\n",status.MPI_SOURCE);
#endif
			}
			else {
#ifdef DEBUG
				fprintf(stderr,"Master received from slave %d: jobid: %d, shotid: %d, status: %d, MPI tag: %d\n",status.MPI_SOURCE,result.id,result.shotid,result.status,status.MPI_TAG);
#endif

				// Updating work array
				if(status.MPI_TAG != JOB_FINISHED) {
					// Flagging job as not started and resetting log system
					log.start[result.id] = 0;
					works[result.id].status = JOB_NOT_STARTED;

					// Printing out error message
					fprintf(stderr,"Job %d was reported not successfull from rank %d. Job is done all over again.\n",result.id,status.MPI_SOURCE);
				}
				else {
					// Job was successfull
					works[result.id].status = status.MPI_TAG;

					// Updating log system
					log.end[result.id] = time(NULL);
					log.jobs_left--;
				}
			}

			// Send the slave new work 
			send_work(work,&log,status.MPI_SOURCE,mpiWork);

			// Getting new job
			work = get_next_work_item(works,nworks);

			// Writing logfile
			log_filewrite(logfile,&log,works);
		}

		if(verb) fprintf(stderr,"Master has sent out all works to the nodes. Starting to wait for results from the computing nodes.\n");

		/* No more work to be done, so receive all remaining job */
		for(i=1; i<nrank; ++i) {			
			// Receive results from a slave
			MPI_Recv(&result,1,mpiResult,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

			// Checking received tag
			if(status.MPI_TAG == TAGNOWORK) {
#ifdef DEBUG
				fprintf(stderr,"Master received NOWORK tag from rank %d.\n",status.MPI_SOURCE);
#endif
			}
			else {
#ifdef DEBUG
				fprintf(stderr,"Master received from slave %d: jobid: %d, shotid: %d, status: %d, MPI tag: %d\n",status.MPI_SOURCE,result.id,result.shotid,result.status,status.MPI_TAG);
#endif

				// Updating work array
				if(status.MPI_TAG != JOB_FINISHED) {
					// Flagging job as not started and resetting log system
					log.start[result.id] = 0;
					works[result.id].status = JOB_NOT_STARTED;

					// Printing out error message
					fprintf(stderr,"Job %d was reported not successfull from rank %d. Job is done all over again.\n",result.id,status.MPI_SOURCE);
				}
				else {
					// Job was successfull
					works[result.id].status = status.MPI_TAG;

					// Updating log system
					log.end[result.id] = time(NULL);
					log.jobs_left--;
				}
			}

			// Writing logfile
			log_filewrite(logfile,&log,works);
		}

		if(verb) fprintf(stderr,"Master has collected all results. Starting to stop each node.\n");

		/* Tell all slaves to die */
		for(i=1; i<nrank; ++i) {
			MPI_Send(0,0,MPI_INT,i,TAGDIE,MPI_COMM_WORLD);
		}

		if(verb) fprintf(stderr,"Catting spikes together.\n");
		operation(OUTPUTLOG, filenames->out, params, paths);

		/* Clearing memory */
		free(works);
		free(log.start);
		free(log.end);

		/* Closing files */
		fclose(logfile);

		if(verb) fprintf(stderr,"Master is done. Program quits...\n");

	}
	else {
		/* SLAVE */

		// Variables
		MPI_Status status;
		Work recv;
		Result result;
		int call1;

		while(1) {
			// Receiving message from master
			MPI_Recv(&recv,1,mpiWork,0,MPI_ANY_TAG,MPI_COMM_WORLD,&status);

			// Check tag of received message
			if(status.MPI_TAG == TAGDIE) {
				break;
			}

			if(status.MPI_TAG == TAGNOWORK) {
				result.id = -1;
				result.status = JOB_FINISHED;
				MPI_Send(&result,1,mpiResult,0,TAGNOWORK,MPI_COMM_WORLD);
			}
			else {
#ifdef DEBUG
				fprintf(stderr,"\nSlave %d received from master:\nSystem call: %s\nStatus:%d\nJobid: %d\nShotid: %d\n\n",myrank,recv.call1,recv.status,recv.id,recv.shotid);
#endif
				// DO the work
				call1 = system(recv.call1);

				// Send result back
				result.id = recv.id;
				result.shotid = recv.shotid;
				if(call1){
					result.status = 1;
				}else{
					result.status = 0;
				}

				if(call1 == 0) MPI_Send(&result,1,mpiResult,0,JOB_FINISHED,MPI_COMM_WORLD);
				else MPI_Send(&result,1,mpiResult,0,JOB_FAILED,MPI_COMM_WORLD);
			}
		}
	}

	/* Closing MPI */
	MPI_Type_free(&mpiWork);
	MPI_Type_free(&mpiResult);
	MPI_Finalize();

	/* Closing and exiting */
	sf_close();
	exit(0);
}

Files files_setup()
/*< Setup for the Files struct. >*/
{
	/* Creating structure */
	Files files;
	files = (Files) malloc(sizeof(*files));
	if(files == NULL) {
		sf_error("Error in allocation memory! Files struct");
	}

	return files;
}

Paths paths_setup()
/*< Setup for the Paths struct. >*/
{
	/* Creating structure */
	Paths paths;
	paths = (Paths) malloc(sizeof(*paths));
	if(paths == NULL) {
		sf_error("Error in memory alloc! Paths struct");
	}

	return paths;
}

Programs programs_setup()
/*< Setup for the Programs struct. >*/
{
	/* Creating structure */
	Programs programs;
	programs = (Programs) malloc(sizeof(*programs));
	if(programs == NULL) {
		sf_error("Error in memory alloc! Program struct");
	}

	return programs;
}

void check_input(Param param, Paths path, Files file, Programs program)
	/*<Checking that all necessary inputs are set properly.>*/
{
	bool error=0;
	char error_buffer[10000];
	strcpy(error_buffer, "The program was aborted due to the following errors:\n");

	//Checking if necessary files are given and exist
	if(file->out == NULL) {
		strcat(error_buffer,"Output filename is missing.\n");
		error=1;
	}

	//Checking if necessary programs are given
	if(program->command == NULL ) {
		strcat(error_buffer,"name of program is missing.\n");
		error=1;
	}

	//Checking if necessary paths are given and exist
	if(path->program == NULL){
		strcat(error_buffer,"Program path is missing.\n");
		error=1;
	}else{
		if(!dir_exists(path->program)) {
			strcat(error_buffer,"Program path does not exist.\n");
			error=1;
		}
	}

	if(path->local == NULL) {
		strcat(error_buffer,"Local path is missing.\n");
		error=1;
	}else{
		if(!dir_exists(path->local)) {
			strcat(error_buffer,"local path does not exist.\n");
			error=1;
		}
	}

	if(path->work == NULL ) {
		strcat(error_buffer,"Work path is missing.\n");
		error=1;
	}else{
		if(!dir_exists(path->work)) {
			strcat(error_buffer,"work path does not exist.\n");
			error=1;
		}
	}

	if(path->temp == NULL ) {
		strcat(error_buffer,"Temp path is missing.\n");
		error=1;
	}else{
		if(!dir_exists(path->temp)) {
			strcat(error_buffer,"temp path does not exist.\n");
			error=1;
		}
	}

	//Checking input parameters
	if(param.nx <= 0 || param.nz <= 0){
		strcat(error_buffer, "size of spatial dimension is invalid.\n");  
		error=1;
	}

	if(param.dx <= 0 || param.dz <= 0){
		strcat(error_buffer, "spatial increment is invalid.\n");  
		error=1;
	}

	if(param.depth0 < 0){
		strcat(error_buffer, "initial depth is invalid.\n");  
		error=1;
	}

	if(param.depth0 < 0){
		strcat(error_buffer, "initial depth is invalid.\n");  
		error=1;
	}

	// Final check 
	if(error){
		fprintf(stderr, "%s", error_buffer);
		sf_error("Program was terminated due to input error(s)");
	}
}

bool dir_exists(char *dirname)
/*<Checking if a directory with dirname exists.>*/
{
	struct stat s;
	if(stat(dirname,&s) == 0) {
		return true;
	}

	return false;
}

char *get_env(const char *var) 
/*<Getting environment variable specified by input without (if any) / at the end.>*/
{
	// Getting environment variable
	char *env = getenv(var);
	int length = strlen(env);

	// Checking if end character is /
	if(env[length - 1] == '/') {
		// Removing / and returning correct path
		char *path = malloc(sizeof(char)*(length-1));
		int i;
		for(i=0; i<length-1;i++) {
			path[i] = env[i];
		}
		path[length-1] = '\0';

		return path;
	}

	return env;
}

void param_var2str(Param *param)
/*<Convert every integer and float variable into string variables.>*/
{
	snprintf(param->c_nx,16,"%d",param->nx);
	snprintf(param->c_nz,16,"%d",param->nz);
	snprintf(param->c_dx,16,"%f",param->dx);
	snprintf(param->c_dz,16,"%f",param->dz);
}

void create_work_array(Work *works, const int nworks, Param param, Paths path, Files filename, Programs program) 
{
	// Variables
	int i, depth;
	char out[256];                              /* filenames */
	char number[16]; 
	char c_depth[16];

	for(i=0; i<nworks; i++) {
		// Creating strings from variables and filenames
		sprintf(number,"%d",i);
		snprintf(out,256,"%s/%d_%s",path->temp,i,filename->out);

		depth = param.depth0 + i * param.Ddepth; /* It is better to obtain this argument as a input file */
		if (depth>param.nz) {
			sf_error("The depth of spikes is over the model size");
		}
		snprintf(c_depth,16,"%d",depth);

		// Creating system call 1
		strcpy(works[i].call1,path->program); strcat(works[i].call1,"/bin/"); 
		strcat(works[i].call1,program->command);
		strcat(works[i].call1, " <"); strcat(works[i].call1,"/dev/null"); 
		strcat(works[i].call1, " n2="); strcat(works[i].call1,param.c_nx); 
		strcat(works[i].call1, " n1="); strcat(works[i].call1,param.c_nz); 
		strcat(works[i].call1, " d2="); strcat(works[i].call1,param.c_dx); 
		strcat(works[i].call1, " d1="); strcat(works[i].call1,param.c_dz); 
		strcat(works[i].call1, " k1="); strcat(works[i].call1,c_depth); 
		strcat(works[i].call1, " l1="); strcat(works[i].call1,param.c_nz); 
		strcat(works[i].call1," out=stdout");
		strcat(works[i].call1," >"); strcat(works[i].call1,out);

#ifdef DEBUG
		/*Print out system call*/
		fprintf (stderr, "System Call1 : %s\n", works[i].call1);
#endif 
		
		// Updating
		works[i].id = i;
		works[i].shotid = i;
		works[i].status = JOB_NOT_STARTED;
	}
}

void create_list_of_files(const char *Precfile, const int nworks, Paths path, Files filename)
{

	// Variables
	int i;
	FILE *cmdlogfile;
	char rec[256];
	if(filename->out != NULL){
		// Opening logfile
		if((cmdlogfile = fopen(Precfile,"w")) == NULL) {
			sf_warning("Unable to open output logfile");
		}
		//Making list of files and outputting
		for(i=0; i<nworks; i++) {
			if(filename->out != NULL) snprintf(rec,256,"%s/%d_%s", path->temp,i,filename->out);
			fprintf(cmdlogfile, "%s\n", rec);
		}
		//Closing logfile
		fclose(cmdlogfile);
	}
}

void log_setup(Log *log, const int nworks) {
	// Variables
	int i;

	// Allocating memory
	log->start = malloc(sizeof(time_t)*nworks);
	log->end = malloc(sizeof(time_t)*nworks);
	log->n = nworks;
	log->jobs_left = nworks;

	for(i=0; i<nworks; i++) {
		log->start[i]=0;
		log->end[i]=0;
	}
}

Work* get_next_work_item(Work *works, const int nworks) {
	// Variables
	int i;

	for(i=0; i<nworks; i++) {
		if(works[i].status == JOB_NOT_STARTED) {
			return &works[i];
		}
	}

	return NULL;
}

void send_work(Work *work, Log *log, const int rank, MPI_Datatype mpiWork)
{
	// Updating status and sending
	work->status = JOB_RUNNING;
	MPI_Send(work,1,mpiWork,rank,TAGWORK,MPI_COMM_WORLD);

	// Updating log system
	log->start[work->id] = time(NULL);	
}

void log_filewrite(FILE *file, Log *log, Work *works){
	// Variables
	char buffer[256],start[256],end[256];
	int i;
	time_t runtime,now;


	// Seeking to beginning of file
	fseek(file,0,SEEK_SET);

	// Title in file
	fputs("***********************************\n",file);
	fputs("*                                 *\n",file);
	fputs("*    CLUSTER QUEUE STATUS FILE    *\n",file);
	fputs("*                                 *\n",file);
	fputs("***********************************\n",file);
	fputs("\n",file);

	// CPU information
	fputs("CPU INFORMATION\n",file);
	// Number of CPUs
	snprintf(buffer,256,"# CPU: %d \n",log->ncpu);
	fputs(buffer,file);
	fputs("\n",file);

	// Job information
	fputs("JOB INFORMATION\n",file);
	snprintf(buffer,256,"Jobs: %d, #Remaining jobs: %d\n\n",log->n,log->jobs_left);
	fputs(buffer,file);
	fputs("#CPUID: Status (0=not started, 1=running, 2=finished). \n",file);
	for(i=0; i<log->n; i++) {
		// Converting time to readable format	
		strcpy(start,ctime(&log->start[i]));
		strcpy(end,ctime(&log->end[i]));
		start[strlen(start)-1] = '\0';
		end[strlen(end)-1] = '\0';

		// Writing beginning of status line
		snprintf(buffer,256,"#%04d: %d  start: ",i,works[i].status);
		fputs(buffer,file);

		// Start time
		if(log->start[i] == 0) fputs("N/A",file);	
		else fputs(start,file);

		// End time
		fputs(", end: ",file);
		if(log->end[i] == 0) 	fputs("N/A",file);
		else fputs(end,file);

		// Runtime
		fputs(", runtime: ",file);
		if(log->start[i] == 0 && log->end[i] == 0) {
			fputs("N/A",file);
		}
		else {
			// Calculating runtime
			if(log->end[i] <= 0) runtime = time(NULL) - log->start[i];
			else runtime = log->end[i] - log->start[i];

			if(runtime < 600) {
				// Less than 10min, printing out sec and min
				snprintf(buffer,256,"%ld",runtime);
				fputs(buffer,file);
				fputs(" sec (",file);
				snprintf(buffer,256,"%ld",runtime/60);
				fputs(buffer,file);
				fputs(" min)",file);
			}
			else if(runtime >= 600 && runtime < (3600*12)) {
				// Less than 12 hours, printing out min and hour
				snprintf(buffer,256,"%5.2f", (float) runtime/60);
				fputs(buffer,file);
				fputs(" min (",file);
				snprintf(buffer,256,"%4.2f",(float) runtime/3600);
				fputs(buffer,file);
				fputs(" h)",file);
			}
			else {
				// Long runtimes, printing out hour and days
				snprintf(buffer,256,"%5.2f",(float) runtime/3600);
				fputs(buffer,file);
				fputs(" h (",file);
				snprintf(buffer,256,"%4.2f",(float) runtime/86400);
				fputs(buffer,file);
				fputs(" d)",file);
			}

			// Printing out running flag if necessary
			if(log->end[i] <= 0) {
				fputs(" (r)",file);
			}
		}

		// New line
		fputs("\n",file);
	}

	// Details summary at end of file
	now = time(NULL);
	strcpy(start,ctime(&now));
	start[strlen(start)-1] = '\0';
	snprintf(buffer,256,"Jobs: %d, #Remaining jobs: %d\n\n",log->n,log->jobs_left);
	fputs(buffer,file);
	fputs("Last updated: ",file); fputs(start,file); fputs("\n\n",file);

	// Flushing
	fflush(file);
}

void operation(const char *list, const char *outname, const Param param, const Paths path) 
{
	// Variables
	char cat[STRING_LENGTH];
	char rm[STRING_LENGTH];
	int cat_return;
	int rm_return;
	int i, nin;
	char line[256];
	FILE *fp;
	char filename[128];

	nin = count_lines_of_file(list);
	fp = fopen(list, "r");

	strcpy(cat,path->program); strcat(cat,"/bin/sfcat axis=2 ");
	strcat(cat,"< ");
	strcpy(rm,path->program); strcat(rm,"/bin/sfrm ");
	for(i=0; i<nin ; i++){
		if(fgets(line, sizeof(line),fp) != NULL){
			sscanf(line, "%s", filename);
		}else{
			fprintf(stderr, "Error reading list of files: %s.\n", list);
			sf_error("Error reading list file.\n");	
		}
		strcat(cat,filename);
		strcat(rm,filename);
		strcat(cat," ");
		strcat(rm," ");
	}
	strcat(cat," out=stdout > "); strcat(cat, path->local);
	strcat(cat,"/"); strcat(cat, outname);

#ifdef DEBUG
	fprintf(stderr,"System Operation:%s\n",cat);
	if (!param.save){
		fprintf(stderr,"System Operation:%s\n",rm);
	}
#endif
	
	cat_return = system(cat);
	cat_return=0;
	if(cat_return != 0) sf_error("Error: Something was not ok in the catting of the spikes.");

	if (!param.save){
		rm_return = system(rm);
		if(rm_return != 0) sf_error("Error: Something was not ok in the remove temp files.");
	}

	fclose(fp);
}

unsigned long count_lines_of_file(const char *file_patch)
/*<Returns number of lines in an ascii file.>*/
{
	FILE *fp = fopen(file_patch, "r");
	unsigned long int n = 0;
	int pc = EOF;
	int c;

	if(fp == NULL){
		fclose(fp);
		return 0;
	}

	while ((c = fgetc(fp)) != EOF) {
		if (c == '\n')
			++n;
		pc = c;
	}
	if (pc != EOF && pc != '\n')
		++n;

	fclose(fp);
	return n;
}
