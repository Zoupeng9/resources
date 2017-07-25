#include "stdlib.h"
#include "stdio.h"

void *alloc1 (size_t n1, size_t size);
void **alloc2 (int n1, int n2, int size);
void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size);
float *alloc1float(size_t n1);
float **alloc2float(int n1, int n2);
float ***alloc3float(size_t n1, size_t n2, size_t n3);
int *alloc1int(size_t n1);
int **alloc2int(size_t n1, size_t n2);
char **alloc2char(int n1, int n2);

void free1 (void *p);
void free2 (void **p);
void free3 (void ***p);
void free1float(float *p);
void free2float(float **p);
void free3float(float ***p);
void free2int(int **p);
void free2char(char **p);

void zero1int(int *p, size_t n1);
void zero2int(int **p, size_t n1, size_t n2);
void zero1float(float *p, size_t n1);
void zero2float(float **p, size_t n1, size_t n2);
void zero3float(float ***p, size_t n1, size_t n2, size_t n3);

int *alloc1int(size_t n1)
{
        return (int*)alloc1(n1,sizeof(int));
}

void free2int(int **p)
{
        free2((void**)p);
}

int **alloc2int(size_t n1, size_t n2)
{
        return (int**)alloc2(n1,n2,sizeof(int));
}

float ***alloc3float(size_t n1, size_t n2, size_t n3)
{
        return (float***)alloc3(n1,n2,n3,sizeof(float));
}

void free3float(float ***p)
{
        free3((void***)p);
}

void zero1float(float *p, size_t n1)
{
     int i;
     for(i=0;i<n1;i++) p[i]=0.0;
}

void zero2float(float **p, size_t n1, size_t n2)
{
     int i, j;
     for(i=0;i<n2;i++)
       for(j=0;j<n1;j++)
         p[i][j]=0.0;
}

void zero3float(float ***p, size_t n1, size_t n2, size_t n3)
{
     int i, j, k;
     for(i=0;i<n3;i++)
       for(j=0;j<n2;j++)
          for(k=0;k<n1;k++)
            p[i][j][k]=0.0;
}


void ***alloc3 (size_t n1, size_t n2, size_t n3, size_t size)
{
        size_t i3,i2;
        void ***p;

        if ((p=(void***)malloc(n3*sizeof(void**)))==NULL)
                return NULL;
        if ((p[0]=(void**)malloc(n3*n2*sizeof(void*)))==NULL) {
                free(p);
                return NULL;
        }
        if ((p[0][0]=(void*)malloc(n3*n2*n1*size))==NULL) {
                free(p[0]);
                free(p);
                return NULL;
        }
        for (i3=0; i3<n3; i3++) {
                p[i3] = p[0]+n2*i3;
                for (i2=0; i2<n2; i2++)
                        p[i3][i2] = (char*)p[0][0]+size*n1*(i2+n2*i3);
        }
        return p;
}

void free3 (void ***p)
{
        free(p[0][0]);
        free(p[0]);
        free(p);
}

void zero1int(int *p, size_t n1)
{
     int i;
     for(i=0;i<n1;i++) p[i]=0;
}

void zero2int(int **p, size_t n1, size_t n2)
{
     int i, j;
     for(i=0;i<n2;i++)
       for(j=0;j<n1;j++)
         p[i][j]=0;
}

void free1 (void *p)
{
        free(p);
}

void free1float(float *p)
{
        free1(p);
}

float *alloc1float(size_t n1)
{
        return (float*)alloc1(n1,sizeof(float));
}

void *alloc1 (size_t n1, size_t size)
{
        void *p;

        if ((p=malloc(n1*size))==NULL)
                return NULL;
        return p;
}


void **alloc2 (int n1, int n2, int size)
{
        int i;
        void **p;

        p=(void**)calloc(sizeof(void*),n2);
        p[0]=(void*)calloc(size, n1*n2);
        for (i=0; i<n2; i++)
                p[i] = (char*)p[0]+size*n1*i;
        return p;
}

void free2 (void **p)
{
        free(p[0]);
        free(p);
}
char **alloc2char(int n1, int n2)
{
        return (char**)alloc2(n1,n2,sizeof(char));
}

void free2char(char **p)
{
        free2((void**)p);
}

float **alloc2float(int n1, int n2)
{
        return (float**)alloc2(n1,n2,sizeof(float));
}

void free2float(float **p)
{
        free2((void**)p);
}


