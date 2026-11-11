// Reference: volumetric path tracer (delta tracking, conservative HG scattering)
// for the crystal-moon density. Outputs per-light transfer images.
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>
#define SIGMAX 12.0
#define GHG 0.5
#define KQ 11.0
#define RB 1.0
static const double AX[6][3]={{0,1,1.6180339887},{0,-1,1.6180339887},{1,1.6180339887,0},{-1,1.6180339887,0},{1.6180339887,0,1},{1.6180339887,0,-1}};
static double AN[6][3];
static inline double sstep(double a,double b,double x){double t=(x-a)/(b-a);if(t<0)t=0;if(t>1)t=1;return t*t*(3-2*t);}
static inline double dens(const double*p){
  double r2=p[0]*p[0]+p[1]*p[1]+p[2]*p[2]; if(r2>=1.0) return 0;
  double r=sqrt(r2); double m=1.0-sstep(0.86,1.0,r);
  double q=0; for(int i=0;i<6;i++) q+=cos(KQ*(AN[i][0]*p[0]+AN[i][1]*p[1]+AN[i][2]*p[2]));
  double c=sstep(-0.3,1.8,q);
  return SIGMAX*m*(0.12+0.88*c);
}
typedef struct{uint64_t s;}rng;
static inline double rnd(rng*r){r->s^=r->s<<13;r->s^=r->s>>7;r->s^=r->s<<17;return (r->s>>11)*(1.0/9007199254740992.0);}
static inline double hg(double c){double g=GHG;double d=1+g*g-2*g*c;return (1-g*g)/(4*M_PI*d*sqrt(d));}
static void sample_hg(const double*w,double*o,rng*r){
  double g=GHG,xi=rnd(r); double s=(1-g*g)/(1-g+2*g*xi); double ct=(1+g*g-s*s)/(2*g);
  double st=sqrt(fmax(0,1-ct*ct)),ph=2*M_PI*rnd(r);
  double a[3]; if(fabs(w[0])<0.9){a[0]=1;a[1]=0;a[2]=0;}else{a[0]=0;a[1]=1;a[2]=0;}
  double u[3]={w[1]*a[2]-w[2]*a[1],w[2]*a[0]-w[0]*a[2],w[0]*a[1]-w[1]*a[0]};
  double n=sqrt(u[0]*u[0]+u[1]*u[1]+u[2]*u[2]);u[0]/=n;u[1]/=n;u[2]/=n;
  double v[3]={w[1]*u[2]-w[2]*u[1],w[2]*u[0]-w[0]*u[2],w[0]*u[1]-w[1]*u[0]};
  for(int k=0;k<3;k++)o[k]=ct*w[k]+st*(cos(ph)*u[k]+sin(ph)*v[k]);
}
// ratio-tracking transmittance from p along d to the sphere exit
static double trans(const double*p0,const double*d,rng*r){
  double p[3]={p0[0],p0[1],p0[2]},T=1;
  for(;;){double t=-log(1-rnd(r))/SIGMAX;for(int k=0;k<3;k++)p[k]+=t*d[k];
    if(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]>=RB*RB)return T; T*=1-dens(p)/SIGMAX; if(T<1e-6)return 0;}
}
int main(int argc,char**argv){
  if(argc<10){fprintf(stderr,"cx cy cz sx sy sz N spp out\n");return 1;}
  double cd[3]={atof(argv[1]),atof(argv[2]),atof(argv[3])},s[3]={atof(argv[4]),atof(argv[5]),atof(argv[6])};
  int N=atoi(argv[7]),spp=atoi(argv[8]);
  for(int i=0;i<6;i++){double n=sqrt(AX[i][0]*AX[i][0]+AX[i][1]*AX[i][1]+AX[i][2]*AX[i][2]);for(int k=0;k<3;k++)AN[i][k]=AX[i][k]/n;}
  double n=sqrt(cd[0]*cd[0]+cd[1]*cd[1]+cd[2]*cd[2]);for(int k=0;k<3;k++)cd[k]/=n;
  n=sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);for(int k=0;k<3;k++)s[k]/=n;
  double cam[3]={3.4*cd[0],3.4*cd[1],3.4*cd[2]},f[3]={-cd[0],-cd[1],-cd[2]};
  double rt[3]={f[2],0,-f[0]}; // right = f x up(0,1,0) -> (f.y*0-f.z*1, f.z*0-f.x*0, f.x*1-f.y*0) = (-f.z,0,f.x); sign fixed below
  rt[0]=-f[2];rt[1]=0;rt[2]=f[0]; n=sqrt(rt[0]*rt[0]+rt[2]*rt[2]);rt[0]/=n;rt[2]/=n;
  double up[3]={rt[1]*f[2]-rt[2]*f[1],rt[2]*f[0]-rt[0]*f[2],rt[0]*f[1]-rt[1]*f[0]};
  float*img=calloc((size_t)N*N*8,sizeof(float));
  #pragma omp parallel for schedule(dynamic,1)
  for(int j=0;j<N;j++){rng r={.s=0x9E3779B97F4A7C15ULL*(j+1)+12345};for(int w=0;w<8;w++)rnd(&r);
   for(int i=0;i<N;i++){double acc[8]={0};
    for(int sp=0;sp<spp;sp++){int h=(sp&1)*4;
      double u=((i+rnd(&r))/N*2-1)*0.36,v=(1-(j+rnd(&r))/N*2)*0.36;
      double d[3];for(int k=0;k<3;k++)d[k]=f[k]+u*rt[k]+v*up[k];
      double dn=sqrt(d[0]*d[0]+d[1]*d[1]+d[2]*d[2]);for(int k=0;k<3;k++)d[k]/=dn;
      double b=cam[0]*d[0]+cam[1]*d[1]+cam[2]*d[2],c=cam[0]*cam[0]+cam[1]*cam[1]+cam[2]*cam[2]-RB*RB,disc=b*b-c;
      if(disc<=0){acc[h+3]+=1;continue;}
      double t0=-b-sqrt(disc);double p[3];for(int k=0;k<3;k++)p[k]=cam[k]+(t0+1e-9)*d[k];
      int ns=0;
      for(;;){double t=-log(1-rnd(&r))/SIGMAX;for(int k=0;k<3;k++)p[k]+=t*d[k];
        if(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]>=RB*RB)break;
        if(rnd(&r)*SIGMAX<dens(p)){ns++;
          acc[h+0]+=hg(d[0]*s[0]+d[1]*s[1]+d[2]*s[2])*trans(p,s,&r);
          double o[3];sample_hg(d,o,&r);d[0]=o[0];d[1]=o[1];d[2]=o[2];}
      }
      if(ns==0)acc[h+3]+=1; else if(d[1]>0)acc[h+1]+=1; else acc[h+2]+=1;
    }
    for(int k=0;k<8;k++)img[((size_t)j*N+i)*8+k]=(float)(acc[k]/(spp/2));
   }}
  FILE*fo=fopen(argv[9],"wb");fwrite(img,sizeof(float),(size_t)N*N*8,fo);fclose(fo);return 0;
}
