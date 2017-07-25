import os, sys, re, string
sys.path.append('../../framework')
import bldutil

progs = ' ' 
# vweks3d2
# revolve_driver
# waveadjtest

libprop = ''

ccprogs = 'fraclr2_zp'
# ewedc3pgrad ewedc3sgrad cfftrtm3
# vwelr3
# eiktest eiktest2 cisolr1eik eiksol

mpi_progs = 'mpiqfwi_sls 2dvisco_acoustic 2dvisco_acoustic_fwi_gradient mpilsrtm1 '
#for distributed FFTW3
#mpiwave2 mpiwave3 mpifftexp1 mpiwave2kiss mpiwave3kiss mpifftexp1kiss

mpicxx_progs = ''

pyprogs = ''
pymods = ''

try:  # distributed version
    Import('env root pkgdir bindir libdir incdir')
    env = env.Clone()
except: # local version
    env = bldutil.Debug()
    root = None
    SConscript('../lexing/SConstruct')

env.Prepend(CPPPATH=['../../include'],
            LIBPATH=['../../lib'],
            LIBS=[env.get('DYNLIB','')+'rsf'])

fftw = env.get('FFTW')
if fftw:
    env.Prepend(CPPDEFINES=['SF_HAS_FFTW'])

src = Glob('[a-z]*.c')
for source in src:
    inc = env.RSF_Include(source,prefix='')
    obj = env.StaticObject(source)
    env.Ignore(inc,inc)
    env.Depends(obj,inc)

csrc = Glob('[a-z]*.cc')
for source in csrc:
    inc = env.RSF_Include(source,prefix='')
    obj = env.StaticObject(source)
    env.Ignore(inc,inc)
    env.Depends(obj,inc)

mpicc = env.get('MPICC')
mpicxx = env.get('MPICXX')
mpi_src = Glob('Q[a-z]*.c')
if mpicc:
    for source in mpi_src:
        inc = env.RSF_Include(source,prefix='')
        env.Ignore(inc,inc)
        obj = env.StaticObject(source,CC=mpicc)
        env.Depends(obj,inc)

mains = Split(progs+' '+libprop)
for prog in mains:
    sources = ['M' + prog]
    bldutil.depends(env,sources,'M'+prog)
    prog = env.Program(prog,map(lambda x: x + '.c',sources))
    if root:
        env.Install(bindir,prog)

mpi_mains = Split(mpi_progs)
for prog in mpi_mains:
    sources = ['M' + prog]
    bldutil.depends(env,sources,'M'+prog)
    if mpicc:
        env.StaticObject('M'+prog+'.c',CC=mpicc)
        #for distributed FFTW3
        #prog = env.Program(prog,map(lambda x: x + '.o',sources),CC=mpicc,LIBS=env.get('LIBS')+['fftw3f_mpi'])
        prog = env.Program(prog,map(lambda x: x + '.o',sources),CC=mpicc)
    else:
        prog = env.RSF_Place('sf'+prog,None,var='MPICC',package='mpi')
    if root:
        env.Install(bindir,prog)


if 'c++' in env.get('API',[]):
    lapack = env.get('LAPACK')
else:
    lapack = None

if lapack:
    libsxx = [env.get('DYNLIB','')+'rsf++','vecmatop']
    if not isinstance(lapack,bool):
        libsxx.extend(lapack)
    env.Prepend(LIBS=libsxx)

#ccsubs = 'lowrank.cc fftomp.c rtmutil.c ksutil.c revolve.c'
ccmains = Split(ccprogs)
for prog in ccmains:
    sources = ['M' + prog + '.cc']
#   if prog == 'cfftrtm3':
#       sources += Split(ccsubs)
    if lapack:
        prog = env.Program(prog,sources)
    else:
        prog = env.RSF_Place('sf'+prog,None,var='LAPACK',package='lapack')
    if root:
        env.Install(bindir,prog)


##################################################################################################################
# To use sfmpicfftrtm, one needs to obtain the source file revolve.c from http://dl.acm.org/citation.cfm?id=347846
##################################################################################################################
xxsubs = 'lowrank fftomp rtmutil ksutil revolve'
mpicxx_mains = Split(mpicxx_progs)
for prog in mpicxx_mains:
    sources = ['M' + prog] + Split(xxsubs)
    if FindFile('revolve.c','.') and mpicxx:
        env.StaticObject('M'+prog+'.cc',CXX=mpicxx)
        prog = env.Program(prog,map(lambda x: x + '.o',sources),CXX=mpicxx)
    else:
        prog = env.RSF_Place('sf'+prog,None,var='MPICXX',package='mpi')
    if root:
        env.Install(bindir,prog)

for prog in Split(''):
    sources = ['Test' + prog,prog]
    if prog=='':
        sources.append('cgmres')
    bldutil.depends(env,sources,prog)
    sources = map(lambda x: x + '.o',sources)
    env.Object('Test' + prog + '.c')
    env.Program(sources,PROGPREFIX='',PROGSUFFIX='.x')

######################################################################
# PYTHON METAPROGRAMS (python API not needed)
######################################################################

if root: # no compilation, just rename
    pymains = Split(pyprogs)
    exe = env.get('PROGSUFFIX','')
    for prog in pymains:
        binary = os.path.join(bindir,'sf'+prog+exe)
        env.InstallAs(binary,'M'+prog+'.py')
        env.AddPostAction(binary,Chmod(str(binary),0755))
    for mod in Split(pymods):
        env.Install(pkgdir,mod+'.py')

######################################################################
# SELF-DOCUMENTATION
######################################################################

if root:
    user = os.path.basename(os.getcwd())
    main = 'sf%s.py' % user
    
    docs = map(lambda prog: env.Doc(prog,'M' + prog),mains+mpi_mains) +  \
           map(lambda prog: env.Doc(prog,'M'+prog+'.py',lang='python'),pymains) + \
           map(lambda prog: env.Doc(prog,'M%s.cc' % prog,lang='c++'),ccmains+mpicxx_mains) 
          
    env.Depends(docs,'#/framework/rsf/doc.py')	
    doc = env.RSF_Docmerge(main,docs)
    env.Install(pkgdir,doc)
