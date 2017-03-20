Default:
	icc -qopenmp -qopt-report-phase=vec -O3 phi_f.c -o phi_f.out -fPIC
	./phi_f.out

setup:
	#COPY PASTE THESE INSTRUCTIONS
	#THEY DO NOT WORK WHEN CALLED FROM MAKE
	source /opt/intel/parallel_studio_xe_2017.2.050/bin/psxevars.sh
	export MIC_ENV_PREFIX=MIC
	export OMP_NUM_THREADS=240
	export MIC_KMP_AFFINITY=scatter
