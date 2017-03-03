# Accelerated Computing - GPU vs Phi
Group Project for Accelerate Computing

### Makefile
__Compile for Cuda__<br />
`make cuda`<br />
__Compile for Xeon Phi__<br />
`make phi`<br />
__Generate Data__<br />
`make ARGS="int" data` Generate n✖n Matrix to text file<br />
`make ARGS="int int" data` Generate n✖m Matrix to text file<br />
line 1 = n
line 2 = m
line 3...end = data
