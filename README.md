<img src="doc/img/featherGPU.png" />

# A GPU Lightweight Compression Library

## Details and Citing
You my find more details in:
K. Kaczmarski and P. Przymus, *Fixed Length Lightweight Compression for GPU Revised*, Journal of Parallel and Distributed Computing, 2017.


## Maintained by
Faculty of Mathematics and Information Science. Warsaw University of Technology

<img src="doc/img/pw-logo.png" width="70" height="70" /> <img src="doc/img/wut-logo.jpg" width="70" height="70" />

The Faculty of Mathematics and Computer Science Nicolaus Copernicus University in Toruń

## Building
```sh
mkdir build
cd build
cmake ..
make
```
## Running tests
```sh
# Help
./run_tests -- help
# list avalible test variants (tags)
./run_tests -t
#to run a test
#./run_tests [ALG_TAG][TEST_TYPE]
./run_test [AAFL][BENCHMARK]
```


Logo based on graphic from [freelogovector.com](http://www.freelogovector.com/detail-f-Feather_vector.php)
