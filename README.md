<img src="doc/img/featherGPU.png" />

# A GPU Lightweight Compression Library

If you use our work please cite us as follows:

- *Krzysztof Kaczmarski, Piotr Przymus, Fixed length lightweight compression for GPU revised, Journal of Parallel and Distributed Computing, Available online 18 April 2017, ISSN 0743-7315, https://doi.org/10.1016/j.jpdc.2017.03.011.*

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
