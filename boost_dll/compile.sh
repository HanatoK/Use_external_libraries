#!/bin/sh
g++ -c -o libfit_plane_eigen.o lib1.cpp -fPIC -I/usr/include/eigen3 -O2
gcc -shared -o libfit_plane_eigen.so libfit_plane_eigen.o
g++ -c -o libfit_plane_gsl.o lib2.cpp -fPIC
gcc -shared -o libfit_plane_gsl.so libfit_plane_gsl.o -lgsl -lopenblas -O2
g++ main.cpp -o main -lboost_filesystem -O2
