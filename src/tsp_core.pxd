# distutils: language = c++
# distutils: sources = src/tsp_utils.cpp

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string

cdef extern from "tsp_utils.h":
    cdef cppclass TSPUtils:
        @staticmethod
        vector[pair[double, double]] loadCities(const string& filepath)
        
        @staticmethod
        vector[vector[double]] buildDistanceMatrix(
            const vector[pair[double, double]]& cities)
        
        @staticmethod
        double calcPathLength(const vector[int]& path,
                             const vector[vector[double]]& dist)
