project(Boost_Test)
cmake_minimum_required(VERSION 2.8.12)

# find python


find_package(PythonInterp REQUIRED)
find_package(PythonLibs ${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR} EXACT REQUIRED)
find_package(BLAS REQUIRED)
find_package(Boost REQUIRED)
find_package(Threads)


find_path(BLAS_INCLUDE_DIRS cblas.h
  /usr/include
  /usr/local/include
)

# now search for the boost component
# depending on the boost version it is called either python,
# python2, python27, python3, python36, python37, ...
list(
    APPEND _components
        ${PYTHON_VERSION_MAJOR}${PYTHON_VERSION_MINOR}
         ${PYTHON_VERSION_MAJOR}
)



set(Boost_DEBUG ON)
 set(_boost_python_found "")
 foreach(_component IN ITEMS ${_components})
   find_package(Boost COMPONENTS python${_component})
   find_package(Boost COMPONENTS numpy${_component})
   if(Boost_FOUND)
     set(_boost_python_found ${_component})
     break()
   endif()
endforeach()

if(_boost_python_found STREQUAL "")
    message(FATAL_ERROR "No matching Boost.Python component found")
endif()

include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("${Boost_INCLUDE_DIRS}")
include_directories("${BLAS_INCLUDE_DIRS}")

message(BLAS_INCLUDE_DIRS "${BLAS_INCLUDE_DIRS}")
message(BLAS_LIBRARIES " ${BLAS_LIBRARIES}")
message(PYTHON_INCLUDE_DIRS " ${PYTHON_INCLUDE_DIRS}")
message(PYTHON_LIBRARIES " ${PYTHON_LIBRARIES}")
message(Boost_INCLUDE_DIRS " ${Boost_INCLUDE_DIRS}")
message(Boost_LIBRARIES " ${Boost_LIBRARIES}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Wall -lpthread")
ADD_SUBDIRECTORY(src)
