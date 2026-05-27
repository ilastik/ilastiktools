#----------------------------------------------------
# default build-type (release)
# (the next lines must be called bevore project(ilastiktools))
#----------------------------------------------------
IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel.")
ENDIF()

#----------------------------------------------------
# ilastiktools
#----------------------------------------------------
cmake_minimum_required(VERSION 3.10)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)

project(ilastiktools)
set (ILASTIKTOOLS_VERSION_MAJOR 0)
set (ILASTIKTOOLS_VERSION_MINOR 3)
set (ILASTIKTOOLS_VERSION_PATCH 0)

#--------------------------------------------------------
# enable_testing
#--------------------------------------------------------
enable_testing()

#--------------------------------------------------------
# global headers
#--------------------------------------------------------
include_directories(include)

FIND_PACKAGE(Vigra REQUIRED)
include_directories(${Vigra_INCLUDE_DIRS})

FIND_PACKAGE(Threads REQUIRED)  # includes pthread

#--------------------------------------------------------
# OpenMP
#--------------------------------------------------------
option (WITH_OPENMP "use openmp" ON)
if(WITH_OPENMP)
  FIND_PACKAGE(OpenMP REQUIRED)
  message(STATUS "build with OpenMP")
  add_definitions(-DWITH_OPENMP)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
   message(STATUS "build without openMP -> multithreaded options disabled")
endif(WITH_OPENMP)

#--------------------------------------------------------
# source directory
#--------------------------------------------------------
add_subdirectory(src)
