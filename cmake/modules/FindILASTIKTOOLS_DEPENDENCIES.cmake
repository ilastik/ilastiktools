# - Find ILASTIKTOOLS_DEPENDENCIES
#
MESSAGE(STATUS "Checking ILASTIKTOOLS_DEPENDENCIES")

FIND_PACKAGE(PythonInterp REQUIRED)
FIND_PACKAGE(PythonLibs REQUIRED)

# find Python library
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                 "import sys; print(sys.exec_prefix)"
                  OUTPUT_VARIABLE PYTHON_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)

IF(APPLE AND ${PYTHON_PREFIX} MATCHES ".*framework.*")
    SET(PYTHON_LIBRARIES "${PYTHON_PREFIX}/Python"
        CACHE FILEPATH "Python libraries"
        FORCE)
ELSE()
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                     "import sys; skip = 2 if sys.platform.startswith('win') else 1; print('python' + sys.version[0:3:skip] + ('' if sys.version_info.major == 2 else sys.abiflags) )"
                      OUTPUT_VARIABLE PYTHON_LIBRARY_NAME OUTPUT_STRIP_TRAILING_WHITESPACE)
    FIND_LIBRARY(PYTHON_LIBRARIES ${PYTHON_LIBRARY_NAME} HINTS "${PYTHON_PREFIX}"
                 PATH_SUFFIXES lib lib64 libs DOC "Python libraries")
ENDIF()

# find Python includes
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                "from distutils.sysconfig import *; print(get_python_inc())"
                OUTPUT_VARIABLE PYTHON_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
SET(PYTHON_INCLUDE_PATH ${PYTHON_INCLUDE}
    CACHE PATH "Path to Python include files"
    FORCE)

IF(PYTHON_LIBRARIES AND PYTHON_INCLUDE_PATH)
    MESSAGE(STATUS "Found Python libraries: ${PYTHON_LIBRARIES}")
    MESSAGE(STATUS "Found Python includes:  ${PYTHON_INCLUDE_PATH}")
    SET(PYTHONLIBS_FOUND TRUE)
ELSE()
    MESSAGE(STATUS "Could NOT find Python libraries and/or includes")
ENDIF()

######################################################################
#
#      find default install directory for Python modules
#      (usually PYTHONDIR/Lib/site-packages)
#
######################################################################
IF(NOT DEFINED PYTHON_SITE_PACKAGES OR PYTHON_SITE_PACKAGES MATCHES "^$")
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                     "from distutils.sysconfig import *; print(get_python_lib(1))"
                      OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
    FILE(TO_CMAKE_PATH ${PYTHON_SITE_PACKAGES} PYTHON_SITE_PACKAGES)
ENDIF()
SET(PYTHON_SITE_PACKAGES ${PYTHON_SITE_PACKAGES}
    CACHE PATH "where to install the VIGRA Python package" FORCE)
# this is the install path relative to CMAKE_INSTALL_PREFIX,
# use this in INSTALL() commands to get packaging right
FILE(RELATIVE_PATH PYTHON_SITE_PACKAGES ${CMAKE_INSTALL_PREFIX} ${PYTHON_SITE_PACKAGES})



######################################################################
#
#      find Python platform
#
######################################################################
execute_process(COMMAND ${PYTHON_EXECUTABLE} -c
                 "import sys; p = sys.platform; print('windows' if p.startswith('win') else p)"
                  OUTPUT_VARIABLE PYTHON_PLATFORM OUTPUT_STRIP_TRAILING_WHITESPACE)

######################################################################
#
#      set outputs
#
######################################################################
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(ILASTIKTOOLS_DEPENDENCIES DEFAULT_MSG
                     PYTHONLIBS_FOUND
                     Boost_PYTHON_FOUND PYTHON_NUMPY_INCLUDE_DIR PYTHON_SITE_PACKAGES)

IF(NOT ILASTIKTOOLS_INCLUDE_DIRS OR ILASTIKTOOLS_INCLUDE_DIRS MATCHES "-NOTFOUND")
    #note that the numpy include dir is set _before_ the python include dir, such that
    #installing a more recent version of numpy on top of an existing python installation
    #works (otherwise, numpy includes are picked up from ${PYTHON_INCLUDE_PATH}/numpy )
    SET(ILASTIKTOOLS_INCLUDE_DIRS ${PYTHON_NUMPY_INCLUDE_DIR} ${PYTHON_INCLUDE_PATH} ${Boost_INCLUDE_DIR})
ENDIF()
SET(ILASTIKTOOLS_INCLUDE_DIRS ${ILASTIKTOOLS_INCLUDE_DIRS}
    CACHE PATH "include directories needed by ilastiktools Python bindings"
    FORCE)
IF(NOT ILASTIKTOOLS_LIBRARIES OR ILASTIKTOOLS_LIBRARIES MATCHES "-NOTFOUND")
    SET(ILASTIKTOOLS_LIBRARIES ${PYTHON_LIBRARIES} ${Boost_PYTHON_LIBRARY})
ENDIF()
SET(ILASTIKTOOLS_LIBRARIES ${ILASTIKTOOLS_LIBRARIES}
    CACHE FILEPATH "libraries needed by ilastiktools Python bindings"
    FORCE)
