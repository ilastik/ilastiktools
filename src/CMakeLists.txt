
#--------------------------------------------------------------
# vigra
#--------------------------------------------------------------
FIND_PACKAGE(VIGRA REQUIRED)
include_directories(${VIGRA_INCLUDE_DIR})

#--------------------------------------------------------------
# Boost
#--------------------------------------------------------------
find_package(Boost 1.47 REQUIRED)
find_package(Boost COMPONENTS python REQUIRED)

#--------------------------------------------------------------
# Python Lib
#--------------------------------------------------------------
include_directories(
    ${PYTHON_INCLUDE_PATH}
    ${PYTHON_NUMPY_INCLUDE_DIR}
    ${Boost_INCLUDE_DIR}
    ${Boost_PYTHON_INCLUDE_DIR}
)

add_subdirectory(python)
