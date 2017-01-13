export CXXFLAGS=""
export CFLAGS=""
export LDFLAGS=""

# Depending on our platform, shared libraries end with either .so or .dylib
if [[ `uname` == 'Darwin' ]]; then
    DYLIB_EXT=dylib
else
    DYLIB_EXT=so
fi

ILASTIKTOOLS_CXXFLAGS="${CFLAGS} -std=c++11"

PY_VER=$(python -c "import sys; print('{}.{}'.format(*sys.version_info[:2]))")
PY_ABIFLAGS=$(python -c "import sys; print('' if sys.version_info.major == 2 else sys.abiflags)")
PY_ABI=${PY_VER}${PY_ABIFLAGS}

mkdir build
cd build
cmake ..\
    -DCMAKE_C_COMPILER=${PREFIX}/bin/gcc \
    -DCMAKE_CXX_COMPILER=${PREFIX}/bin/g++ \
    -DCMAKE_CXX_FLAGS="${ILASTIKTOOLS_CXXFLAGS}" \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_PREFIX_PATH=${PREFIX} \
    -DPYTHON_EXECUTABLE=${PYTHON} \
    -DPYTHON_LIBRARY=${PREFIX}/lib/libpython${PY_ABI}.${DYLIB_EXT} \
    -DPYTHON_INCLUDE_DIR=${PREFIX}/include/python${PY_ABI} \
    -DVIGRA_INCLUDE_DIR=${PREFIX}/include \
    -DVIGRA_IMPEX_LIBRARY=${PREFIX}/lib/libvigraimpex.${DYLIB_EXT} \
    -DVIGRA_NUMPY_CORE_LIBRARY=${SP_DIR}/vigra/vigranumpycore.so \
    -DWITH_OPENMP=ON \
##

make -j${CPU_COUNT}
make install
