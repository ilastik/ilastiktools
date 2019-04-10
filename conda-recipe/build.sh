if [[ $(uname) == 'Darwin' ]]; then
    WITH_OPENMP=OFF # As of Xcode 8.0, Apple's clang doesn't include openmp support.
                    # We could install our own version of clang-omp via brew, but I'm not exploring that right now.
    ILASTIKTOOLS_CXXFLAGS="${CXXFLAGS} -std=c++11 -stdlib=libc++"
else
    WITH_OPENMP=ON
    ILASTIKTOOLS_CXXFLAGS="${CXXFLAGS} -std=c++11"
fi


mkdir build
cd build
cmake ..\
    -DCMAKE_CXX_FLAGS="${ILASTIKTOOLS_CXXFLAGS}" \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_PREFIX_PATH=${PREFIX} \
    -DPYTHON_EXECUTABLE=${PYTHON} \
    -DPYTHON_LIBRARY=${PREFIX}/lib/libpython${CONDA_PY}${SHLIB_EXT} \
    -DPYTHON_INCLUDE_DIR=${PREFIX}/include/python${CONDA_PY} \
    -DVIGRA_INCLUDE_DIR=${PREFIX}/include \
    -DVIGRA_IMPEX_LIBRARY=${PREFIX}/lib/libvigraimpex${SHLIB_EXT} \
    -DVIGRA_NUMPY_CORE_LIBRARY=${SP_DIR}/vigra/vigranumpycore${SHLIB_EXT} \
    -DWITH_OPENMP=${WITH_OPENMP} \
##

make -j${CPU_COUNT}
make install
