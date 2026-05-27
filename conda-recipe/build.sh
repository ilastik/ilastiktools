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
    ${CMAKE_ARGS} \
    -DCMAKE_CXX_FLAGS="${ILASTIKTOOLS_CXXFLAGS}" \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_PREFIX_PATH=${PREFIX} \
    -DPython_EXECUTABLE=${PYTHON} \
    -DWITH_OPENMP=${WITH_OPENMP} \
##

make -j${CPU_COUNT}
make install
