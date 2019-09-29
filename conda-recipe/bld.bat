mkdir build
cd build

set CONFIGURATION=Release
set PATH=%PATH%;%LIBRARY_PREFIX%\bin

cmake .. ^
	-G "NMake Makefiles" ^
    -DCMAKE_BUILD_TYPE=%CONFIGURATION% ^
	-DCMAKE_PREFIX_PATH="%LIBRARY_PREFIX%" ^
	-DCMAKE_INSTALL_PREFIX="%LIBRARY_PREFIX%" ^
	-DPYTHON_EXECUTABLE="%PYTHON%" ^
	-DCMAKE_CXX_FLAGS="-DBOOST_ALL_NO_LIB /EHsc" ^
	-DVIGRA_INCLUDE_DIR="%LIBRARY_PREFIX%\include" ^
	-DWITH_OPENMP=ON

if errorlevel 1 exit 1

nmake all
if errorlevel 1 exit 1

nmake install
if errorlevel 1 exit 1