

#--------------------------------------------------------------
# Add Library
#--------------------------------------------------------------
add_library( _core SHARED
    core.cxx
)
#--------------------------------------------------------------
# Link libraries
#--------------------------------------------------------------
find_library(RT rt)
target_link_libraries( _core
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)
IF(PYTHON_PLATFORM MATCHES "windows$" OR WIN32)
SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "core" PREFIX "" SUFFIX ".pyd")
ELSEIF(MACOSX)
SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "core" PREFIX "" SUFFIX ".so")
ELSE()
SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "core" PREFIX "_")
ENDIF()



file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )