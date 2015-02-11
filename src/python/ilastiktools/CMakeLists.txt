

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
    ${PYTHON_LIBRARIES}
    ${Boost_PYTHON_LIBRARIES}
    ${VIGRA_IMPEX_LIBRARY}
    ${VIGRA_NUMPY_CORE_LIBRARY}
    ${VIGRA_NUMPY_IMPEX_LIBRARY}
)
IF(PYTHON_PLATFORM MATCHES "windows$" OR WIN32)
    SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "_core" PREFIX "" SUFFIX ".pyd")
ELSEIF(APPLE)
    SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "_core" PREFIX "" SUFFIX ".so")
ELSE()
    SET_TARGET_PROPERTIES( _core PROPERTIES OUTPUT_NAME "_core" PREFIX "")
ENDIF()



file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py  DESTINATION ${CMAKE_CURRENT_BINARY_DIR} )

install(FILES __init__.py
        DESTINATION ${PYTHON_PREFIX}/lib/python2.7/site-packages/ilastiktools)

install(TARGETS _core
        LIBRARY DESTINATION ${PYTHON_PREFIX}/lib/python2.7/site-packages/ilastiktools)

