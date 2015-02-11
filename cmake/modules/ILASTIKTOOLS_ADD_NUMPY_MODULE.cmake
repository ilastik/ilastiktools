###################################################################
#
# ILASTIKTOOLS_ADD_NUMPY_MODULE: setup a module dependening on ilastiktools_python
#
# ILASTIKTOOLS_ADD_NUMPY_MODULE(modulename [SOURCES] source1.cpp, source2.cpp ...
#                                   [LIBRARIES dependency1 dependency2 ...]
#                                   [VIGANUMPY])
#
#        'modulename' is the module name to be used within Python (e.g. 'import modulename'). 
#        Unless 'VIGRANUMPY' is specified (see below), it is also the cmake target name.
#
#        SOURCE are the C++ sources of the module, LIBRARIES the necessary libraries. 
#        Dependency syntax must conform to the requirements of the cmake command
#        TARGET_LINK_LIBRARIES. Modules are automatically linked against vigranumpycore
#        and its dependencies (libpython, boost_python), so it is not necessary to state 
#        this dependency explicitly.
#   
#        If VIGRANUMPY is given, the module is considered part of 'ilastiktools_python' and will
#        be compiled and installed along with the other ilastiktools_python modules (otherwise,
#        no installation target will be defined). The cmake target name becomes 
#        'vigranumpy_modulename' in order to get useful alphabetic sorting of 
#        targets in project files.
FUNCTION(ILASTIKTOOLS_ADD_NUMPY_MODULE target_with_prefix)


    get_filename_component(target ${target_with_prefix} NAME)
    get_filename_component(target_dir_prefix ${target_with_prefix} DIRECTORY)
    SET(target_tmp_path   ${ilastiktools_python_tmp_dir}/${target_dir_prefix})
    #MESSAGE(STATUS "ilastiktools_python_tmp_dir" ${ilastiktools_python_tmp_dir})
    #MESSAGE(STATUS "TARGETNAME "${target})
    #MESSAGE(STATUS "FOLDERPREFIX "${target_dir_prefix})
    #MESSAGE(STATUS "TARGET_TMP_DIR "${target_tmp_path})


    set(PART_OF_VIGRANUMPY 1)
    # parse the args
    set(v SOURCES)
    set(PART_OF_VIGRANUMPY 1)
    foreach(i ${ARGN})
        if(${i} MATCHES "^SOURCES$")
            set(v SOURCES)
        elseif(${i} MATCHES "^LIBRARIES$")
            set(v LIBRARIES)
        elseif(${i} MATCHES "^VIGRANUMPY$")
            set(PART_OF_VIGRANUMPY 1)
        else()
            set(${v} ${${v}} ${i})
        endif()
    endforeach(i)
    


    IF(PART_OF_VIGRANUMPY)
        set(TARGET_NAME ilastiktools_python_${target})
        #if(target MATCHES "^core$")
        #    set(LIBRARY_NAME vigranumpycore)
        #else()
            set(LIBRARY_NAME ${target})
        #endif()
    ELSE()
        set(TARGET_NAME ${target})
        set(LIBRARY_NAME ${target})
    ENDIF()

    ADD_LIBRARY(${TARGET_NAME} SHARED ${SOURCES})    
    
    IF(PART_OF_VIGRANUMPY)
        ADD_DEPENDENCIES(ilastiktools_python ${TARGET_NAME})
        
        # Store dependencies as a custom target property, so that we can 
        # later query them.
        # TODO: Does cmake provide a standard way to query the dependencies? 
        GET_TARGET_PROPERTY(VIGRANUMPY_DEPENDS ilastiktools_python VIGRA_DEPENDS)
        IF(NOT VIGRANUMPY_DEPENDS)
            set(VIGRANUMPY_DEPENDS "")
        ENDIF()
        SET_TARGET_PROPERTIES(ilastiktools_python PROPERTIES VIGRA_DEPENDS "${VIGRANUMPY_DEPENDS} ${TARGET_NAME}")
    ENDIF()
    
    #if(DEFINED LIBRARIES)
        TARGET_LINK_LIBRARIES(${TARGET_NAME} 
            ${LIBRARIES} 
            ${Boost_PYTHON_LIBRARIES}
            ${VIGRA_IMPEX_LIBRARY}
            ${VIGRA_NUMPY_CORE_LIBRARY}
            ${VIGRA_NUMPY_IMPEX_LIBRARY}

        )
    #endif()
    
    # if(LIBRARY_NAME MATCHES "^vigranumpycore$")
        # TARGET_LINK_LIBRARIES(${TARGET_NAME} ${VIGRANUMPY_LIBRARIES})
    # else()
        # TARGET_LINK_LIBRARIES(${TARGET_NAME} ${VIGRANUMPY_LIBRARIES} vigranumpy_core)
    # endif()
    
    TARGET_LINK_LIBRARIES(${TARGET_NAME} ${VIGRANUMPY_LIBRARIES})
    
    IF(PYTHON_PLATFORM MATCHES "^windows$")
        SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${LIBRARY_NAME}" 
                                                           PREFIX "" SUFFIX  ".pyd")
    ELSEIF(MACOSX)
        SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${LIBRARY_NAME}" PREFIX "" 
                              SUFFIX ".so" INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/${VIGRANUMPY_INSTALL_DIR}/vigra") 
    ELSE()
        SET_TARGET_PROPERTIES(${TARGET_NAME} PROPERTIES OUTPUT_NAME "${LIBRARY_NAME}" 
                                                           PREFIX "")
    ENDIF()
    
    IF(PART_OF_VIGRANUMPY)
        #IF(PYTHON_PLATFORM MATCHES "^windows$")
        #    INSTALL(TARGETS ${TARGET_NAME} RUNTIME DESTINATION ${VIGRANUMPY_INSTALL_DIR}/vigra)
        #ELSE()
        #    INSTALL(TARGETS ${TARGET_NAME}
        #            LIBRARY DESTINATION ${VIGRANUMPY_INSTALL_DIR}/vigra)
        #ENDIF()
        

        file(MAKE_DIRECTORY ${target_tmp_path})

        # create a temporary ilastiktools_python installation in ${vigranumpy_tmp_dir}
        # (required for testing and documentation generation)
        GET_TARGET_PROPERTY(loc ${TARGET_NAME} LOCATION)
        ADD_CUSTOM_COMMAND(
            TARGET ${TARGET_NAME}
            POST_BUILD
            COMMAND ${CMAKE_COMMAND} ARGS -E copy_if_different ${loc} ${target_tmp_path}
            COMMENT "Copying module to temporary module directory")


    ENDIF()
ENDFUNCTION(ILASTIKTOOLS_ADD_NUMPY_MODULE)
