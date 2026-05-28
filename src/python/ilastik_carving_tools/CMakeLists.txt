find_package(Python REQUIRED Interpreter Development)
find_package(pybind11 CONFIG)

pybind11_add_module(ilastik_carving_tools core.cxx)

install(TARGETS ilastik_carving_tools
          LIBRARY DESTINATION ${Python_SITELIB}
          RUNTIME DESTINATION ${Python_SITELIB})
