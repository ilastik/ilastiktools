find_package(Python REQUIRED Interpreter Development)
find_package(pybind11 CONFIG)

pybind11_add_module(ilastiktools core.cxx)

install(TARGETS ilastiktools
          LIBRARY DESTINATION ${Python_SITELIB}
          RUNTIME DESTINATION ${Python_SITELIB})
