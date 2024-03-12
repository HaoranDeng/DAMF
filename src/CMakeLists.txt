set(MODULE_NAME  embedding_propagation)

include_directories(${CMAKE_SOURCE_DIR})

add_library(${MODULE_NAME} SHARED
	stream_ppr.h
	stream_ppr.cc
	)

if (UNIX)
  set_target_properties(${MODULE_NAME}
    PROPERTIES
    PREFIX ""
  )
elseif (WIN32)
  set_target_properties(${MODULE_NAME}
  PROPERTIES
  SUFFIX ".pyd"
  )
endif()

target_link_libraries(${MODULE_NAME}
  ${Boost_LIBRARIES}
  ${PYTHON_LIBRARIES}
  ${BLAS_LIBRARIES}
  ${CMAKE_THREAD_LIBS_INIT}
)

SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/)
