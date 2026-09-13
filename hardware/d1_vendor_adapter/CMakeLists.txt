cmake_minimum_required(VERSION 3.8)
project(loco_mani_d1_vendor_adapter)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)

# Pass the exact D1 vendor package root on the target image.  The fallback
# roots make this package easy to build in the developer workspace, but the
# library/header pair must always come from the same vendor installation.
set(D1_TITA_ROBOT_ROOT "" CACHE PATH "Root of the tita_robot vendor package")
if(NOT D1_TITA_ROBOT_ROOT AND DEFINED ENV{D1_TITA_ROBOT_ROOT})
  set(D1_TITA_ROBOT_ROOT "$ENV{D1_TITA_ROBOT_ROOT}")
endif()
set(_d1_roots)
if(D1_TITA_ROBOT_ROOT)
  list(APPEND _d1_roots "${D1_TITA_ROBOT_ROOT}")
endif()
list(APPEND _d1_roots
  /opt/d1_ros2
  /home/hh/ddt_ros2_ws/ddt_ros2_control/hardware/tita_robot
  /home/hh/ddt_sim2real/ddt_ros2_ws/src/hardware/tita_robot
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(_d1_arch_suffixes lib/arm64 lib)
else()
  set(_d1_arch_suffixes lib/amd64 lib)
endif()

find_path(D1_TITA_INCLUDE_DIR
  NAMES tita_robot/canfd_api.hpp
  PATHS ${_d1_roots}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
)
find_library(D1_TITA_LIBRARY
  NAMES tita_robot libtita_robot
  PATHS ${_d1_roots}
  PATH_SUFFIXES ${_d1_arch_suffixes}
  NO_DEFAULT_PATH
)

if(D1_TITA_INCLUDE_DIR AND D1_TITA_LIBRARY)
  add_library(${PROJECT_NAME} SHARED src/d1_vendor_adapter.cpp)
  target_include_directories(${PROJECT_NAME}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:include>
    PRIVATE ${D1_TITA_INCLUDE_DIR}
  )
  target_link_libraries(${PROJECT_NAME} PRIVATE ${D1_TITA_LIBRARY})
  set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME "loco_mani_d1_vendor_adapter")
  install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
  )
  message(STATUS "Building D1 adapter with ${D1_TITA_INCLUDE_DIR} and ${D1_TITA_LIBRARY}")
else()
  message(WARNING
    "D1 vendor header/library not found; ${PROJECT_NAME} will install its C header "
    "but no shared adapter. Pass -DD1_TITA_ROBOT_ROOT=/path/to/tita_robot on the target.")
endif()

install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)
ament_package()
