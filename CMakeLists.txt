cmake_minimum_required(VERSION 2.8.3)
project(amcl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -O3")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static-libgcc -static-libstdc++")

find_package(catkin REQUIRED
  COMPONENTS
    rosbag
    roscpp
    tf
    nav_msgs
)

find_package(PkgConfig)
pkg_check_modules(csm REQUIRED csm)

find_package(Boost REQUIRED)

catkin_package (
  CATKIN_DEPENDS
    rosbag
    roscpp
    tf
  INCLUDE_DIRS include
)

include_directories (
  include
  include/amcl 
  include/amcl/map 
  include/amcl/sensors 
  include/amcl/pf
)

include_directories (
  ${catkin_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${csm_INCLUDE_DIRS}
)

link_directories(${csm_LIBRARY_DIRS})

add_executable(amcl
  src/amcl/pf/pf.c
  src/amcl/pf/pf_kdtree.c
  src/amcl/pf/pf_pdf.c
  src/amcl/pf/pf_vector.c
  src/amcl/pf/eig3.c
  src/amcl/pf/pf_draw.c
  src/amcl/map/map.c
  src/amcl/map/map_cspace.cpp
  src/amcl/map/map_range.c
  src/amcl/map/map_store.c
  src/amcl/map/map_draw.c
  src/amcl/sensors/amcl_sensor.cpp
  src/amcl/sensors/amcl_odom.cpp
  src/amcl/sensors/amcl_laser.cpp
  src/amcl_node.cc
  src/main.cc
)
                    
target_link_libraries(amcl
  ${catkin_LIBRARIES}
  ${Boost_LIBRARIES}
  ${csm_LIBRARIES}
)

install( TARGETS amcl
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
