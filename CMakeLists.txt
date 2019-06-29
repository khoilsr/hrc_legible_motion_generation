cmake_minimum_required(VERSION 2.8.3)
project(predictable_motion_generator)


find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  urdf
)

catkin_package(
  CATKIN_DEPENDS
  roscpp std_msgs urdf
  INCLUDE_DIRS include
)


add_definitions (-std=c++11)

include_directories(include
  ${Boost_INCLUDE_DIR}
  /usr/local/include/eigen3/
  ${catkin_INCLUDE_DIRS}
  ${SDL_INCLUDE_DIR} 
)
find_package( Boost 1.34 COMPONENTS filesystem system serialization iostreams REQUIRED)
find_package(SDL REQUIRED)

link_directories(${catkin_LIBRARY_DIRS})

set(bbo src/dmpbbo_node/DistributionGaussian.cpp 
src/dmpbbo_node/UpdateSummary.cpp 
src/dmpbbo_node/UpdaterCovarDecay.cpp 
src/dmpbbo_node/UpdaterMean.cpp)

set(dmp src/dmpbbo_node/Dmp.cpp
src/dmpbbo_node/Trajectory.cpp)

set(dmp_bbo src/dmpbbo_node/TaskSolverDmp.cpp
src/dmpbbo_node/UpdateSummaryParallel.cpp
src/dmpbbo_node/runEvolutionaryOptimization.cpp)

set(TaskViapoint src/dmpbbo_node/TaskViapoint.cpp)

set(dynamicalsystems src/dmpbbo_node/DynamicalSystem.cpp
src/dmpbbo_node/ExponentialSystem.cpp
src/dmpbbo_node/SigmoidSystem.cpp
src/dmpbbo_node/SpringDamperSystem.cpp
src/dmpbbo_node/TimeSystem.cpp)

set(functionapproximators src/dmpbbo_node/BasisFunction.cpp
src/dmpbbo_node/FunctionApproximator.cpp
src/dmpbbo_node/FunctionApproximatorLWR.cpp
src/dmpbbo_node/FunctionApproximatorRBFN.cpp
src/dmpbbo_node/MetaParameters.cpp
src/dmpbbo_node/MetaParametersLWR.cpp
src/dmpbbo_node/MetaParametersRBFN.cpp
src/dmpbbo_node/ModelParameters.cpp
src/dmpbbo_node/ModelParametersLWR.cpp
src/dmpbbo_node/ModelParametersRBFN.cpp
src/dmpbbo_node/Parameterizable.cpp
src/dmpbbo_node/UnifiedModel.cpp)


## Declare a C++ library
add_library(bbo ${SHARED_OR_STATIC} ${bbo})
add_library(dmp ${SHARED_OR_STATIC} ${dmp})
add_library(dmp_bbo ${SHARED_OR_STATIC} ${dmp_bbo})
add_library(taskviapoint ${SHARED_OR_STATIC} ${TaskViapoint})
add_library(dynamicalsystems ${SHARED_OR_STATIC} ${dynamicalsystems})
add_library(functionapproximators ${SHARED_OR_STATIC} ${functionapproximators})
add_library(leftovers src/DmpSolver.cpp src/Kinematics.cpp)
## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(dmp_bbo ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

add_executable(dmp_ros_ctl_ik 
   src/dmp_ros_ctl_ik.cpp
)

target_link_libraries(dmp_ros_ctl_ik taskviapoint dmp functionapproximators dynamicalsystems dmp_bbo bbo leftovers
  /usr/local/include/eigen3
  ${Boost_LIBRARIES}
  ${catkin_LIBRARIES} 
)

