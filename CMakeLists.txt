# CMake version
cmake_minimum_required(VERSION 3.16.3)

#project info
project(Mmulop LANGUAGES CXX CUDA)

#设置编译后的程序存放位置
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# 找到CUDA路径并导入头文件
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 设置C++版本
set(CMAKE_CXX_STANDARD 11)

# 检索出源文件与头文件
file(GLOB_RECURSE TEST_HEAD ${CMAKE_CURRENT_LIST_DIR}/src/.h ${CMAKE_CURRENT_LIST_DIR}/src/.cuh)
file(GLOB TEST_SRC1 ${CMAKE_CURRENT_LIST_DIR}/src/basicMmul.cu )
file(GLOB TEST_SRC2 ${CMAKE_CURRENT_LIST_DIR}/src/tiledMmul.cu )

# 设置CUAD编译配置
set(CUDA_NVCC_FLAGS -gencode arch=compute_80,code=sm_80;-G;-g)


# 生成可执行文件
cuda_add_executable(basic_matrix_multiplication ${TEST_SRC1} ${TEST_HEAD} OPTIONS ${GENCODE})
cuda_add_executable(tiled_matrix_multiplication ${TEST_SRC2} ${TEST_HEAD} OPTIONS ${GENCODE})

# >>> global configuration
set(PROFILING_TARGETS 
  basic_matrix_multiplication
  tiled_matrix_multiplication
)
foreach( profiling_target ${PROFILING_TARGETS} )
  target_link_libraries(${profiling_target} nvToolsExt -lcublas -lcurand)
  target_compile_features(${profiling_target} PUBLIC cxx_std_11)
  target_include_directories(${profiling_target} PUBLIC include)
  target_compile_options(
    ${profiling_target} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_80,code=sm_80>
  )
endforeach( profiling_target ${PROFILING_TARGETS} )