cmake_minimum_required(VERSION 3.20)
project(mlsys_dag_scheduler_mlir C CXX)

# Configure with: cmake -B build-mlir -G Ninja -DCMAKE_PREFIX_PATH=$(brew --prefix llvm)
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

set(CMAKE_CXX_STANDARD 20)
add_compile_options(-fexceptions -frtti)
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/mlir/include ${CMAKE_CURRENT_BINARY_DIR})
separate_arguments(LLVM_DEFS UNIX_COMMAND "${LLVM_DEFINITIONS}")
add_definitions(${LLVM_DEFS})

set(LLVM_TARGET_DEFINITIONS mlir/include/Sched/SchedOps.td)
mlir_tablegen(SchedOps.h.inc -gen-op-decls)
mlir_tablegen(SchedOps.cpp.inc -gen-op-defs)
mlir_tablegen(SchedDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SchedDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(SchedOpsIncGen)

# Original (non-MLIR) problem/solution/evaluator code, reused as the cost model.
add_library(mlsys_core STATIC source/problem.cpp source/solution.cpp source/evaluator.cpp)
target_include_directories(mlsys_core PRIVATE third_party)

add_library(SchedDialect STATIC mlir/lib/Sched.cpp mlir/lib/Convert.cpp
            mlir/lib/Cost.cpp mlir/lib/Passes.cpp)
add_dependencies(SchedDialect SchedOpsIncGen)
target_link_libraries(SchedDialect PUBLIC mlsys_core MLIRIR MLIRFuncDialect MLIRPass MLIRParser)

add_executable(mlsys-mlir mlir/tools/mlsys-mlir.cpp)
target_link_libraries(mlsys-mlir PRIVATE SchedDialect)
add_executable(sched-opt mlir/tools/sched-opt.cpp)
target_link_libraries(sched-opt PRIVATE SchedDialect MLIROptLib)
