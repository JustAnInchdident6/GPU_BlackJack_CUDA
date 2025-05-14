# CMake minimum version. 3.18+ has good CUDA support.
cmake_minimum_required(VERSION 3.18)

# Project Name and Languages
# Since .cu files are typically compiled as C++ by nvcc,
# and your main() function will be in the .cu file.
project(BlackjackCudaProject LANGUAGES CXX CUDA)

# Set C++ Standard (for the C++ parts of .cu files and any .cpp files)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Set CUDA Standard (often matches C++ standard)
set(CMAKE_CUDA_STANDARD 17) # nvcc compiles .cu as C++ by default
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# Specify CUDA Architectures you are targeting.
# For RTX 4060, compute capability 8.9 is correct.
set(CMAKE_CUDA_ARCHITECTURES 89)

# Enable Separable Compilation and Relocatable Device Code (RDC)
# Good practice for larger projects, may not be strictly needed if all
# device code and kernel calls are within the single .cu file.
# However, it doesn't hurt to leave it on.
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# --- Define your executable ---
# List ALL source files that make up your executable.
# Since you are only using blackjack_cuda.cu, it's the only one listed.
add_executable(blackjack_cuda
        blackjack_cuda.cu       # Your CUDA code (which includes main())
)

# --- Configuration-Specific Compile Options for CUDA (nvcc) ---
# These options are for NVCC (CUDA Compiler) for DEVICE code:
target_compile_options(blackjack_cuda PRIVATE
        # For Debug builds, pass -G to nvcc for device debug symbols and less optimization.
        $<$<CONFIG:DEBUG>:-G>

        # For Release builds, pass -O3 to nvcc for device code optimization.
        $<$<CONFIG:RELEASE>:-O3>
)

# HOST Compiler Flags (for cl.exe, the MSVC compiler, used for host part of .cu):
# CMake automatically sets appropriate flags for the host compiler (cl.exe)
# based on the build type (CMAKE_BUILD_TYPE e.g., Debug or Release).
# For MSVC (Visual Studio):
# - Debug builds (CMAKE_BUILD_TYPE=Debug):
#   CMake typically sets CMAKE_CXX_FLAGS_DEBUG
#   to something like "/MDd /Zi /Ob0 /Od /RTC1".
#   NVCC uses these for the host compilation part of .cu files via -Xcompiler.
#
# - Release builds (CMAKE_BUILD_TYPE=Release):
#   CMake typically sets CMAKE_CXX_FLAGS_RELEASE
#   to something like "/MD /O2 /Ob2 /DNDEBUG".

# --- Linking ---
# Linking to CUDA libraries (like cudart, cudadevrt) is typically handled
# automatically by CMake when CUDA is specified as a language and
# CMAKE_CUDA_SEPARABLE_COMPILATION is ON.

# If you have other libraries to link, you would add them here:
# target_link_libraries(blackjack_cuda PRIVATE SomeOtherLib)

# Ensure compile_commands.json is generated for IDEs like CLion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
