# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.6

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build

# Include any dependencies generated for this target.
include CMakeFiles/cpu.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cpu.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cpu.dir/flags.make

CMakeFiles/cpu.dir/test_module.cpp.o: CMakeFiles/cpu.dir/flags.make
CMakeFiles/cpu.dir/test_module.cpp.o: ../test_module.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cpu.dir/test_module.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpu.dir/test_module.cpp.o -c /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/test_module.cpp

CMakeFiles/cpu.dir/test_module.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpu.dir/test_module.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/test_module.cpp > CMakeFiles/cpu.dir/test_module.cpp.i

CMakeFiles/cpu.dir/test_module.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpu.dir/test_module.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/test_module.cpp -o CMakeFiles/cpu.dir/test_module.cpp.s

CMakeFiles/cpu.dir/test_module.cpp.o.requires:

.PHONY : CMakeFiles/cpu.dir/test_module.cpp.o.requires

CMakeFiles/cpu.dir/test_module.cpp.o.provides: CMakeFiles/cpu.dir/test_module.cpp.o.requires
	$(MAKE) -f CMakeFiles/cpu.dir/build.make CMakeFiles/cpu.dir/test_module.cpp.o.provides.build
.PHONY : CMakeFiles/cpu.dir/test_module.cpp.o.provides

CMakeFiles/cpu.dir/test_module.cpp.o.provides.build: CMakeFiles/cpu.dir/test_module.cpp.o


CMakeFiles/cpu.dir/test_identity_module.cpp.o: CMakeFiles/cpu.dir/flags.make
CMakeFiles/cpu.dir/test_identity_module.cpp.o: ../test_identity_module.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/cpu.dir/test_identity_module.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cpu.dir/test_identity_module.cpp.o -c /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/test_identity_module.cpp

CMakeFiles/cpu.dir/test_identity_module.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cpu.dir/test_identity_module.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/test_identity_module.cpp > CMakeFiles/cpu.dir/test_identity_module.cpp.i

CMakeFiles/cpu.dir/test_identity_module.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cpu.dir/test_identity_module.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/test_identity_module.cpp -o CMakeFiles/cpu.dir/test_identity_module.cpp.s

CMakeFiles/cpu.dir/test_identity_module.cpp.o.requires:

.PHONY : CMakeFiles/cpu.dir/test_identity_module.cpp.o.requires

CMakeFiles/cpu.dir/test_identity_module.cpp.o.provides: CMakeFiles/cpu.dir/test_identity_module.cpp.o.requires
	$(MAKE) -f CMakeFiles/cpu.dir/build.make CMakeFiles/cpu.dir/test_identity_module.cpp.o.provides.build
.PHONY : CMakeFiles/cpu.dir/test_identity_module.cpp.o.provides

CMakeFiles/cpu.dir/test_identity_module.cpp.o.provides.build: CMakeFiles/cpu.dir/test_identity_module.cpp.o


# Object files for target cpu
cpu_OBJECTS = \
"CMakeFiles/cpu.dir/test_module.cpp.o" \
"CMakeFiles/cpu.dir/test_identity_module.cpp.o"

# External object files for target cpu
cpu_EXTERNAL_OBJECTS =

libcpu.so: CMakeFiles/cpu.dir/test_module.cpp.o
libcpu.so: CMakeFiles/cpu.dir/test_identity_module.cpp.o
libcpu.so: CMakeFiles/cpu.dir/build.make
libcpu.so: CMakeFiles/cpu.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX shared library libcpu.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cpu.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cpu.dir/build: libcpu.so

.PHONY : CMakeFiles/cpu.dir/build

CMakeFiles/cpu.dir/requires: CMakeFiles/cpu.dir/test_module.cpp.o.requires
CMakeFiles/cpu.dir/requires: CMakeFiles/cpu.dir/test_identity_module.cpp.o.requires

.PHONY : CMakeFiles/cpu.dir/requires

CMakeFiles/cpu.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cpu.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cpu.dir/clean

CMakeFiles/cpu.dir/depend:
	cd /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build /BS/dstutz/work/torch/torch-c-article/lib/cpp/cpu/build/CMakeFiles/cpu.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cpu.dir/depend

