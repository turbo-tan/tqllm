# This backend uses only tqLLM sources, POSIX/NDK APIs and Arm NEON.
find_package(Threads REQUIRED)
add_library(tqandroid STATIC
  src/backends/android/android_model.cpp src/core/mapped_file.cpp
  src/core/gguf.cpp src/core/tokenizer.cpp src/formats/weight/dequant.cpp)
target_include_directories(tqandroid PUBLIC include src/backends/android PRIVATE
  src/core src/kv src/formats/weight)
target_link_libraries(tqandroid PRIVATE Threads::Threads)
target_compile_options(tqandroid PRIVATE -Wall -Wextra -Wpedantic)
add_executable(tqandroid-run tools/tqandroid_run.cpp)
target_link_libraries(tqandroid-run PRIVATE tqandroid)
install(TARGETS tqandroid-run RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
if(BUILD_TESTING)
  add_executable(test-android-native tests/android_native.cpp)
  target_include_directories(test-android-native PRIVATE src/core src/kv src/formats/weight)
  target_link_libraries(test-android-native PRIVATE tqandroid Threads::Threads)
  add_test(NAME android_native COMMAND test-android-native)
  add_executable(test-android-packed tests/android_packed.cpp)
  target_include_directories(test-android-packed PRIVATE src/core src/kv src/formats/weight)
  target_link_libraries(test-android-packed PRIVATE tqandroid Threads::Threads)
  add_test(NAME android_packed COMMAND test-android-packed)
  add_executable(test-android-state tests/android_state.cpp)
  target_link_libraries(test-android-state PRIVATE tqandroid)
  if(NOT CMAKE_CROSSCOMPILING)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_test(NAME android_reference COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_android_reference.py
      --runner $<TARGET_FILE:tqandroid-run> --state $<TARGET_FILE:test-android-state>)
  endif()
endif()
