#!/usr/bin/bash

##### android armv7
mkdir -p build-android-armv7
pushd build-android-armv7
cmake -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_ABI="armeabi-v7a with NEON" -DANDROID_NATIVE_API_LEVEL=android-9 -DANDROID_FORCE_ARM_BUILD=OFF -DANDROID_STL_FORCE_FEATURES=OFF ..
make
make install
popd

##### android aarch64
mkdir -p build-android-aarch64
pushd build-android-aarch64
cmake -DCMAKE_TOOLCHAIN_FILE=../android.toolchain.cmake -DANDROID_ABI="arm64-v8a" -DANDROID_NATIVE_API_LEVEL=android-21 -DANDROID_FORCE_ARM_BUILD=OFF -DANDROID_STL_FORCE_FEATURES=OFF ..
make
make install
popd
