git clone -b 3.1.3 --depth 1 https://github.com/libjpeg-turbo/libjpeg-turbo.git

cmake -G Ninja -S libjpeg-turbo -B libjpeg-turbo/build -DENABLE_SHARED=OFF -DENABLE_STATIC=ON -DCMAKE_BUILD_TYPE=Release -DWITH_TURBOJPEG=OFF -DWITH_CRT_DLL=ON
cmake --build libjpeg-turbo/build --config Release --parallel
