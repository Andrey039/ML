### Dockerfile with Ubuntu 18.04 and cuda 9.0
### Changes are indicated by CHANGED
### Everything else was copied together from the original Dockerfiles (as per comments)

### 1st part from https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/base/Dockerfile

FROM ubuntu:18.04
# CHANGED
#LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL maintainer="tobycheese https://github.com/tobycheese/"
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"
ENV CUDA_VERSION 9.0.176
ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"
ENV NCCL_VERSION 2.3.7
ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
ENV CUDNN_VERSION 7.4.1.5
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

ENV TZ=Europe/Kaliningrad


# CHANGED: below, add the two repos from 17.04 and 16.04 so all packages are found
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64 /" >> /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" >> /etc/apt/sources.list.d/nvidia-ml.list && \
### end 1st part from from https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/base/Dockerfile
### 2nd part from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile
    apt-get update && apt-get install  -y --no-install-recommends \
            cuda-cudart-$CUDA_PKG_VERSION && \
            ln -s cuda-9.0 /usr/local/cuda && \ 
# CHANGED: commented out
# nvidia-docker 1.0
#LABEL com.nvidia.volumes.needed="nvidia_driver"
#LABEL com.nvidia.cuda.version="${CUDA_VERSION}"
    echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf && \
### end 2nd part from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile
### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile
    apt-get install -y --no-install-recommends \
            cuda-libraries-$CUDA_PKG_VERSION \
            cuda-cublas-9-0=9.0.176.4-1 \
            libnccl2=$NCCL_VERSION-1+cuda9.0 && \
            apt-mark hold libnccl2 \
### end all of from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile
### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/Dockerfile
            cuda-libraries-dev-$CUDA_PKG_VERSION \
            cuda-nvml-dev-$CUDA_PKG_VERSION \
            cuda-minimal-build-$CUDA_PKG_VERSION \
            cuda-command-line-tools-$CUDA_PKG_VERSION \
            cuda-core-9-0=9.0.176.3-1 \
            cuda-cublas-dev-9-0=9.0.176.4-1 \
            libnccl-dev=$NCCL_VERSION-1+cuda9.0 \
### end all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/Dockerfile
### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/cudnn7/Dockerfile
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    adduser --disabled-password --gecos "" user && \
    apt-get install -y --no-install-recommends \
            apt-utils git  unzip wget pkg-config \
            build-essential cmake gcc \
            libopenblas-dev && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone




RUN apt-get install -y --no-install-recommends python3-dev python3-pip python3-tk python3-wheel && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
#Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow && \
# Science libraries and other common packages
    pip3 --no-cache-dir install \
        numpy scipy scikit-image pandas seaborn matplotlib Cython requests && \
    apt-get install -y --no-install-recommends \
            libjpeg-dev libpng-dev libtiff-dev \
            libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
            libxvidcore-dev libx264-dev \
            libgtk2.0-dev \
            libatlas-base-dev gfortran ffmpeg && \
    apt-get install -y --no-install-recommends qt5-default && \
    pip3 install --user pyqt5  && \
    apt-get install -y --no-install-recommends python3-pyqt5  && \
    apt-get install -y --no-install-recommends pyqt5-dev-tools && \
    apt-get install -y --no-install-recommends qttools5-dev-tools



# Get source from github
RUN git clone https://github.com/opencv/opencv.git /usr/local/src/opencv && \
# Compile
    cd /usr/local/src/opencv && mkdir build && cd build && \
    cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D BUILD_TESTS=OFF \
          -D BUILD_PERF_TESTS=OFF \
          -D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
          -D WITH_FFMPEG=1 \
          #-D WITH_QT=ON \
          -D WITH_GTK=ON \
          -D WITH_GTK_2_X=ON \
          .. && \
    make -j"$(nproc)" && \
    make install && \
    rm -rf /usr/local/src/opencv


#RUN apt-get install -y --no-install-recommends libopencv-*
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN apt-get install -y --no-install-recommends libsnappy-dev locales && \
    locale-gen en_US.UTF-8 && \
    pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir --upgrade pymongo pika numpy scipy flask && \
    pip3 install --no-cache-dir tqdm annoy pandas pymongo && \
    pip3 install --no-cache-dir --upgrade tensorflow-gpu==1.8.0 && \
    pip3 install --no-cache-dir keras==2.2.4  && \
    pip3 install --no-cache-dir --upgrade lxml && \
    
    pip3 install --no-cache-dir python-snappy && \
    pip3 install --no-cache-dir keras-resnet && \
    
    pip3 install --no-cache-dir imgaug scikit-learn && \
    pip3 install --no-cache-dir --upgrade numpy && \
    pip3 install --no-cache-dir --upgrade pika boto3 kafka-python && \
    pip3 install --no-cache-dir jupyter
#ADD ./ /home/user
#COPY keras-retinanet /root

#RUN cd /root/tagger && python3 setup.py install
#RUN mkdir /root/.aws
#COPY credentials /root/.aws
WORKDIR /home/user/


RUN rm -rf /var/lib/apt/lists/*

#CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888"]
CMD  ["python3", "./mnt/person-tracker/consumer.py"]
