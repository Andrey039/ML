### Dockerfile with Ubuntu 18.04 and cuda 9.0
### Changes are indicated by CHANGED
### Everything else was copied together from the original Dockerfiles (as per comments)

### 1st part from https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/base/Dockerfile

FROM ubuntu:18.04
# CHANGED
#LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"
LABEL maintainer="tobycheese https://github.com/tobycheese/"

# CHANGED: below, add the two repos from 17.04 and 16.04 so all packages are found
RUN apt-get update && apt-get install -y --no-install-recommends gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64 /" >> /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" >> /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

### end 1st part from from https://gitlab.com/nvidia/cuda/blob/ubuntu18.04/10.0/base/Dockerfile

### 2nd part from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile

ENV CUDA_VERSION 9.0.176

ENV CUDA_PKG_VERSION 9-0=$CUDA_VERSION-1
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION && \
    ln -s cuda-9.0 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# CHANGED: commented out
# nvidia-docker 1.0
#LABEL com.nvidia.volumes.needed="nvidia_driver"
#LABEL com.nvidia.cuda.version="${CUDA_VERSION}"

RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=9.0"

### end 2nd part from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/base/Dockerfile

### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile

ENV NCCL_VERSION 2.3.7

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-$CUDA_PKG_VERSION \
        cuda-cublas-9-0=9.0.176.4-1 \
        libnccl2=$NCCL_VERSION-1+cuda9.0 && \
    apt-mark hold libnccl2 && \
    rm -rf /var/lib/apt/lists/*

### end all of from https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/runtime/Dockerfile

### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/Dockerfile

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDA_PKG_VERSION \
        cuda-nvml-dev-$CUDA_PKG_VERSION \
        cuda-minimal-build-$CUDA_PKG_VERSION \
        cuda-command-line-tools-$CUDA_PKG_VERSION \
        cuda-core-9-0=9.0.176.3-1 \
        cuda-cublas-dev-9-0=9.0.176.4-1 \
        libnccl-dev=$NCCL_VERSION-1+cuda9.0 && \
    rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs

### end all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/Dockerfile

### all of https://gitlab.com/nvidia/cuda/blob/ubuntu16.04/9.0/devel/cudnn7/Dockerfile

ENV CUDNN_VERSION 7.4.1.5
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
            libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


RUN adduser --disabled-password --gecos "" user

ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH


RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl unzip wget pkg-config \
    build-essential cmake gcc \
    libopenblas-dev

ENV TZ=Europe/Kaliningrad
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get install -y --no-install-recommends python3-dev python3-pip python3-tk python3-wheel && \
    pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev && \
    pip3 --no-cache-dir install Pillow
# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
    numpy scipy scikit-image pandas seaborn matplotlib Cython requests

RUN apt-get install -y --no-install-recommends \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev libv4l-dev \
    libxvidcore-dev libx264-dev \
    libgtk2.0-dev \
    libatlas-base-dev gfortran ffmpeg

RUN apt-get install -y --no-install-recommends qt5-default
RUN pip3 install --user pyqt5  
RUN apt-get install -y --no-install-recommends python3-pyqt5  
RUN apt-get install -y --no-install-recommends pyqt5-dev-tools
RUN apt-get install -y --no-install-recommends qttools5-dev-tools



# Get source from github
RUN git clone https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
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
    make install

#RUN apt-get install -y --no-install-recommends libopencv-*

RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir --upgrade pymongo pika numpy scipy flask

RUN pip3 install --no-cache-dir tqdm annoy pandas pymongo
#RUN pip3 install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.10.0-cp35-cp35m-linux_x86_64.whl
RUN pip3 install --upgrade tensorflow-gpu==1.8.0
RUN pip3 install --no-cache-dir keras==2.2.4 
RUN pip3 install --no-cache-dir --upgrade lxml



RUN apt-get install -y --no-install-recommends libsnappy-dev
RUN pip3 install --no-cache-dir python-snappy

RUN pip3 install --no-cache-dir keras-resnet

RUN apt-get clean && apt-get update && apt-get install -y locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

RUN pip3 install imgaug scikit-learn
RUN pip3 install --no-cache-dir --upgrade numpy

RUN pip3 install --no-cache-dir --upgrade pika boto3 kafka-python
RUN pip3 install jupyter
#ADD ./ /home/user
#COPY keras-retinanet /root

#RUN cd /root/tagger && python3 setup.py install
#RUN mkdir /root/.aws
#COPY credentials /root/.aws
WORKDIR /home/user/

#CMD ["jupyter", "notebook", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888"]
CMD  ["python3", "./mnt/person-tracker/consumer.py"]

