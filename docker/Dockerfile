# Get base environment
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ARG UID
ARG GID
ARG USERNAME

# Set environment variables
ENV CONDA_DIR /opt/conda
ENV CONDA_VERSION 2019.03
ENV PYCHARM_VERSION 2019.2

# Setup prerequisites
RUN apt-get -y update
RUN apt-get -y upgrade

# Root-level installations
RUN apt-get install -y wget bzip2 build-essential sudo python-pyqt5 git curl \
    unzip software-properties-common default-jre

# Prepare custom user
RUN groupadd $USERNAME -g $GID
RUN useradd -l -m -s /bin/bash -u $UID -g $GID $USERNAME
RUN usermod -aG sudo $USERNAME
RUN echo "$USERNAME ALL=(ALL:ALL) NOPASSWD: ALL" > /etc/sudoers

# Install Pycharm
WORKDIR /opt
RUN wget https://download.jetbrains.com/python/pycharm-community-$PYCHARM_VERSION.tar.gz -q -O pycharm.tar.gz
RUN tar xzf pycharm.tar.gz
RUN rm pycharm.tar.gz

# Configure and switch to local user
RUN mkdir $CONDA_DIR && chown -R $UID:$GID $CONDA_DIR
ENV USERHOME /home/$USERNAME
RUN chown -R $UID:$GID $USERHOME
RUN chmod a+rw -R /tmp
USER $USERNAME

# Prepare python environment
ENV CONDA_INSTALLATION_SCRIPT /tmp/conda.sh
RUN wget https://repo.anaconda.com/archive/Anaconda3-$CONDA_VERSION-Linux-x86_64.sh -q -O $CONDA_INSTALLATION_SCRIPT
RUN bash $CONDA_INSTALLATION_SCRIPT -b -u -p $CONDA_DIR
RUN rm $CONDA_INSTALLATION_SCRIPT
ENV PATH $CONDA_DIR/bin:$PATH
RUN pip install --upgrade pip

# Install python tool
RUN pip install dill==0.2.9 pytest==4.6.3 pytest-mock==1.10.4 pytest-repeat==0.8.0 pytest-cov==2.7.1

# Install pytorch
RUN conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

# Install NVIDIA Apex
WORKDIR /tmp
RUN git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
WORKDIR /tmp
RUN rm -rf /tmp/apex
