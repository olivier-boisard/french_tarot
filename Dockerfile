FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Input parameters
ARG USERNAME
ARG USERID

# Parameters
ENV MINICONDA_SETUP_FILE /tmp/miniconda.sh
ENV SPARK_TGZ /tmp/spark.tgz
ENV LIBTORCH_ZIP /tmp/libtorch.zip
ENV WGET_COMMAND "wget -q"
ENV INSTALL_DIR /opt

# Prepare linux environment
RUN apt-get update && apt-get install -y wget git unzip

# Install miniconda
RUN ${WGET_COMMAND} https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ${MINICONDA_SETUP_FILE}
RUN chmod u+x ${MINICONDA_SETUP_FILE}
RUN ${MINICONDA_SETUP_FILE} -b -p $INSTALL_DIR/miniconda
ENV PATH "$PATH:$INSTALL_DIR/miniconda/bin/"
RUN conda config --add channels conda-forge
RUN conda config --add channels pytorch

# Download ReAgent
WORKDIR $INSTALL_DIR
# An error might happen with the next command saying "Direct fetching of that commit failed". It can be ignored.
RUN git clone --recurse-submodules https://github.com/facebookresearch/ReAgent.git; exit 0
ENV REAGENT_HOME $INSTALL_DIR/ReAgent
WORKDIR $REAGENT_HOME
RUN git checkout b000663598c42a87d000cf18902cb1c18d6a2c86
RUN git submodule update --force --recursive --init --remote

# Install required python packages
RUN conda install -y --file requirements.txt

# Install spark
ENV JAVA_HOME $INSTALL_DIR/miniconda
RUN ${WGET_COMMAND} https://archive.apache.org/dist/spark/spark-2.3.3/spark-2.3.3-bin-hadoop2.7.tgz -O ${SPARK_TGZ}
RUN tar xvzf ${SPARK_TGZ} -C /usr/local/bin/
RUN mv /usr/local/bin/spark* /usr/local/spark
ENV PATH "$PATH:/usr/local/spark/bin"

# Install libtorch
RUN ${WGET_COMMAND} https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.4.0.zip -O ${LIBTORCH_ZIP}
RUN unzip ${LIBTORCH_ZIP} -d $INSTALL_DIR

# Build reagent
RUN mkdir $REAGENT_HOME/serving/build
WORKDIR $REAGENT_HOME/serving/build
RUN cmake -DCMAKE_PREFIX_PATH=$INSTALL_DIR/libtorch ..
RUN make -j

# Build python package
WORKDIR $REAGENT_HOME
RUN pip install -e .
RUN pip install "gym[classic_control,box2d,atari]"
RUN pytest -n$(cat /proc/cpuinfo | grep processor | wc -l)

# Build preprocessing package
RUN mvn -f preprocessing/pom.xml clean package

# Setup user to avoid running reagent as root
RUN useradd --create-home --user-group --uid $USERID $USERNAME
RUN chown $USERNAME:$USERNAME $INSTALL_DIR/miniconda/lib/python3.7/site-packages/
RUN chown $REAGENT_HOME
USER $USERNAME
