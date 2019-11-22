PYTHON_VIRTUAL_ENV_DIR=venv
DOCKER_HOME_DIR=/home/ReAgent
DOCKER_RUN_COMMAND=docker run --runtime=nvidia -v "$PWD":$DOCKER_HOME_DIR -w $DOCKER_HOME_DIR/ReAgent -p 0.0.0.0:6006:6006 -it horizon:dev

export PYTHONPATH=$PYTHONPATH:$PWD/src/package

virtualenv_activation_script=$PYTHON_VIRTUAL_ENV_DIR/bin/activate

# Prepare python environment
virtualenv --python=python3.7 $PYTHON_VIRTUAL_ENV_DIR
# shellcheck source=$virtualenv_activation_script
source $virtualenv_activation_script
pip install -r requirements.txt

# Install reagent environment
git clone https://github.com/facebookresearch/ReAgent.git
docker build -f ReAgent/docker/cuda.Dockerfile -t horizon:dev ReAgent/
$DOCKER_RUN_COMMAND ./scripts/setup.sh
$DOCKER_RUN_COMMAND mvn -f preprocessing/pom.xml clean package

# Run tests
pytest --cov --cov-report=term-missing
