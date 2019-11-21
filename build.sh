PYTHON_VIRTUAL_ENV_DIR=venv
export PYTHONPATH=$PYTHONPATH:$PWD/src/package

virtualenv_activation_script=$PYTHON_VIRTUAL_ENV_DIR/bin/activate

# Prepare python environment
virtualenv --python=python3.7 $PYTHON_VIRTUAL_ENV_DIR
# shellcheck source=$virtualenv_activation_script
source $virtualenv_activation_script
pip install -r requirements.txt

# Install and test ReAgent
git clone https://github.com/facebookresearch/ReAgent.git
docker build -f ReAgent/docker/cuda.Dockerfile -t horizon:dev ReAgent/

# Run tests
pytest --cov --cov-report=term-missing
