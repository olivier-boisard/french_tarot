PYTHON_VIRTUAL_ENV_DIR=venv
DOCKER_HOME_DIR=/home/ReAgent
DOCKER_RUN_COMMAND="docker run --runtime=nvidia -v ${PWD}:${DOCKER_HOME_DIR} -w ${DOCKER_HOME_DIR}/ReAgent -p 0.0.0.0:6006:6006 -it horizon:dev"
export PYTHONPATH=${PWD}/src/package
VIRTUALENV_VALIDATION_SCRIPT=${PYTHON_VIRTUAL_ENV_DIR}/bin/activate

setup_python_venv:
	# Prepare python environment
	virtualenv --python=python3.7 $PYTHON_VIRTUAL_ENV_DIR}
	. ${VIRTUALENV_VALIDATION_SCRIPT}
	pip install -r requirements.txt

run_tests: config setup_python_venv venv/bin/activate
	# Run tests
	(cd src/tests && pytest --cov --cov-report=term-missing)

setup_reagent:
	# Install reagent environment
	git clone https://github.com/facebookresearch/ReAgent.git
	docker build -f ReAgent/docker/cuda.Dockerfile -t horizon:dev ReAgent/
	${DOCKER_RUN_COMMAND} ./scripts/setup.sh
	${DOCKER_RUN_COMMAND} mvn -f preprocessing/pom.xml clean package

