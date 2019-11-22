PYTHON_VIRTUAL_ENV_DIR=venv
DOCKER_HOME_DIR=/home/ReAgent
BASE_DOCKER_IMAGE=horizon:dev
DOCKER_IMAGE=french_tarot:latest
DOCKER_RUN_COMMAND=docker run \
    --rm \
    --runtime=nvidia \
    -v ${PWD}:${PWD} \
    -w ${PWD}/ReAgent \
    -p 0.0.0.0:6006:6006 \
    -it \
    ${DOCKER_IMAGE}

export PYTHONPATH=${PWD}/src/package
VIRTUALENV_VALIDATION_SCRIPT=${PYTHON_VIRTUAL_ENV_DIR}/bin/activate

build: ReAgent/preprocessing/target/ run_tests

ReAgent/preprocessing/target/: ReAgent/
	docker build -f ReAgent/docker/cuda.Dockerfile -t ${BASE_DOCKER_IMAGE} ReAgent/
	docker build -f Dockerfile --build-arg USERID=$(shell id -u) --build-arg USERGROUP=$(shell id -g) -t ${DOCKER_IMAGE} .
	${DOCKER_RUN_COMMAND} ./scripts/setup.sh
	${DOCKER_RUN_COMMAND} mvn -f preprocessing/pom.xml clean package

run_tests: build_french_tarot
	(cd src/tests && pytest --cov --cov-report=term-missing)

build_french_tarot: venv/bin/pip
	venv/bin/pip install -r requirements.txt

venv/bin/pip:
	virtualenv --python=python3.7 ${PYTHON_VIRTUAL_ENV_DIR}
	. ${VIRTUALENV_VALIDATION_SCRIPT}

ReAgent/:
	git clone https://github.com/facebookresearch/ReAgent.git

clean:
	rm -rf venv ReAgent
