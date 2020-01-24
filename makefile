PYTHON_VIRTUAL_ENV_DIR=venv
BASE_DOCKER_IMAGE=horizon:dev
REAGENT_FOLDER=${PWD}/ReAgent
DOCKER_IMAGE=french_tarot:latest
DOCKER_RUN_COMMAND=docker run \
    --rm \
    --runtime=nvidia \
    --volume=${REAGENT_FOLDER}:/home/ReAgent \
    --workdir=/home/ReAgent \
    -p 0.0.0.0:6006:6006 \
    ${DOCKER_IMAGE}
VENV_BIN_FOLDER=${PWD}/venv/bin

export PYTHONPATH=${PWD}/src/package
VIRTUALENV_VALIDATION_SCRIPT=${PYTHON_VIRTUAL_ENV_DIR}/bin/activate

build: build_french_tarot ReAgent/preprocessing/target/ test

ReAgent/preprocessing/target/: ReAgent/ build_french_tarot
	(cd ReAgent && docker build -f docker/cuda.Dockerfile -t horizon:dev .)
	docker build -f Dockerfile --build-arg USERNAME=$(shell whoami) --build-arg USERID=$(shell id -u) -t ${DOCKER_IMAGE} .
	${DOCKER_RUN_COMMAND} ./scripts/setup.sh
	${DOCKER_RUN_COMMAND} mvn -f preprocessing/pom.xml clean package

test: build_french_tarot
	(cd src/tests && ${VENV_BIN_FOLDER}/pytest --cov --cov-report=term-missing)

build_french_tarot: ${VENV_BIN_FOLDER}/pip
	${VENV_BIN_FOLDER}/pip install -r requirements.txt

${VENV_BIN_FOLDER}/pip:
	virtualenv --python=python3.7 ${PYTHON_VIRTUAL_ENV_DIR}
	. ${VIRTUALENV_VALIDATION_SCRIPT}

ReAgent/:
	git clone https://github.com/facebookresearch/ReAgent.git
	docker build -f ReAgent/docker/cuda.Dockerfile -t ${BASE_DOCKER_IMAGE} ReAgent/

clean:
	rm -rf venv ReAgent
