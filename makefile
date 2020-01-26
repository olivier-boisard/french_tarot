PYTHON_VIRTUAL_ENV_DIR=venv
BASE_DOCKER_IMAGE=horizon:dev
REAGENT_FOLDER=${PWD}/ReAgent
DOCKER_IMAGE=french_tarot:latest
DOCKER_RUN_COMMAND=docker run \
    --rm \
    --workdir=/opt/ReAgent \
    -p 0.0.0.0:6006:6006 \
    ${DOCKER_IMAGE}
VENV_BIN_FOLDER=${PWD}/venv/bin

export PYTHONPATH=${PWD}/src/package
VIRTUALENV_ACTIVATION_SCRIPT=${PYTHON_VIRTUAL_ENV_DIR}/bin/activate

build: french_tarot_built reagent_built test

test: french_tarot_built
	(cd src/tests && ${VENV_BIN_FOLDER}/pytest --cov --cov-report=term-missing)

reagent_built: reagent_built french_tarot_built
	docker build -f Dockerfile --build-arg USERNAME=$(shell whoami) --build-arg USERID=$(shell id -u) -t ${DOCKER_IMAGE} .
	${DOCKER_RUN_COMMAND} ./scripts/setup.sh
	${DOCKER_RUN_COMMAND} mvn -f preprocessing/pom.xml clean package
	touch reagent_built

french_tarot_built: ${VENV_BIN_FOLDER}/pip
	${VENV_BIN_FOLDER}/pip install -r requirements.txt
	touch french_tarot_built

${VENV_BIN_FOLDER}/pip:
	virtualenv --python=python3.7 ${PYTHON_VIRTUAL_ENV_DIR}
	. ${VIRTUALENV_ACTIVATION_SCRIPT}

reagent_built:
	docker build --build-arg USERNAME=$(shell whoami) --build-arg USERID=$(shell id -u) -t ${DOCKER_IMAGE} .
	touch reagent_built

clean:
	rm -rf venv reagent_built french_tarot_built
