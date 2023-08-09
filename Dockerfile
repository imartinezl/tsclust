ARG VERSION=3
ARG NAME=tsclust
FROM python:$VERSION

# Install pip and poetry
RUN python -m pip install --upgrade pip

# Setup as minimal a stub project as posible, simply to allow caching base dependencies
# between builds.
#
# If error is encountered in these steps, can safely be removed locally.
RUN mkdir -p /$NAME/$NAME
RUN mkdir -p /$NAME/tests
RUN touch /isort/isort/__init__.py
RUN touch /isort/tests/__init__.py
RUN touch /isort/README.md
WORKDIR /isort
COPY pyproject.toml poetry.lock /isort/
RUN poetry install

# Install latest code for actual project
RUN rm -rf /isort
COPY . /isort
RUN poetry install

# Run full test suite
CMD /isort/scripts/test.sh