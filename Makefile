.PHONY: format lint tests

format:
	poetry run black .
	poetry run isort .

lint:
	poetry run mypy .
	poetry run black . --check
	poetry run isort . --check
	poetry run flake8 .

test: tests
tests:
	poetry run pytest

clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-tests:
	find . -name "*.yaml" -type f | xargs rm -f

release:
	test -z "$$(git status --porcelain)"
	poetry version patch
	git commit -am "Creating version v$$(poetry version -s)"
	git tag -a -m "Creating version v$$(poetry version -s)" "v$$(poetry version -s)"
	git push --follow-tags
	poetry publish --build --username $$PYPI_USERNAME --password $$PYPI_PASSWORD

all: test format lint
