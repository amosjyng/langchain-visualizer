.PHONY: format lint tests

all: format lint test

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
	poetry run pytest -v -k 'not network'

tests-ci:
	poetry run pytest -v

clean:
	find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

clean-tests:
	find . -name "*.yaml" -type f | xargs rm -f

release:
	test -z "$$(git status --porcelain)"
	git checkout main
	git pull
	poetry version patch
	git checkout -b "release/v$$(poetry version -s)"
	git commit -am "Releasing version v$$(poetry version -s)"
	git tag -a -m "Releasing version v$$(poetry version -s)" "v$$(poetry version -s)"
	poetry publish --build --username $$PYPI_USERNAME --password $$PYPI_PASSWORD
# git push at the very end to get Github PR link
	git push --set-upstream origin "release/v$$(poetry version -s)"
# --follow-tags seems to suppress Github message output
	git push --follow-tags
