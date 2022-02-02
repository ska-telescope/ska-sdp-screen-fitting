
# Due to incompatibility, some checks are disabled -> will be fixed in Jira ticket ST-1102
# E203 whitespace before ':', W503 line break before binary operator
PYTHON_SWITCHES_FOR_FLAKE8=--ignore=E203,W503

# Disable linting errors for W0613(unused-argument) and R0913(too-many-arguments). Implementation will be added in later MRs
PYTHON_SWITCHES_FOR_PYLINT=--disable=R0913,W0613