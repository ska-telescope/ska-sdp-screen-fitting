# Include Python support
include .make/python.mk

# Due to incompatibility, some checks are disabled -> will be fixed in Jira ticket ST-1102
# E203 whitespace before ':', E501 line too long, W503 line break before binary operator
PYTHON_SWITCHES_FOR_FLAKE8=--ignore=E203,W503

