# Include Python support
include .make/python.mk

# E203 whitespace before ':', E501 line too long, W503 line break before binary operator
PYTHON_SWITCHES_FOR_FLAKE8=--ignore=E203,E501,W503

