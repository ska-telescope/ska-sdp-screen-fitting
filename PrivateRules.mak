
# Due to incompatibility, some checks are disabled -> will be fixed in Jira ticket ST-1102
# E203 whitespace before ':', W503 line break before binary operator
PYTHON_SWITCHES_FOR_FLAKE8=--ignore=E203,W503


# List of disabled linting errors: to be enabled in future tickets

# [R0801(duplicate-code), ] Similar lines in 2 files
# [R0902(too-many-instance-attributes)] Too many instance attributes
# [R0912(too-many-branches), Soltab.set_selection] Too many branches (27/12)
# [R0913(too-many-arguments)] Too many arguments (11/5)
# [R0914(too-many-locals)] Too many local variables (18/15)
# [R0915(too-many-statements)] Too many statements (83/50)
# [R1702(too-many-nested-blocks)] Too many nested blocks (6/5)
# [R1721(unnecessary-comprehension)] Unnecessary use of a comprehension
# [R1725(super-with-arguments)] Consider using Python 3 style super() without arguments
# [R1728(consider-using-generator)] Consider using a generator instead 
# [W0102(dangerous-default-value)] Dangerous default value [] as argument
# [W0212(protected-access)] Access to a protected member _f_setattr of a client class
# [W0511(fixme), ] TODO: determine best-fit coefficients
# [W0601(global-variable-undefined)] Global variable 'SCREEN_PH' undefined at the module level
# [W0602(global-variable-not-assigned)] Using global for 'VAR_DICT' but no assignment is done
# [W0703(broad-except)] Catching too general exception Exception
# [W0622(redefined-builtin)] Redefining built-in 'filter'
# [W1505(deprecated-method)] Using deprecated method warn()
PYTHON_SWITCHES_FOR_PYLINT=--disable=R0801,R0902,R0912,R0913,R0914,R0915,R1702,R1721,R1725,R1728,W0102,W0212,W0511,W0601,W0602,W0622,W0703,W1505