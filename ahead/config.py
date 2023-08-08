from subprocess import Popen, PIPE 

proc = Popen(["which", "R"], stdout=PIPE, stderr=PIPE)
R_IS_INSTALLED = proc.wait() == 0
#python_version = resources.require("MyProject")[0].version

try: 
    import rpy2.robjects.packages as rpackages    
    from rpy2.robjects.packages import importr    
    from rpy2.robjects.vectors import FloatVector, StrVector
    from rpy2 import rinterface, robjects
    from rpy2.rinterface_lib import callbacks
    from rpy2.rinterface_lib.embedded import RRuntimeError
except ImportError as e: 
    RPY2_ERROR_MESSAGE = str(e)
    RPY2_IS_INSTALLED = False
else: 
    RPY2_IS_INSTALLED = True

USAGE_MESSAGE = """
This Python class, BasicForecaster, is based on R package 'ahead' (https://techtonique.github.io/ahead/). 
You need to install R (https://www.r-project.org/) and rpy2 (https://pypi.org/project/rpy2/).

Then, install R package 'ahead' (if necessary): 
>> R -e 'install.packages("ahead", repos = https://techtonique.r-universe.dev)'    
"""

required_packages = ["ahead"]  # list of required R packages

# if all(rpackages.isinstalled(x) for x in required_packages):
#     print("AHEAD IS INSTALLED")
#     CHECK_PACKAGES = True  # True if packages are already installed
# else:
#     print("AHEAD IS NOT INSTALLED")
#     CHECK_PACKAGES = False  # False if packages are not installed

# if CHECK_PACKAGES is False:  # Not installed? Then install.

packages_to_install = [
    x for x in required_packages if not rpackages.isinstalled(x)
]

base = importr("base")
utils = importr("utils")
if len(packages_to_install) > 0:            
    utils.install_packages(StrVector(packages_to_install), 
                           repos = "https://techtonique.r-universe.dev")  

print(f"R version: {utils.packageVersion('ahead')}") 
#print(f"Python version: {python_version}")                       

FLOATVECTOR = FloatVector
AHEAD = importr("ahead")
CHECK_PACKAGES = True
