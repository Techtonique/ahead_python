from subprocess import Popen, PIPE 

proc = Popen(["which", "R"], stdout=PIPE, stderr=PIPE)
R_IS_INSTALLED = proc.wait() == 0

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
This Python class is based on R package 'ahead' (https://techtonique.github.io/ahead/). 
You need to install R (https://www.r-project.org/) and rpy2 (https://pypi.org/project/rpy2/).

Then, install R package 'ahead' (if necessary): 
>> R -e 'install.packages("ahead", repos = https://techtonique.r-universe.dev)'    
"""

required_packages = ["ahead"]  # list of required R packages

packages_to_install = [
    x for x in required_packages if not rpackages.isinstalled(x)
]

base = importr("base")
utils = importr("utils")
if len(packages_to_install) > 0:            
    utils.install_packages(StrVector(packages_to_install), 
                           repos = "https://techtonique.r-universe.dev")  

# check R version
#print(f"R version: {utils.packageVersion('ahead')}") 

FLOATVECTOR = FloatVector
AHEAD_PACKAGE = importr("ahead")
CHECK_PACKAGES = True
