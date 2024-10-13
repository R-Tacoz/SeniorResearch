# USING PYTHON 3.11.4 
Export environment using "conda env export --no-builds | findstr /V "prefix" > environment.yml"
Import environment using "conda env import -f environment.yml"
Update environment using "conda env update -f environment.yml"