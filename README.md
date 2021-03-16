# DriveEasy Analytics python package

## Contributing
Refer to `https://github.com/FilippoBovo/production-data-science` for an excellent introduciton to workflow of data science projects.

## Setting up the development environment
1. Clone the repository:
   If you are on PARC internal network (have access to PARC gitlab-internal):
```
git clone https://gitlab-external.parc.com/iot-tower/driveeasy/driveeasy-analytics.git
```
   If not, please use the github repo:
   `git clone https://github.com/hyuparc/driveasy-analytics.git` 
   
If you want to save your username and password to avoid typing it everytime your push/pull:

You can use the git config to enable credentials storage in git.
```
git config --global credential.helper store
```
When running this command, the first time you pull or push from the remote repository, you'll get asked about the username and password.
Afterwards, for consequent communications with the remote repository you don't have to provide the username and password.


You can aslo clone with ssh to save the trouble of typing password everytime pushing to remote branch.
Refer to `https://support.atlassian.com/bitbucket-cloud/docs/set-up-an-ssh-key/` for setting up ssh key.
   ```bash
git clone git@gitlab-external.parc.com:iot-tower/driveeasy/driveeasy-analytics.git
   ```
If not used to git command line, `sourcetree` is a graphical git tool.

2. Install [Anaconda](https://docs.anaconda.com/anaconda/install/) or [Miniconda](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/index.html).

3. Open **Anaconda Prompt**, create Conda environment
On Windows, if you run 
```
C:\...\driveeasy-analytics>conda env create -f environment.yaml
```
from `Powershell` or `CMD`, it may pop up this error:
```
'conda' is not recognized as an internal or external command,
operable program or batch file.
```
If so, please create the Conda environment in **Anaconda Prompt**.

First, update `conda`:
```
conda update -n base -c defaults conda
```
Navigate to project root directory (`driveeasy-analytics`), then
```
conda env create --file environment.yml
```
Later on, if dependencies are updated in `environment.yml`, use the following command to update the `pydriveeasy` Conda environment:
   ```bash
   conda env update --file environment.yml
   ```

4. Recommend to use Spyder or PyCharm for code development. PyCharm community edition is sufficient. 
Configure PyCharm to use the created Conda environment. In terminal, list Conda information:
   ```bash
   conda info --envs
   base                  *  $HOME/miniconda3
   pydea                   $HOME/miniconda3/envs/pydea
   ```
   
   Run PyCharm, select "Open" from the main project dialog and pick the `driveeasy-analytics`  directory.
   
   In PyCharm, go to "Preferences...", "Project: driveeasy-analytics", "Project Interpreter", then click the small gear icon next to the dropdown and click "Add...". A new window will show up and select "Existing environment". In the "Existing environment" section, next to the dropdown click the "..." button and select: `$HOME/miniconda3/envs/pydriveeasy/bin/python`. This should be the path that was shown by the `conda info` command.

   > NOTE: The path may differ depending on whether Anaconda or Miniconda was installed.

5. Configure PyCharm to use `pytest`. In PyCharm, go to "Preferences...", "Tools", "Python Integrated Tools". Under "Testing", select `pytest` as the default test runner. 

6. Configure PyCharm to use Google docstring style. In PyCharm, go to "Preferences...", "Tools", "Python Integrated Tools". In docstring format, select "Google". After a function signature, type `"""` followed by `Enter` and PyCharm will automatically fill out your docstring template in Google style.

### Documentation 

In this project, we use the Google docstring style. All functions should be documented using docstrings.

#### Optional (not required to build the docs by yourself)
To build the latest API documentation, ensure that `sphinx` and `sphinx_rtd_theme` are installed in your conda environment.
From the project root, run `./docs/build.sh`. The documentation will be placed in `./docs/build/html` and can be viewed in your browser locally.

Note that any errors or warnings in the docstrings will cause the CI/CD pipeline to fail, so these must be corrected before any new merge requests can be merged.

### Directory structure

The `driveeasy` directory structure looks as follows.
```bash
driveeasy-analytics
├── apps
├── docs
├── explore
├── environment.yml
├── setup.py
├── pydea
└── tests
    └── 
```
`explore`: early stage scripts and notebooks for exploration. Its like a draft pad or playground. 

`docs`: Contains documentation-related documents, everything should be written in the source code and Sphinx will automatically create API documentation.

`environment.yml`: Configuration for the development environment depending on Conda.

`setup.py`: Configuration for distributing `pydea` as a library.

`tests`: All unit tests should be put in this directory.

