https://docs.google.com/document/d/1JQQmpCj-jLwnnD5rTZnnWhOc-gRT498DUApYpPyPv3o/edit?usp=sharing

Не забудьте скопировать config.py.default в config.py (он в .gitignore)

How to install on Ubuntu (tested using Ubuntu 13.10):
1. Copy config.py.default to config.py:
```bash
cp config.py.default config.py
```
2. Add parent directory of project to PYTHON_PATH environment variable.
To do so add the following line to yours .bashrc file:
```bash
export PYTHONPATH=$PYTHONPATH:/path_to_dir_where_project_dir_lives
```
3. Run in project root directory:
```bash
chmod +x ./install_ubuntu.sh
sudo ./install_ubuntu.sh
```
This runs script which installs all python and non-python dependencies and builds bindings
to asmlib-opencv library.
