# Bicycle Model Racing Environment

- numpy == 1.21.6
- scipy == 1.9.0
- matplotlib == 3.5.2
- casadi == 3.5.5
- gym == 0.24.1
- [acados](https://docs.acados.org/)

## Run
```
python3 main.py
```

## How to Use
```
$ SOURCE LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"/home/<Username>/<acados_root>/lib"
$ SOURCE ACADOS_SOURCE_DIR="/home/<Username>/<acados_root>"
$ cd ~/car_racing_env && gedit main.py
 # set useDWA True or False
$ python3 main.py
```