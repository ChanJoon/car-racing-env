# Nonlinear MPC with Bezier curve referenece trajectory

## Dependencies
- acados (https://github.com/acados/acados.git)
  - CMake installation
  ```
  git clone https://github.com/acados/acados.git && cd acados
  git submodule update --recursive --init
  mkdir -p build && cd build
  cmake -DACADOS_WITH_OPENMP=ON -DACADOS_PYTHON=ON ..
  make install
  ```
  - Python interface
  ```
  pip3 install -e <acados_root>/interfaces/acados_template
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"<acados_root>/lib"
  export ACADOS_SOURCE_DIR="<acados_root>"
  ```
- numpy == 1.21.6
- scipy == 1.9.0
- matplotlib == 3.5.2
- casadi == 3.5.5
- gym == 0.24.1

## Run
```
python3 main.py
```
