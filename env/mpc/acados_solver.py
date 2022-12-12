from operator import ge
from os.path import dirname, realpath
from acados_template import AcadosOcpSolver



def acados_solver(ocp, config):

    # set QP solver and integration
    ocp.dims.N = config.mpc.N
    ocp.solver_options.tf = config.mpc.N * config.mpc.dt
    ocp.solver_options.qp_solver = config.mpc.qp_solver
    ocp.solver_options.qp_solver_iter_max = config.mpc.qp_solver_iter_max
    ocp.solver_options.qp_solver_warm_start = config.mpc.qp_solver_warm_start
    ocp.solver_options.hpipm_mode = config.mpc.hpipm_mode
    ocp.solver_options.nlp_solver_type = config.mpc.nlp_solver_type
    ocp.solver_options.integrator_type = config.mpc.integrator_type
    ocp.solver_options.hessian_approx = config.mpc.hessian_approx
    ocp.solver_options.sim_method_num_stages = config.mpc.sim_method_num_stages
    ocp.solver_options.sim_method_num_steps  = config.mpc.sim_method_num_steps
    ocp.solver_options.nlp_solver_max_iter = config.mpc.nlp_solver_max_iter
    ocp.solver_options.tol = config.mpc.tol
    ocp.solver_options.print_level = config.mpc.print_level

    export_dir = dirname(realpath(__file__))
    ocp.code_export_directory = export_dir+"/c_generated_code"
    solver = AcadosOcpSolver(ocp, json_file=export_dir+"/acados_ocp.json")

    return solver