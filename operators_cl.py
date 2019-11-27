import bempp.api 
import numpy as np
from scipy.sparse import coo_matrix
import time
import scipy.linalg


def rescale(A, d1, d2):
    """Rescale the 2x2 block operator matrix A"""
    
    A[0, 1] = A[0, 1] * (d2 / d1)
    A[1, 0] = A[1, 0] * (d1 / d2)
    
    return A

def rotate_x(x,y,z,theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R_x = np.array([[1,0,0],[0,c, -s], [0,s, c]])
    v = np.array([x,y,z])
    rotated_v = R_x.dot(v)
    return rotated_v

def rotate_y(x,y,z,theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    v = np.array([x,y,z])
    rotated_v = R_y.dot(v)
    return rotated_v

def rotate_z(x,y,z,theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    v = np.array([x,y,z])
    rotated_v = R_z.dot(v)
    return rotated_v

def translation(x,y,z,d):
    x += d[0]
    y += d[1]
    z += d[2]
    return (x,y,z)

def PMCHWT_operator(grids, k_ext, k_int, mu_ext, mu_int):

    """Creates the PMCHWT operator Ae+Ai for single and multi-particle problesm"""
    from bempp.api.assembly.blocked_operator import BlockedOperator
    from bempp.api.operators.boundary.maxwell import magnetic_field, electric_field
    from bempp.api.operators.boundary.sparse import identity

    number_of_scatterers = len(grids)
    interior_operators = []
    exterior_operators = []
    identity_operators = []

    rwg_space = [bempp.api.function_space(grid, 'RWG', 0) for grid in grids]
    snc_space = [bempp.api.function_space(grid, 'SNC', 0) for grid in grids]

    for i in range(number_of_scatterers):
        Ai = BlockedOperator(2,2)
        Ae = BlockedOperator(2,2)
        op_ident = BlockedOperator(2,2)
        magnetic_ext = magnetic_field(domain = rwg_space[i], range_ = rwg_space[i],dual_to_range = snc_space[i], wavenumber = k_ext)
        electric_ext = electric_field(domain = rwg_space[i], range_ = rwg_space[i],dual_to_range = snc_space[i], wavenumber = k_ext)

        magnetic_int = magnetic_field(domain = rwg_space[i], range_ = rwg_space[i],dual_to_range = snc_space[i], wavenumber = k_int[i])
        electric_int = electric_field(domain = rwg_space[i], range_ = rwg_space[i],dual_to_range = snc_space[i], wavenumber = k_int[i])

        op_identity = identity(rwg_space[i], rwg_space[i], snc_space[i])
        op_ident[0,0] = op_identity
        op_ident[1,1] = op_identity
        identity_operators.append(op_ident)

        Ai[0,0] = magnetic_int
        Ai[0,1] = electric_int
        Ai[1,0] = -1 * electric_int
        Ai[1,1] = magnetic_int

        Ae[0,0] = magnetic_ext
        Ae[0,1] = electric_ext
        Ae[1,0] = -1 * electric_ext
        Ae[1,1] = magnetic_ext

        Ai = rescale(Ai, k_int[i], mu_int[i])
        interior_operators.append(Ai)

        Ae = rescale(Ae, k_ext, mu_ext)
        exterior_operators.append(Ae)

    filter_operators = []
    transfer_operators = np.empty((number_of_scatterers, number_of_scatterers), dtype = np.object)
    op = np.empty((number_of_scatterers, number_of_scatterers), dtype = np.object)
    for i in range(number_of_scatterers):
        filter_operators.append(0.5 * identity_operators[i] - interior_operators[i])
        for j in range(number_of_scatterers):
            if i == j:
                op[i,j] = interior_operators[j] + exterior_operators[j]
            else:
                Aij = BlockedOperator(2,2)
                magnetic_ij = magnetic_field(domain = rwg_space[j], range_=rwg_space[i],dual_to_range=snc_space[i], wavenumber = k_ext)
                electric_ij = electric_field(domain = rwg_space[j], range_=rwg_space[i],dual_to_range=snc_space[i], wavenumber = k_ext)

                Aij[0,0] = magnetic_ij
                Aij[0,1] = electric_ij
                Aij[1,0] = -1 * electric_ij
                Aij[0,1] = magnetic_ij

                transfer_operators[i,j] = Aij
                op[i,j] = rescale(Aij, k_ext, mu_ext)

    result = bempp.api.assembly.blocked_operator.GeneralizedBlockedOperator(op)

    return [result, filter_operators]

'''
def PMCHWT_operator(grids, k_ext, k_int, mu_ext, mu_int):
    """Creates the PMCHWT operator Ae+Ai for single and multi-particle problems"""
    number_of_scatterers = len(grids)
        
    interior_operators = []
    exterior_operators = []
    identity_operators = []
    
    for i in range(number_of_scatterers):
        Ai = bempp.api.operators.boundary.maxwell.multitrace_operator(grid = grids[i], wavenumber = k_int[i],
								     target = None, space_type = 'all_rwg')
        Ae = bempp.api.operators.boundary.maxwell.multitrace_operator(grid = grids[i], wavenumber = k_ext,
                                                                      target = None, space_type = 'all_rwg')
        Ai = rescale(Ai, k_int[i], mu_int[i])
        Ae = rescale(Ae, k_ext, mu_ext)

        op_ident = bempp.api.assembly.blocked_operator.BlockedOperator(2,2)
        domain0, domain1 = Ai.domain_spaces
        dual_to_range0, dual_to_range1 = Ai.dual_to_range_spaces
        range0, range1 = Ai.range_spaces
        op_identity0 = bempp.api.operators.boundary.sparse.identity(domain0,range0,dual_to_range0)
        op_identity1 = bempp.api.operators.boundary.sparse.identity(domain1, range1, dual_to_range1)
        op_ident[0,0] = op_identity0
        op_ident[1,1] = op_identity1
        identity_operators.append(op_ident)

        interior_operators.append(Ai)
        exterior_operators.append(Ae)

    filter_operators = []
    transfer_operators = np.empty((number_of_scatterers, number_of_scatterers), dtype=np.object)
    op = np.empty((number_of_scatterers, number_of_scatterers), dtype = np.object)
    for i in range(number_of_scatterers):
        filter_operators.append(0.5*identity_operators[i] - interior_operators[i])
        for j in range(number_of_scatterers):
            if i==j:
                op[i,j] = interior_operators[j] + exterior_operators[j]                
            else:
                transfer_operators[i,j] = bempp.api.operators.boundary.maxwell.multitrace_operator(grid = grids[j],
                                                                                                   wavenumber = k_ext,
                                                                                                   target = grids[i],
                                                                                                space_type = 'all_rwg')
                op[i,j] = rescale(transfer_operators[i,j], k_ext, mu_ext)

    result = bempp.api.assembly.blocked_operator.GeneralizedBlockedOperator(op)
            
    return [result, filter_operators]
'''

def PMCHWT_preconditioner(grids, k_ext, k_int, mu_ext, mu_int, interior_preconditioner = True):
    """Creates the PMCHWT block-diagonal preconditioner Ai or Ai+Ae for single and multi-particle problems"""

    from bempp.api.operators.boundary.maxwell import magnetic_field, electric_field
    from bempp.api.assembly.blocked_operator import GeneralizedBlockedOperator
    from bempp.api.assembly.blocked_operator import BlockedOperator

    number_of_scatterers = len(grids)
    op = np.empty((number_of_scatterers, number_of_scatterers), dtype = np.object)
    bc_space = [bempp.api.function_space(grid, 'BC',0) for grid in grids]
    rbc_space = [bempp.api.function_space(grid, 'RBC', 0) for grid in grids]
    rwg_space = [bempp.api.function_space(grid, 'RWG', 0) for grid in grids]

    for i in range(number_of_scatterers):
        Ai_pre = BlockedOperator(2,2) 
        Ae_pre = BlockedOperator(2,2)

        magnetic_ext = magnetic_field(domain = bc_space[i], range_= rwg_space[i], dual_to_range = rbc_space[i], wavenumber = k_ext,
                                      assembler = 'dense_evaluator')
        electric_ext = electric_field(domain = bc_space[i], range_= rwg_space[i], dual_to_range = rbc_space[i], wavenumber = k_ext,
                                      assembler = 'dense_evaluator')

        magnetic_int = magnetic_field(domain = bc_space[i], range_= rwg_space[i], dual_to_range = rbc_space[i], wavenumber = k_int[i],
                                      assembler = 'dense_evaluator')
        electric_int = electric_field(domain = bc_space[i], range_= rwg_space[i], dual_to_range = rbc_space[i], wavenumber = k_int[i],
                                      assembler = 'dense_evaluator')

        Ai_pre[0,0] = magnetic_int
        Ai_pre[0,1] = electric_int
        Ai_pre[1,0] = -1 * electric_int
        Ai_pre[1,1] = magnetic_int

        Ae_pre[0,0] = magnetic_ext
        Ae_pre[0,1] = electric_ext
        Ae_pre[1,0] = -1 * electric_ext
        Ae_pre[1,1] = magnetic_ext

        Ai_pre = rescale(Ai_pre, k_int[i], mu_int[i])
        Ae_pre = rescale(Ae_pre, k_ext, mu_ext)

        if interior_preconditioner:
            op[i,i] = Ai_pre
        else:
            op[i,i] = Ai_pre + Ae_pre

    result = GeneralizedBlockedOperator(op)
    return result

'''
def PMCHWT_preconditioner(grids, k_ext, k_int, mu_ext, mu_int, interior_preconditioner = True):
    """Creates the PMCHWT block-diagonal preconditioner Ai or Ai+Ae for single and multi-particle problems"""

    number_of_scatterers = len(grids)
    op = np.empty((number_of_scatterers, number_of_scatterers), dtype = np.object)
    for i in range(number_of_scatterers):
        Ai_pre = bempp.api.operators.boundary.maxwell.multitrace_operator(grid = grids[i], wavenumber = k_int[i],
									space_type = 'all_bc', 
									assembler = 'dense_evaluator')
        Ae_pre = bempp.api.operators.boundary.maxwell.multitrace_operator(grid = grids[i], wavenumber = k_ext,
									space_type = 'all_bc',
									assembler = 'dense_evaluator')
        Ai_pre = rescale(Ai_pre, k_int[i], mu_int[i])
        Ae_pre = rescale(Ae_pre, k_ext, mu_ext)
       
        if interior_preconditioner == False:
            op[i,i] = Ai_pre + Ae_pre
        else:
            op[i,i] = Ai_pre
    result = bempp.api.assembly.blocked_operator.GeneralizedBlockedOperator(op)
    return result
'''

def mass_matrix_BC_SNC(grids):
    number_of_scatterers = len(grids)
    
    bc_space = [bempp.api.function_space(grid, 'BC', 0) for grid in grids]

    rwg_space = [bempp.api.function_space(grid, "RWG", 0) for grid in grids]
    snc_space = [bempp.api.function_space(grid, "SNC", 0) for grid in grids]
    
    result = np.empty((2*number_of_scatterers,2*number_of_scatterers), dtype = 'O')

    for i in range(number_of_scatterers):
        ident = bempp.api.operators.boundary.sparse.identity(bc_space[i], rwg_space[i], snc_space[i])
        inv_ident = bempp.api.assembly.discrete_boundary_operator.InverseSparseDiscreteBoundaryOperator(ident.weak_form())
        result[2*i,2*i] = inv_ident
        result[2*i+1,2*i+1] = inv_ident

    result = bempp.api.assembly.blocked_operator.BlockedDiscreteOperator(result)
    return result

class _it_counter(object):
    """Iteration counter class."""
    def __init__(self, store_residuals, iteration_is_cg = False, operator = None, rhs = None):
        self._count = 0
        self._store_residuals = store_residuals
        self._residuals = []
        self._iteration_is_cg = iteration_is_cg
        self._operator = operator
        self._rhs = rhs

    def __call__(self,x):
        from bempp.api import log
        self._count += 1
        if self._store_residuals:
            if self._iteration_is_cg:
                res = self.rhs - self._operator * x
            else:
                res = x
                self._residuals.append(np.linalg.norm(res))
                log(f"GMRES Iteration {self._count} with residual {self._residuals[-1]}")
        else:
            log(f"GMRES Iteration {self._count}")

    @property
    def count(self):
        """Return the number of iterations."""
        return self._count

    @property
    def residuals(self):
        """Return the vector of residuals."""
        return self._residuals


def gmres(A, b, tol=1E-5, restart = None, 
          maxiter = None, use_strong_form = False, 
          return_residuals = False, return_iteration_count = False):
    """ Interface to the scipy.sparse.linalg.gmres function.

    This function behaves like the scipy.sparse.linalg.gmres function. But
    instead of a linear operator and a vector b it takes a boundary operator
    and a grid function or a blocked operator and a list of grid functions.
    The result is returned as a grid function or as a list of grid functions
    in the correct spaces.

    Modified to allow for numpy matrices, LinearOperator or weak_forms() as an input."""

    from bempp.api.assembly.boundary_operator import BoundaryOperator
    from bempp.api.assembly.blocked_operator import BlockedOperatorBase, coefficients_from_grid_functions_list, projections_from_grid_functions_list, grid_function_list_from_coefficients
    from bempp.api.assembly.grid_function import GridFunction

    if not isinstance(A, BoundaryOperator) and use_strong_form == True:
        raise ValueError("Strong form only with BoundaryOperator")
    if isinstance(A, BoundaryOperator) and not isinstance(b,GridFunction):
        raise ValueError("Instance Error")
    if not isinstance(A, BoundaryOperator) and isinstance(b, GridFunction):
        raise ValueError("Instance Error")

    # Assemble weak form before the logging messages
    if isinstance(A, BoundaryOperator) and isinstance(b, GridFunction):
        """Implementation for single operator"""
        if use_strong_form:
            if not A.range.is_compatible(b.space):
                raise ValueError("The range of A and the domain of A must have"
                		+ "the same number of unknowns if the strong form is used")
            A_op = A.strong_form()
            b_vec = b.coefficients
        else:
            A_op = A.weak_form()
            b_vec = b.projections(A.dual_to_range)
    elif isinstance(A, BlockedOperatorBase):
        """Implementation for blocked operators"""
        if use_strong_form:
            b_vec = coefficients_from_grid_functions_list(b)
            A_op = A.strong_form()
        else:
            A_op = A.weak_form()
            b_vec = projections_from_grid_functions_list(b,A.dual_to_range_spaces)
    else:
        A_op = A
        b_vec = b

    callback = _it_counter(return_residuals)


    bempp.api.log("Starting GMRES iteration")
    start_time = time.time()
    x, info = scipy.sparse.linalg.gmres(A_op, b_vec, tol=tol, restart=restart, maxiter=maxiter, callback=callback)
    end_time = time.time()
#    bempp.api.log("GMRES finished in %i iterations and took %.2 sec." %
#		   (callback.count, end_time-start_time))

    if isinstance(A, BoundaryOperator) and isinstance(b,GridFunction):
        res_fun = GridFunction(A.domain, coefficients = x.ravel())
    if isinstance(A, BlockedOperatorBase):
        res_fun = grid_function_list_from_coefficients(x.ravel(), A.domain_spaces)
    else:
        res_fun = x

    if return_residuals and return_iteration_count:
        return res_fun, info, callback.residuals, callback.count
    if return_residuals:
        return res_fun, info, callback.residuals
    if return_iteration_count:
        return res_fun, info, callback.count
    return res_fun, info
