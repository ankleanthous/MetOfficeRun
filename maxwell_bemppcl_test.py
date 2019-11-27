import bempp.api
import numpy as np
import time

bempp.api.set_default_device(0,0)
bempp.api.enable_console_logging()
bempp.api.DEVICE_PRECISION_GPU = 'single'

# Test case for hexagonal column
# small size particles
# size parameter 1
frequency = 664E9
wavelength = 3E8/frequency
print('wavelength: {0}'.format(wavelength))
k_ext = 2*np.pi/wavelength
print('k_ext: {0}'.format(k_ext))
h = 2*np.pi/(10*k_ext)
print(h)
#grid = bempp.api.import_grid('test_cases/small_size/hex_1_20_elements.msh')
grid = bempp.api.import_grid('test_cases/hex_10.msh')
grids = [grid]
#r = [1.7746 + 0.00064j]
r = [1.7746 + 0.00940j]
number_of_scatterers = len(grids)

#data = np.loadtxt('test_cases/small_size/Mishchenko_hexcol_1.txt', dtype = float)
data = np.loadtxt('test_cases/Mishchenko_hexcol_10.txt',dtype = float)
Z11 = data[:,1]
Z22 = data[:,2]
Z33 = data[:,3]
Z44 = data[:,4]
Z12 = data[:,5]
Z34 = data[:,6]
Tmatrix_coords = data[:,0]
Tmatrix_coords = np.radians(Tmatrix_coords)

'''
data = np.loadtxt('test_cases/small_size/SSPs_hexcol_1.txt')
Cext_Tmatrix = data[0] * 1E-12
Csca_Tmatrix = data[1] * 1E-12
g_Tmatrix = data[2]
w_Tmatrix = data[3]
'''

Cext_Tmatrix = 2882707.98
Csca_Tmatrix = 2473437
g_Tmatrix = 0.7118
w_Tmatrix = 0.8580

Nelements = np.sum([np.shape(grid.elements)[1] for grid in grids])
print('number of elements: {0}'.format(Nelements))

n_ind = r * number_of_scatterers
k_int = [k_ext * i for i in n_ind]

mu_ext = 1.0
mu_int = [1.0] * number_of_scatterers

def rescale(A, d1, d2):
    """Rescale the 2x2 block operator matrix A"""
    
    A[0, 1] = A[0, 1] * (d2 / d1)
    A[1, 0] = A[1, 0] * (d1 / d2)
    
    return A

print('Creating operators')
scaled_interior_operators = [
    rescale(bempp.api.operators.boundary.maxwell.multitrace_operator(
        grid, wavenumber, space_type='electric_dual', assembler='dense_evaluator'), 
            wavenumber, mu) for grid, wavenumber, mu in
            zip(grids, k_int, mu_int)
]

identity_operators = [
    bempp.api.operators.boundary.sparse.multitrace_identity(op)
    for op in scaled_interior_operators
]

exterior_operators = [
    rescale(bempp.api.operators.boundary.maxwell.multitrace_operator(
        grid, k_ext, space_type='electric_dual', assembler='dense_evaluator'), k_ext, mu_ext) for grid in grids
]

from bempp.api.assembly.blocked_operator import GeneralizedBlockedOperator

filter_operators = number_of_scatterers * [None]
transfer_operators = np.empty((number_of_scatterers, number_of_scatterers), dtype=np.object)

#The following will contain the left-hand side block operator
op = np.empty((number_of_scatterers, number_of_scatterers), dtype=np.object)

for i in range(number_of_scatterers):
    filter_operators[i] = .5 * identity_operators[i]- scaled_interior_operators[i]
    for j in range(number_of_scatterers):
        if i == j:
            # Create the diagonal elements
            op[i, j] = scaled_interior_operators[j] + exterior_operators[j]
        else:
            # Do the off-diagonal elements
            transfer_operators[i, j] = bempp.api.operators.boundary.maxwell.multitrace_operator(
                grids[j], k0, target=grids[i], assembler='dense_evaluator')
            op[i, j] = filter_operators[i] * transfer_operators[i, j]
blocked_operator = GeneralizedBlockedOperator(op)


number_of_angles = 1801

S11 = []
S12 = []
S13 = []
S14 = []
S21 = []
S22 = []
S23 = []
S24 = []
S31 = []
S32 = []
S33 = []
S34 = []
S41 = []
S42 = []
S43 = []
S44 = []

n_averaging = 3
coord = np.loadtxt('far_field/PointDistFiles/lebedev/lebedev_00{0}.txt'.format(n_averaging))
w_averaging = coord[:,2] #weights
print(np.shape(w_averaging)[0])
phi_averaging = np.radians(coord[:,0])
theta_averaging = np.radians(coord[:,1])
#w_averaging = np.array([1.])

n_leb = 59
coord = np.loadtxt('far_field/PointDistFiles/lebedev/lebedev_0{0}.txt'.format(n_leb))
w_leb = coord[:,2] #weights
phi_leb = np.radians(coord[:,0]) #theta
theta_leb = np.radians(coord[:,1]) #phi
coord_leb = np.vstack([np.sin(theta_leb)*np.cos(phi_leb), np.sin(theta_leb)*np.sin(phi_leb), np.cos(theta_leb)])

Cext = []
Csca = []
g = []
w = []

Cext_error = []
Csca_error = []
g_error = []
w_error = []

for counter in range(1):
#for counter in range(np.shape(w_averaging)[0]):
    print('Number of incident wave: {0}'.format(counter))
    theta_inc = theta_averaging[counter]
    phi_inc = phi_averaging[counter]

    print('theta = {0}, phi = {1}'.format(theta_inc, phi_inc))
    incident_direction = np.array([np.sin(theta_inc) * np.cos(phi_inc),
                                   np.sin(theta_inc) * np.sin(phi_inc),
                                    np.cos(theta_inc)])
    vector_theta_inc = np.array([np.cos(theta_inc) * np.cos(phi_inc),
                                 np.cos(theta_inc) * np.sin(phi_inc),
                                 -np.sin(theta_inc)])
    vector_phi_inc = np.array([-np.sin(phi_inc), np.cos(phi_inc), 0])

    print(incident_direction)
    print(vector_theta_inc)
    print(vector_phi_inc)

    incident_trace_phi = number_of_scatterers * [None]
    incident_trace_theta = number_of_scatterers * [None]
    rhs_phi = number_of_scatterers * [None]
    rhs_theta = number_of_scatterers * [None]

    @bempp.api.complex_callable
    def dirichlet_trace_phi_inc(point, n, domain_index, result):
        plane_wave_phi = vector_phi_inc * np.exp(1j * k_ext * np.dot(point, incident_direction))
        result[:] =  np.cross(plane_wave_phi, n)

    @bempp.api.complex_callable
    def dirichlet_trace_theta_inc(point, n, domain_index, result):
        plane_wave_theta = vector_theta_inc * np.exp(1j * k_ext * np.dot(point, incident_direction))
        result[:] =  np.cross(plane_wave_theta, n)

    @bempp.api.complex_callable
    def neumann_trace_phi_inc(point, n, domain_index, result):
        plane_wave_curl_phi = np.cross(incident_direction, vector_phi_inc) * 1j * k_ext *np.exp(
                                        1j * k_ext * np.dot(point, incident_direction))
        result[:] =  1./ (1j * k_ext) * np.cross(plane_wave_curl_phi, n)

    @bempp.api.complex_callable
    def neumann_trace_theta_inc(point, n, domain_index, result):
        plane_wave_curl_theta = np.cross(incident_direction, vector_theta_inc) * 1j * k_ext *np.exp(
                                        1j * k_ext * np.dot(point, incident_direction))
        result[:] =  1./ (1j * k_ext) * np.cross(plane_wave_curl_theta, n)

    print('creating rhs')

    rhs_phi = 2 * number_of_scatterers * [None]
    rhs_theta = 2*number_of_scatterers * [None]
    incident_trace_data_phi = number_of_scatterers * [None]
    incident_trace_data_theta = number_of_scatterers * [None]

    for i in range(number_of_scatterers):
        incident_trace_data_phi[i] = (bempp.api.GridFunction(blocked_operator.domain_spaces[2 * i], 
					fun=dirichlet_trace_phi_inc, dual_space=blocked_operator.dual_to_range_spaces[2 * i]),
            			      (k_ext/mu_ext) * bempp.api.GridFunction(blocked_operator.domain_spaces[2 * i + 1], 
					fun=neumann_trace_phi_inc, dual_space=blocked_operator.dual_to_range_spaces[2 * i + 1]))
        rhs_phi[2 * i], rhs_phi[2 * i + 1] = filter_operators[i] * incident_trace_data_phi[i]

        incident_trace_data_theta[i] = (bempp.api.GridFunction(blocked_operator.domain_spaces[2 * i], 
                                        fun=dirichlet_trace_theta_inc, dual_space=blocked_operator.dual_to_range_spaces[2 * i]),
                                      (k_ext/mu_ext)*bempp.api.GridFunction(blocked_operator.domain_spaces[2 * i + 1], 
                                        fun=neumann_trace_theta_inc, dual_space=blocked_operator.dual_to_range_spaces[2 * i + 1]))
        rhs_theta[2 * i], rhs_theta[2 * i + 1] = filter_operators[i] * incident_trace_data_theta[i]


    print('solving the system')
    t0 = time.time()
    x_phi, info_phi, residuals_phi = bempp.api.linalg.gmres(blocked_operator*blocked_operator, blocked_operator*rhs_phi, use_strong_form=True, return_residuals=True)
    t_solve_phi = time.time() - t0
    print(len(residuals_phi))
    print('solver time phi: {0} mins'.format(t_solve_phi/60))

    t0 = time.time()
    x_theta, info_theta, residuals_theta = bempp.api.linalg.gmres(blocked_operator*blocked_operator, blocked_operator*rhs_theta, use_strong_form=True, return_residuals=True)
    t_solve_theta = time.time() - t0
    print(len(residuals_theta))
    print('solver time phi: {0} mins'.format(t_solve_theta/60))

    scattered_dirichlet_phi = number_of_scatterers * [None]
    scattered_neumann_phi = number_of_scatterers * [None]

    scattered_dirichlet_theta = number_of_scatterers * [None]
    scattered_neumann_theta = number_of_scatterers * [None]

    for i in range(number_of_scatterers):
        scattered_dirichlet_phi[i] = x_phi[2*i]
        scattered_neumann_phi[i] =  (mu_ext/k_ext) * x_phi[2*i+1]

        scattered_dirichlet_theta[i] = x_theta[2*i]
        scattered_neumann_theta[i] = (mu_ext/k_ext)*x_theta[2*i+1]

    ###################################################################################
    ## Computing Cext
    ###################################################################################
    far_field_phi = np.zeros((3, 1), dtype='complex128')
    far_field_theta = np.zeros((3, 1), dtype = 'complex128')

    incident_direction = np.array([[np.sin(theta_inc) * np.cos(phi_inc),
                                   np.sin(theta_inc) * np.sin(phi_inc),
                                   np.cos(theta_inc)]])


    for i in range(number_of_scatterers):

        electric_far = bempp.api.operators.far_field.maxwell.electric_field(
                                        scattered_neumann_phi[i].space, incident_direction.T, k_ext)
        magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(
                                scattered_dirichlet_phi[i].space, incident_direction.T, k_ext)

        far_field_phi += -electric_far * scattered_neumann_phi[i] - magnetic_far * scattered_dirichlet_phi[i]
        far_field_theta += -electric_far * scattered_neumann_theta[i] - magnetic_far * scattered_dirichlet_theta[i]

    Cext_phi = 4*np.pi/(k_ext * np.linalg.norm(vector_phi_inc)**2) * np.imag(np.dot(far_field_phi[:,0], np.conjugate(vector_phi_inc)))
    Cext_theta = 4*np.pi/(k_ext * np.linalg.norm(vector_theta_inc)**2) * np.imag(np.dot(far_field_theta[:,0],np.conjugate(vector_theta_inc)))
    print('Cext_phi: {0}'.format(Cext_phi))
    print('Cext_theta: {0}'.format(Cext_theta))

    Cext.append(0.5 * (Cext_phi + Cext_theta))
    print("Cext: {0}".format(Cext[-1]))
    Cext_error.append(abs(np.dot(Cext, w_averaging[0:counter+1]) - Cext_Tmatrix)/Cext_Tmatrix)
    print('Error in Cext: {0}'.format(Cext_error[-1]))


    ###################################################################################
    ## Computing Csca
    ###################################################################################
    coord_leb_Csca = np.vstack([np.sin(theta_leb)*np.cos(phi_leb),
                                np.sin(theta_leb)*np.sin(phi_leb),
                                np.cos(theta_leb)])
    ff_quad_phi = np.zeros(np.shape(coord_leb_Csca), dtype='complex128')
    ff_quad_theta = np.zeros(np.shape(coord_leb_Csca), dtype = 'complex128')

    for i in range(number_of_scatterers):
        electric_far = bempp.api.operators.far_field.maxwell.electric_field(scattered_neumann_phi[i].space,
                                                                            coord_leb_Csca, k_ext)
        magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(scattered_dirichlet_phi[i].space,
                                                                            coord_leb_Csca, k_ext)

        ff_quad_phi += -electric_far * scattered_neumann_phi[i] - magnetic_far * scattered_dirichlet_phi[i]
        ff_quad_theta += -electric_far * scattered_neumann_theta[i] - magnetic_far * scattered_dirichlet_theta[i]

    ff_quad_phi_mag = np.linalg.norm(ff_quad_phi, axis = 0)**2
    ff_quad_theta_mag = np.linalg.norm(ff_quad_theta, axis = 0)**2

    Int_phi = 4*np.pi * np.dot(ff_quad_phi_mag,w_leb)
    Int_theta = 4*np.pi * np.dot(ff_quad_theta_mag,w_leb)# the 4*np.pi factor comes from the quadrature rule

    Csca_phi = 1/np.linalg.norm(vector_phi_inc)**2 * Int_phi
    Csca_theta = 1/np.linalg.norm(vector_theta_inc)**2 * Int_theta

    Csca.append(0.5 * (Csca_phi + Csca_theta))
    print("Csca: {0}".format(Csca[-1]))
    print(w_averaging[0:counter+1])
    Csca_error.append(abs(np.dot(Csca, w_averaging[0:counter+1]) - Csca_Tmatrix)/Csca_Tmatrix)
    print("Error in Csca: {0}".format(Csca_error[-1]))

    w.append(Csca[-1]/Cext[-1])
    w_error.append(abs(np.dot(w, w_averaging[0:counter+1]) - w_Tmatrix)/w_Tmatrix)
    print("Error in w: {0}".format(w_error[-1]))

    ##################################################################
    ## Computing g
    ##################################################################
    coord_leb_g = np.vstack([np.sin(theta_leb + theta_inc)*np.cos(phi_inc),
                             np.sin(theta_leb + theta_inc)*np.sin(phi_inc),
                             np.cos(theta_leb + theta_inc)])
    ff_quad_phi = np.zeros(np.shape(coord_leb_g), dtype='complex128')
    ff_quad_theta = np.zeros(np.shape(coord_leb_g), dtype = 'complex128')

    for i in range(number_of_scatterers):
        electric_far = bempp.api.operators.far_field.maxwell.electric_field(scattered_neumann_phi[i].space,
                                                                        coord_leb_g, k_ext)
        magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(scattered_dirichlet_phi[i].space,
                                                                        coord_leb_g, k_ext)

        ff_quad_phi += -electric_far * scattered_neumann_phi[i] - magnetic_far * scattered_dirichlet_phi[i]
        ff_quad_theta += -electric_far * scattered_neumann_theta[i] - magnetic_far * scattered_dirichlet_theta[i]

    ff_quad_phi_mag = np.linalg.norm(ff_quad_phi, axis = 0)**2
    ff_quad_theta_mag = np.linalg.norm(ff_quad_theta, axis = 0)**2

    Int_g_phi = 4*np.pi * np.dot(ff_quad_phi_mag * np.cos(theta_leb), w_leb)
    Int_g_theta = 4 * np.pi * np.dot(ff_quad_theta_mag * np.cos(theta_leb), w_leb)

    g.append(0.5/Csca[-1] * (Int_g_phi + Int_g_theta))

    g_error.append(abs(np.dot(g, w_averaging[0:counter+1]) - g_Tmatrix)/g_Tmatrix)
    print('g: {0}'.format(g[-1]))
    print("Error in g: {0}".format(g_error[-1]))


    ##################################################################
    ## Computing phase matrix
    ##################################################################
    far_field_phi = np.zeros((3, number_of_angles), dtype='complex128')
    far_field_theta = np.zeros((3, number_of_angles), dtype = 'complex128')

#    R_theta = np.array([[np.cos(theta_inc), 0 ,np.sin(theta_inc)], [0,1,0], [-np.sin(theta_inc), 0, np.cos(theta_inc)]])
#    R_phi = np.array([[np.cos(phi_inc), -np.sin(phi_inc), 0], [np.sin(phi_inc), np.cos(phi_inc), 0], [0,0,1]])

 #   beta_matrix = np.dot(R_phi, R_theta)
 #   beta_matrix_inv = np.linalg.inv(beta_matrix)

    angles_theta= np.linspace(0 + theta_inc, np.pi + theta_inc, number_of_angles)
    angles_phi = 0. + phi_inc


    scattering_direction = np.array([np.sin(angles_theta) * np.cos(angles_phi),
                                     np.sin(angles_theta) * np.sin(angles_phi),
                                     np.cos(angles_theta)])
    for i in range(number_of_scatterers):
        electric_far = bempp.api.operators.far_field.maxwell.electric_field(scattered_neumann_phi[i].space,
                                                                        scattering_direction, k_ext)
        magnetic_far = bempp.api.operators.far_field.maxwell.magnetic_field(scattered_dirichlet_phi[i].space,
                                                                        scattering_direction, k_ext)

        far_field_phi += -electric_far * scattered_neumann_phi[i] - magnetic_far * scattered_dirichlet_phi[i]
        far_field_theta += -electric_far * scattered_neumann_theta[i] - magnetic_far * scattered_dirichlet_theta[i]

    vector_theta_sca = np.array([np.cos(angles_theta)*np.cos(angles_phi),
                                np.cos(angles_theta)*np.sin(angles_phi),
                                -np.sin(angles_theta)])
    vector_phi_sca = np.array([-np.sin(angles_phi)*np.ones(number_of_angles),
                               np.cos(angles_phi)*np.ones(number_of_angles),
                               np.zeros(number_of_angles)])

    a11 = []
    a12 = []
    a21 = []
    a22 = []

    for i in range(number_of_angles):
        a11.append(np.dot(vector_theta_sca[:,i], far_field_theta[:,i]))
        a12.append(np.dot(vector_theta_sca[:,i], far_field_phi[:,i]))
        a21.append(np.dot(vector_phi_sca[:,i], far_field_theta[:,i]))
        a22.append(np.dot(vector_phi_sca[:,i], far_field_phi[:,i]))

    A11 = np.array(a11)
    A12 = np.array(a12)
    A21 = np.array(a21)
    A22 = np.array(a22)

    S11.append(0.5 * (abs(A11)**2 + abs(A12)**2 + abs(A21)**2 + abs(A22)**2))
    S12.append(0.5*(abs(A11)**2-abs(A22)**2+abs(A21)**2-abs(A12)**2))
    S13.append(-np.real(A11*np.conjugate(A12)+A22*np.conjugate(A21)))
    S14.append(-np.imag(A11*np.conjugate(A12)-A22*np.conjugate(A21)))
    S21.append(0.5*(abs(A11)**2-abs(A22)**2-abs(A21)**2+abs(A12)**2))
    S22.append(0.5*(abs(A11)**2+abs(A22)**2-abs(A21)**2-abs(A12)**2))
    S23.append(-np.real(A11*np.conjugate(A12)-A22*np.conjugate(A21)))
    S24.append(-np.imag(A11*np.conjugate(A12)+A22*np.conjugate(A21)))
    S31.append(-np.real(A11*np.conjugate(A21)+A22*np.conjugate(A12)))
    S32.append(-np.real(A11*np.conjugate(A21)-A22*np.conjugate(A12)))
    S33.append(np.real(np.conjugate(A11)*A22+A12*np.conjugate(A21)))
    S34.append(np.imag(A11*np.conjugate(A22)+A21*np.conjugate(A12)))
    S41.append(-np.imag(np.conjugate(A11)*A21+np.conjugate(A12)*A22))
    S42.append(-np.imag(np.conjugate(A11)*A21-np.conjugate(A12)*A22))
    S43.append(np.imag(A22*np.conjugate(A11)-A12*np.conjugate(A21)))
    S44.append(np.real(np.conjugate(A11)*A22-A12*np.conjugate(A21)))

    print('---------------------------------------------------------------')

import matplotlib.pyplot as plt
plt.figure()
plt.semilogy(Csca_error, '8-', label = 'Csca')
plt.semilogy(Cext_error,'*-', label = 'Cext')
plt.semilogy(w_error,'o-', label = 'w')
plt.semilogy(g_error,'D-', label = 'g')
plt.legend()
plt.savefig('hexcol_smallrefined_SSPs_bemppcl_blockdisc.png')

print("Error in Csca: {0}".format(abs(np.dot(Csca, w_averaging) - Csca_Tmatrix)/Csca_Tmatrix))
print("Error in Cext: {0}".format(abs(np.dot(Cext, w_averaging) - Cext_Tmatrix)/Cext_Tmatrix))
print("Error in g: {0}".format(abs(np.dot(g, w_averaging) - g_Tmatrix)/g_Tmatrix))
print("Error in w: {0}".format(abs(np.dot(w, w_averaging) - w_Tmatrix)/w_Tmatrix))


S11 = np.reshape(S11, (len(w_averaging), number_of_angles))
S12 = np.reshape(S12, (len(w_averaging), number_of_angles))
S13 = np.reshape(S13, (len(w_averaging), number_of_angles))
S14 = np.reshape(S14, (len(w_averaging), number_of_angles))
S21 = np.reshape(S21, (len(w_averaging), number_of_angles))
S22 = np.reshape(S22, (len(w_averaging), number_of_angles))
S23 = np.reshape(S23, (len(w_averaging), number_of_angles))
S24 = np.reshape(S24, (len(w_averaging), number_of_angles))
S31 = np.reshape(S31, (len(w_averaging), number_of_angles))
S32 = np.reshape(S32, (len(w_averaging), number_of_angles))
S33 = np.reshape(S33, (len(w_averaging), number_of_angles))
S34 = np.reshape(S34, (len(w_averaging), number_of_angles))
S41 = np.reshape(S41, (len(w_averaging), number_of_angles))
S42 = np.reshape(S42, (len(w_averaging), number_of_angles))
S43 = np.reshape(S43, (len(w_averaging), number_of_angles))
S44 = np.reshape(S44, (len(w_averaging), number_of_angles))

s11 = []
s12 = []
s13 = []
s14 = []
s21 = []
s22 = []
s23 = []
s24 = []
s31 = []
s32 = []
s33 = []
s34 = []
s41 = []
s42 = []
s43 = []
s44 = []
for i in range(np.shape(S11)[1]):
    s11.append(np.dot(S11[:,i], w_averaging))
    s12.append(np.dot(S12[:,i], w_averaging))
    s13.append(np.dot(S13[:,i], w_averaging))
    s14.append(np.dot(S14[:,i], w_averaging))
    s21.append(np.dot(S21[:,i], w_averaging))
    s22.append(np.dot(S22[:,i], w_averaging))
    s23.append(np.dot(S23[:,i], w_averaging))
    s24.append(np.dot(S24[:,i], w_averaging))
    s31.append(np.dot(S31[:,i], w_averaging))
    s32.append(np.dot(S32[:,i], w_averaging))
    s33.append(np.dot(S33[:,i], w_averaging))
    s34.append(np.dot(S34[:,i], w_averaging))
    s41.append(np.dot(S41[:,i], w_averaging))
    s42.append(np.dot(S42[:,i], w_averaging))
    s43.append(np.dot(S43[:,i], w_averaging))
    s44.append(np.dot(S44[:,i], w_averaging))


s11 = np.array(s11)
s12 = np.array(s12)
s13 = np.array(s13)
s14 = np.array(s14)
s21 = np.array(s21)
s22 = np.array(s22)
s23 = np.array(s23)
s24 = np.array(s24)
s31 = np.array(s31)
s32 = np.array(s32)
s33 = np.array(s33)
s34 = np.array(s34)
s41 = np.array(s41)
s42 = np.array(s42)
s43 = np.array(s43)
s44 = np.array(s44)

Csca_avg = np.dot(Csca, w_averaging)
scaling = 4*np.pi/Csca_avg
angles_theta = np.linspace(0,np.pi,number_of_angles)


fig, axs = plt.subplots(4, 4, figsize = (15,15))
axs[0, 0].semilogy(Tmatrix_coords, Z11, 'b', label = 'T-matrix')
axs[0, 0].semilogy(angles_theta, scaling*s11.T, '--r', label = 'BEM')
axs[0, 0].set_title('F11')
axs[0, 0].legend()

axs[0, 1].plot(Tmatrix_coords, Z12, 'b', label = 'T-matrix')
axs[0, 1].plot(angles_theta, scaling*s12.T, '--r', label = 'BEM')
axs[0, 1].set_title('F12')
axs[0, 1].legend()

axs[0, 2].plot(angles_theta, scaling*s13.T, '--r', label = 'BEM')
axs[0, 2].set_title('F13')
axs[0, 2].legend()

axs[0, 3].plot(angles_theta, scaling*s14.T, '--r', label = 'BEM')
axs[0, 3].set_title('F14')
axs[0, 3].legend()

axs[1, 0].plot(Tmatrix_coords, Z12, 'b', label = 'T-matrix')
axs[1, 0].plot(angles_theta, scaling*s21.T, '--r', label = 'BEM')
axs[1, 0].set_title('F21')
axs[1, 0].legend()

axs[1, 1].semilogy(Tmatrix_coords, Z22, 'b', label = 'T-matrix')
axs[1, 1].semilogy(angles_theta, scaling*s22.T, '--r', label = 'BEM')
axs[1, 1].set_title('F22')
axs[1, 1].legend()

axs[1, 2].plot(angles_theta,scaling* s23.T, '--r', label = 'BEM')
axs[1, 2].set_title('F23')
axs[1, 2].legend()

axs[1, 3].plot(angles_theta,scaling* s24.T, '--r', label = 'BEM')
axs[1, 3].set_title('F24')
axs[1, 3].legend()


axs[2, 0].plot(angles_theta,scaling* s31.T, '--r', label = 'BEM')
axs[2, 0].set_title('F31')
axs[2, 0].legend()

axs[2, 1].plot(angles_theta,scaling* s32.T, '--r', label = 'BEM')
axs[2, 1].set_title('F32')
axs[2, 1].legend()

axs[2, 2].plot(Tmatrix_coords, Z33, 'b', label = 'T-matrix')
axs[2, 2].plot(angles_theta,scaling* s33.T, '--r', label = 'BEM')
axs[2, 2].set_title('F33')
axs[2, 2].legend()

axs[2, 3].plot(Tmatrix_coords, Z34, 'b', label = 'T-matrix')
axs[2, 3].plot(angles_theta,scaling* s34.T, '--r', label = 'BEM')
axs[2, 3].set_title('F34')
axs[2, 3].legend()

axs[3, 0].plot(angles_theta,scaling* s41.T, '--r', label = 'BEM')
axs[3, 0].set_title('F41')
axs[3, 0].legend()

axs[3, 1].plot(angles_theta,scaling* s42.T, '--r', label = 'BEM')
axs[3, 1].set_title('F42')
axs[3, 1].legend()

axs[3, 2].plot(Tmatrix_coords, -Z34, 'b', label = 'T-matrix')
axs[3, 2].plot(angles_theta,scaling* s43.T, '--r', label = 'BEM')
axs[3, 2].set_title('F43')
axs[3, 2].legend()

axs[3, 3].plot(Tmatrix_coords, Z44, 'b', label = 'T-matrix')
axs[3, 3].plot(angles_theta,scaling* s44.T, '--r', label = 'BEM')
axs[3, 3].set_title('F44')
axs[3, 3].legend()

fig.savefig('hexcol_smallrefined_phasematrix_bemppcl_blockdisc.png')







