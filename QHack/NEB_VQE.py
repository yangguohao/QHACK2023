import multiprocessing

import pennylane as qml
from pennylane import numpy as np
import time
import warnings

warnings.filterwarnings('ignore')

from pennylane.templates import AllSinglesDoubles


def energy_diff(coordinates, energy_method='diag'):
    r"""
    According to the molecular coordinates, the Hamiltonian is generated and its ground state energy is returned
    """
    mult = 2
    charge = 0
    basis = "sto-3g"
    symbols = ["H", "H", "H"]
    bohr_angs = 0.529177210903
    coordinates = coordinates / bohr_angs

    if energy_method == 'diag':
        energy = qml.eigvals(qml.qchem.molecular_hamiltonian(symbols,
                                                             qml.math.toarray(coordinates),
                                                             mult=mult,
                                                             charge=charge,
                                                             basis=basis,
                                                             )[0])[0]
    elif energy_method == 'vqe':

        # Givens vqe 
        singles, doubles, hf = single_double()
        params = np.zeros(len(singles) + len(doubles), requires_grad=True)

        hamiltonian = qml.qchem.molecular_hamiltonian(symbols,
                                                      qml.math.toarray(coordinates),
                                                      mult=mult,
                                                      charge=charge,
                                                      basis=basis,
                                                      )[0]

        # select optimizer
        opt = qml.GradientDescentOptimizer(stepsize=0.4)
        prev_energy = 0.0
        for n in range(100):
            # perform optimization step
            params_list, energy = opt.step_and_cost(givens_circuit, hamiltonian, params)
            params = params_list[1]

            if np.abs(energy - prev_energy) < 1e-6:
                break
            prev_energy = energy

    return energy


dev_givens = qml.device("default.qubit", wires=6)


@qml.qnode(dev_givens)
def givens_circuit(hamiltonian, params, qubit_num=6):
    r"""
    Construct givens circuit to solve the expected value of Hamiltonian
    """
    singles, doubles, hf = single_double()
    AllSinglesDoubles(params, range(qubit_num), hf, singles, doubles)
    return qml.expval(hamiltonian)


def single_double(electrons=3, orbitals=6):
    r"""
    Generate all single and double excited gates and hf states
    """
    hf = qml.qchem.hf_state(electrons, orbitals)

    delta_sz_list = [0, 1, -1, 2, -2]
    singles_list = []
    doubles_list = []

    for delta_sz in delta_sz_list:
        singles, doubles = qml.qchem.excitations(electrons, orbitals, delta_sz=delta_sz)

        singles_list += singles
        doubles_list += doubles

    return singles_list, doubles_list, hf


def grad_energy_diff(reaction_coor, energy_method='diag'):
    r"""
    Calculate the energy gradient, $∇E(R_i)$
    """
    grad = []
    shift_val = 0.1
    for i in [5, -1]:
        shift = np.zeros_like(reaction_coor)
        shift[i] = shift_val
        value_1 = energy_diff(reaction_coor + shift, energy_method)
        value_2 = energy_diff(reaction_coor - shift, energy_method)

        grad_value = (value_1 - value_2) / (2 * shift_val)

        grad.append(grad_value)

    gradient = [0., 0., 0., 0., 0., grad[0], 0., 0., grad[-1]]

    return np.array(gradient)


@qml.qnode(qml.device("default.qubit", wires=6))
def frame_circuit(train_params, cir_depth, qubit_num, fix_params):
    r"""
    Reaction path coding circuit
    """
    for i in range(qubit_num):
        qml.RY(fix_params[i], wires=i)

    for d in range(cir_depth):
        for q in range(qubit_num):
            qml.RY(train_params[2 * qubit_num * d + 2 * q], wires=q)
            qml.RZ(train_params[2 * qubit_num * d + 2 * q + 1], wires=q)

    if qubit_num > 1:
        for q in range(qubit_num - 1):
            qml.CZ(wires=[q, q + 1])

    measurement = []
    for i in range(qubit_num):
        measurement.append(qml.probs(wires=[i]))
    return measurement


def reaction_coordinate(measure_pro, frame_num, key_num, ref_key):
    r"""
    Restore reaction coordinates
    """
    measure_pro = measure_pro.reshape(frame_num, key_num)
    reaction_coor = []

    for j in range(frame_num):
        reaction_coor_1 = [0., 0., 0., 0., 0., measure_pro[j][0] * ref_key, 0., 0., measure_pro[j][1] * ref_key]
        reaction_coor.append(reaction_coor_1)

    return np.array(reaction_coor)


def fix_params_funtion(key_length_array, ref_key):
    r"""
    Fixed parameters in frame circuit
    """
    _, y = key_length_array.shape
    for i in range(1, y):
        for j in range(0, i):
            key_length_array[:, i] = key_length_array[:, i] + key_length_array[:, j]
    fractional_coo_array = key_length_array / ref_key

    phi_array = fractional_coo_array.ravel()
    phi_list = list(map(lambda x: 2 * np.arccos(np.sqrt(x)), phi_array))

    return phi_list


def tangent_vector(gold_coor_array, energy_list):
    r"""
    Formula 5 and 6
    """
    tao_normalized_list = []
    for i in range(1, len(energy_list) - 1):
        if energy_list[i + 1] > energy_list[i] > energy_list[i - 1]:
            tao = gold_coor_array[i + 1] - gold_coor_array[i]

        elif energy_list[i + 1] < energy_list[i] < energy_list[i - 1]:
            tao = gold_coor_array[i] - gold_coor_array[i - 1]

        elif energy_list[i + 1] < energy_list[i] > energy_list[i - 1] or energy_list[i + 1] > energy_list[i] < \
                energy_list[i - 1]:

            tao_plus = gold_coor_array[i + 1] - gold_coor_array[i]
            tao_minus = gold_coor_array[i] - gold_coor_array[i - 1]

            max_delta_E = max(abs(energy_list[i + 1] - energy_list[i]), abs(energy_list[i - 1] - energy_list[i]))
            min_delta_E = min(abs(energy_list[i + 1] - energy_list[i]), abs(energy_list[i - 1] - energy_list[i]))

            if energy_list[i + 1] > energy_list[i - 1]:
                tao = tao_plus * max_delta_E + tao_minus * min_delta_E
            elif energy_list[i + 1] < energy_list[i - 1]:
                tao = tao_plus * min_delta_E + tao_minus * max_delta_E
        tao_normalized = tao / np.linalg.norm(tao)
        tao_normalized_list.append(tao_normalized)

    return np.array(tao_normalized_list)


def tangent_grad_energy(energy_grad_list, tao_normalized_list):
    r"""
    Formula 4
    """
    tangent_grad_energy_list = []

    for i in range(1, len(energy_grad_list) - 1):
        vec = energy_grad_list[i] - energy_grad_list[i] * tao_normalized_list[i - 1]

        tangent_grad_energy_list.append(vec)
    return np.array(tangent_grad_energy_list)


def tangent_spring_force(gold_coor_array, tao_normalized_list, K=0.1):
    r"""
    Formula 3
    """
    tangent_spring_force_list = []

    for i in range(1, len(gold_coor_array) - 1):
        tangent_F = K * (np.linalg.norm(gold_coor_array[i + 1] - gold_coor_array[i])
                         - np.linalg.norm(gold_coor_array[i] - gold_coor_array[i - 1])) * tao_normalized_list[i - 1]
        tangent_spring_force_list.append(tangent_F)

    return np.array(tangent_spring_force_list)


def average_force(gold_coor_array, fix_energy_list, fix_energy_grad_list, K=0.1, energy_method='diag'):
    r"""
    Formula 1
    """
    energy_list = []

    energy_grad_list = []
    for i in range(1, len(gold_coor_array) - 1):
        energy = energy_diff(gold_coor_array[i], energy_method)
        energy_list.append(energy)
        energy_grad = grad_energy_diff(gold_coor_array[i], energy_method)
        energy_grad_list.append(energy_grad)

    tao_normalized_list = tangent_vector(gold_coor_array, [fix_energy_list[0]] + energy_list + [fix_energy_list[1]])

    tangent_grad_energy_list = tangent_grad_energy(
        [fix_energy_grad_list[0]] + energy_grad_list + [fix_energy_grad_list[1]], tao_normalized_list)

    tangent_spring_force_list = tangent_spring_force(gold_coor_array, tao_normalized_list, K=K)

    image_num = len(gold_coor_array) - 2

    ave_force = 0.0
    for i in range(image_num):
        force_vec = tangent_spring_force_list[i] - tangent_grad_energy_list[i]

        ave_force += np.linalg.norm(force_vec)

    ave_force = ave_force / image_num
    return np.array(ave_force)


def get_reaction_coor(train_params, cir_depth, qubit_num, fix_params, frame_num, key_num, ref_key):
    measure_prob = frame_circuit(train_params, cir_depth, qubit_num, fix_params)[:, 0]
    reaction_coor = reaction_coordinate(measure_prob, frame_num, key_num, ref_key)
    return reaction_coor


def get_gold_coor(train_params, cir_depth, qubit_num, fix_params, frame_num, key_num, ref_key, fix_coor):
    reaction_coor = get_reaction_coor(train_params, cir_depth, qubit_num, fix_params, frame_num, key_num, ref_key)
    gold_coor = [fix_coor[0]]

    for i in range(len(reaction_coor)):
        gold_coor.append(reaction_coor[i])
    gold_coor.append(fix_coor[1])

    gold_coor = np.array(gold_coor)
    return gold_coor


def computing_ave_force(cir_dict, train_params):
    r"""
    Calculate average force.
    """
    try:
        frame_num = cir_dict['frame_num']
        key_num = cir_dict['key_num']
        fix_coor = cir_dict['fix_coor']
        ref_key = cir_dict['ref_key']

        fix_params = cir_dict['fix_params']
        cir_depth = cir_dict['cir_depth']
        qubit_num = cir_dict['qubit_num']
        energy_method = cir_dict['energy_method']
        fix_energy_list = cir_dict['fix_energy_list']
        fix_energy_grad_list = cir_dict['fix_energy_grad_list']

        gold_coor = get_gold_coor(train_params, cir_depth, qubit_num, fix_params, frame_num, key_num, ref_key, fix_coor)

        ave_force = average_force(gold_coor, fix_energy_list, fix_energy_grad_list, energy_method=energy_method)
        return ave_force
    except KeyboardInterrupt:
        import os
        print(f'Process Done... {os.getpid()}')


def average_force_grad(cir_dict, train_params):
    r"""
    Calculate the gradient of the average force. Using the central difference method.
    """
    shift_val = 0.001
    shift_params_list = []
    for i in range(len(train_params)):
        shift = np.zeros_like(train_params)
        shift[i] += shift_val
        shift_params_list.append(train_params + shift)
        shift_params_list.append(train_params - shift)

    # Use multi-process parallel computing
    num_process = cir_dict['num_process']
    with multiprocessing.Pool(num_process) as p:
        arr = p.starmap(computing_ave_force, [(cir_dict, params) for params in shift_params_list])

    gradient_list = []
    for i in range(0, len(arr) - 1, 2):
        gradient_list.append((arr[i] - arr[i + 1]) / (2 * shift_val))

    return np.array(gradient_list)


def train_NEB_VQE(fix_coor,
                  frame_key_length_list,
                  ref_key,
                  frame_cir_depth=1,
                  iter_num=100,
                  lr=0.01,
                  opt='adam',
                  energy_method='diag',
                  verbose=True,
                  num_process=6):
    r"""
    Complete algorithm flow
    """
    fix_coor = np.array(fix_coor, requires_grad=False)
    key_length_array = np.array(frame_key_length_list, requires_grad=False)
    x, y = key_length_array.shape
    qubit_num = x * y

    phi_list = fix_params_funtion(key_length_array, ref_key)

    train_params = np.zeros(frame_cir_depth * qubit_num * 2, requires_grad=True)

    fix_energy_list = [energy_diff(fix_coor[0]), energy_diff(fix_coor[1])]
    fix_energy_grad_list = [grad_energy_diff(fix_coor[0]), grad_energy_diff(fix_coor[1])]

    ave_force_list = []

    cir_dict = {
        'qubit_num': qubit_num,
        'cir_depth': frame_cir_depth,
        'fix_params': phi_list,
        'frame_num': x,
        'key_num': y,
        'fix_coor': fix_coor,
        'ref_key': ref_key,
        'energy_method': energy_method,
        'fix_energy_list': fix_energy_list,
        'fix_energy_grad_list': fix_energy_grad_list,
        'num_process': num_process
    }

    if opt == 'adam':
        opt_force = qml.AdamOptimizer(stepsize=lr)
    elif opt == 'sdg':
        opt_force = qml.GradientDescentOptimizer(stepsize=lr)
    elif opt == 'momentum':
        opt_force = qml.MomentumOptimizer(stepsize=lr)

    prams_list = []
    corr_list = []

    for itr in range(iter_num):
        start = time.time()

        prams, loss = opt_force.step_and_cost(computing_ave_force, cir_dict, train_params, grad_fn=average_force_grad)
        ave_force_list.append(loss)
        prams_list.append(train_params)

        train_params = prams[-1]

        measure_pro = frame_circuit(train_params, frame_cir_depth, qubit_num, phi_list)[:, 0]
        reaction_coor = reaction_coordinate(measure_pro, x, y, ref_key)
        corr_list.append(reaction_coor)

        end = time.time()

        if verbose:
            print(f"iter: {itr + 1}, ave force: {loss: .8f}, time: {end - start}")
            print(f"gold_coor: {reaction_coor}")

    return ave_force_list, prams_list, corr_list


def coor_to_key(gold_coor_list):
    r"""
    Input the response coordinates obtained from NEB training, 
    take only the coordinates of the last group of images, and convert them 
    into key length information for storage

    """
    result_key_len_list = []

    gold_coor = gold_coor_list[-1]

    for i in range(len(gold_coor)):
        bond_1 = gold_coor[i][5]
        bond_2 = gold_coor[i][-1] - gold_coor[i][5]

        result_key_len_list.append([bond_1, bond_2])

    init_ket_list = []
    for i in range(len(gold_coor_list[0])):
        bond_1 = gold_coor_list[0][i][5]
        bond_2 = gold_coor_list[0][i][-1] - gold_coor_list[0][i][5]

        init_ket_list.append([bond_1, bond_2])

    return np.array(result_key_len_list), np.array(init_ket_list)


# Use the classical method to calculate the ground state energy of each reaction coordinate
def Q_orig(d, alpha, r, r0):
    return (d / 2) * (1.5 * np.exp(-2 * alpha * (r - r0)) - np.exp(-alpha * (r - r0)))


def J_orig(d, alpha, r, r0):
    return (d / 4) * (np.exp(-2 * alpha * (r - r0)) - 6 * np.exp(-alpha * (r - r0)))


def V_LEPS(rAB, rBC):
    a = 0.05
    b = 0.30
    c = 0.05
    dAB = dBC = 2
    dAC = 4
    r0 = 0.73
    alpha = 1.942
    QAB = Q_orig(dAB, alpha, rAB, r0)
    QBC = Q_orig(dBC, alpha, rBC, r0)
    rAC = rAB + rBC
    QAC = Q_orig(dAC, alpha, rAC, r0)
    Q_values = (QAB / (1 + a)) + (QBC / (1 + b)) + (QAC / (1 + c))

    JAB = J_orig(dAB, alpha, rAB, r0)
    JBC = J_orig(dBC, alpha, rBC, r0)
    JAC = J_orig(dAC, alpha, rAC, r0)
    J_values = (JAB / (1 + a)) ** 2 + (JBC / (1 + b)) ** 2 + (JAC / (1 + c)) ** 2
    J_values = J_values - ((JAB * JBC / ((1 + a) * (1 + b))) + (JBC * JAC / ((1 + b) * (1 + c))) + (
            JAB * JAC / ((1 + a) * (1 + c))))
    return Q_values - np.sqrt(J_values)


def V_LEPS_II(rAB, x):
    rAC = 4
    kC = 0.2025
    V_normal = V_LEPS(rAB, rAC - rAB)
    c = 1.154
    return V_normal + 2 * kC * (rAB - (rAC / 2 - x / c)) ** 2


# if __name__ == '__main__':
#     h3_key_length = np.array([[0.73, 2.38],
#                               [0.73, 0.73],
#                               [2.38, 0.73]], requires_grad=False)
#
#     fix_coor = [[0, 0, 0, 0, 0, 0.73, 0, 0, 4.0],
#                 [0, 0, 0, 0, 0, 3.27, 0, 0, 4.0]]
#     symbols = ["H", "H", "H"]
#     ref_key = 6
#     frame_cir_depth = 1
#     qubit_num = 6
#     opt = 'sdg'
#     iter_num = 2
#     lr = 0.01
#     energy_method = 'vqe'
#
#     verbose = True
#     train_NEB_VQE(fix_coor,
#                   h3_key_length,
#                   ref_key,
#                   frame_cir_depth=frame_cir_depth,
#                   iter_num=iter_num,
#                   lr=lr,
#                   opt=opt,
#                   energy_method=energy_method,
#                   verbose=verbose)
