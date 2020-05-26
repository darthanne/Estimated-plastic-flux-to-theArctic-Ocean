import scipy.io
import scipy.sparse



# P-Matrices
P1 = scipy.io.loadmat('P1.mat')['P1'].tocsr()
P2 = scipy.io.loadmat('P2.mat')['ans'].tocsr()
P3 = scipy.io.loadmat('P3.mat')['ans'].tocsr()
P4 = scipy.io.loadmat('P4.mat')['ans'].tocsr()
P5 = scipy.io.loadmat('P5.mat')['ans'].tocsr()
P6 = scipy.io.loadmat('P6.mat')['ans'].tocsr()
P = [P1, P2, P3, P4, P5, P6]

# Influx to arctic circle part of P matrices
F1 = scipy.sparse.csr_matrix(P1.shape).tolil()
print('F1 init')
F2 = scipy.sparse.csr_matrix(P1.shape).tolil()
print('F2 init')
F3 = scipy.sparse.csr_matrix(P1.shape).tolil()
print('F3 init')
F4 = scipy.sparse.csr_matrix(P1.shape).tolil()
print('F4 init')
F5 = scipy.sparse.csr_matrix(P1.shape).tolil()
print('F5 init')
F6 = scipy.sparse.csr_matrix(P1.shape).tolil()
print('F6 init')
F1[:56316, 56317:] = P1.tolil()[:56316, 56317:]
print('F1 done')
F2[:56316, 56317:] = P3[:56316, 56317:].tolil()
print('F2 done')
F3[:56316, 56317:] = P3[:56316, 56317:].tolil()
print('F3 done')
F4[:56316, 56317:] = P4[:56316, 56317:].tolil()
print('F4 done')
F5[:56316, 56317:] = P5[:56316, 56317:].tolil()
print('F5 done')
F6[:56316, 56317:] = P6[:56316, 56317:].tolil()
print('F6 done')

F = {'F1': F1, 'F2': F2, 'F3': F3, 'F4': F4, 'F5': F5, 'F6': F6}
scipy.io.savemat('F.mat', F)
# Outflux from arctic circle part of P matrices

E1 = scipy.sparse.csr_matrix(P1.shape).tolil()
E2 = scipy.sparse.csr_matrix(P1.shape).tolil()
E3 = scipy.sparse.csr_matrix(P1.shape).tolil()
E4 = scipy.sparse.csr_matrix(P1.shape).tolil()
E5 = scipy.sparse.csr_matrix(P1.shape).tolil()
E6 = scipy.sparse.csr_matrix(P1.shape).tolil()
E1[56316:, :56317] = P1[56316:, :56317].tolil()
print('F6 done')
E2[56316:, :56317] = P2[56316:, :56317].tolil()
print('F6 done')
E3[56316:, :56317] = P3[56316:, :56317].tolil()
print('F6 done')
E4[56316:, :56317] = P4[56316:, :56317].tolil()
print('F6 done')
E5[56316:, :56317] = P5[56316:, :56317].tolil()
print('F6 done')
E6[56316:, :56317] = P6[56316:, :56317].tolil()

E = {'E1': E1, 'E2': E2, 'E3': E3, 'E4': E4, 'E5': E5, 'E6': E6}
scipy.io.savemat('E.mat', E)
