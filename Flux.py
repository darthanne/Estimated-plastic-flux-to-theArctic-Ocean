from IM import*
from FindShore import isShore

#
# Inputs
#

# P-Matrices
P1 = scipy.io.loadmat('P1.mat')['P1'].tocsr()
P2 = scipy.io.loadmat('P2.mat')['ans'].tocsr()
P3 = scipy.io.loadmat('P3.mat')['ans'].tocsr()
P4 = scipy.io.loadmat('P4.mat')['ans'].tocsr()
P5 = scipy.io.loadmat('P5.mat')['ans'].tocsr()
P6 = scipy.io.loadmat('P6.mat')['ans'].tocsr()
P = [P1, P2, P3, P4, P5, P6]

# Land
landpoints = np.array(scipy.io.loadmat('landpoints.mat')['landpoints'])

# Create shoreline file
""" 
# Find shoreline as any land with adjacent water
shoreline = np.zeros(65341, dtype=bool)
shoreline = shoreline.reshape((181, 361))

for y in range(181):
    for x in range(361):
        shoreline[y, x] = isShore(landpoints, y, x)
"""

# Import from file
shoreline = np.array(pd.read_csv("shoreline.txt", header=None))

# Flip up/down for plot reasons
landpoints = landpoints.reshape((181, 361))
landpoints = np.flipud(landpoints)
shoreline = np.flipud(shoreline)
shoreline = np.abs(shoreline - landpoints)

shoreline = np.flipud(shoreline)
shoreline = np.roll(shoreline, shift=180)
# Grid
lats = np.array(scipy.io.loadmat('lats.mat')['lats'][0][0][0])
lons = np.array(scipy.io.loadmat('lons.mat')['lons'][0][0][0])

# S matrix test
s = np.random.rand(65341) * 0.001
s = scipy.sparse.csr_matrix(s)
s = s.todense().reshape((181, 361))

# Corrections
landpoints[:10, :] = 1
landpoints[:25, 95:270] = 1

s = np.flipud(ma.masked_where(landpoints == 1, s))

s = scipy.sparse.csr_matrix(s.reshape(65341))


# Setup for animation
def matrixStep(i, c):
    return c @ P[i % 6]


cmap = plt.cm.get_cmap('RdBu_r', 28)

mats = np.array([s])
its = 800

PFlux = []
PTot = []
ims = []

for i in range(its):
    mats = np.append(mats, matrixStep(i, mats[i]))
    mats[i] = mats[i].reshape((181, 361))

    # Calculate Flux to/from arctic for timestep
    PFlux.append(
        np.sum(mats[i].todense().reshape((181, 361))[156:, :] - mats[i - 1].todense().reshape((181, 361))[156:, :]))

    # Total plastic at that time
    PTot.append(np.sum(mats[i].todense()))

    # Start plot from 2nd time step
    if i > 0:
        # Print progress
        if i % int(its / 10) == 0:
            print(int(i / its * 100), "%")

        if its - i == 1:
            print('100%')
            print('Generating animation...')

plt.show()

deltaFlux = []
sumFlux = []

for i in range(10, its):
    if (i + 1) % 6 == 0:
        sumFlux.append(np.sum(PFlux[i - 6:i]))
        if i > 14:
            deltaFlux.append(sumFlux[int((i + 1) / 6) - 2] - sumFlux[int((i + 1) / 6) - 3])

fig = plt.figure(figsize=(4, 3))
plt.subplot()
ax = plt.axes()

ax.add_patch(patches.Rectangle((0, -10000), 10, 60000, facecolor='0.8', hatch="//", edgecolor='0.7'))
plt.plot([10, 10], [-10000, 50000], color='gray', linestyle='dotted', linewidth=1)
plt.plot([0, 30], [0, 0], color='gray', linestyle='dashed', linewidth=1)
plt.plot(np.arange(2, 133), deltaFlux, color='0.1')
plt.xlim(0, 30)
plt.ylim(-0.03, 0.03)
plt.xticks(np.arange(0, 40, 10), np.arange(2010, 2050, 10), rotation=45, )
plt.xlabel('Years since start of simulation')
plt.ylabel("Change in flux")
plt.minorticks_on()
plt.savefig("Fluxchangerandom.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
