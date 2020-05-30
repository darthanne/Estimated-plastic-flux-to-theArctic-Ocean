from IM import *
import matplotlib.colors as colors
import scipy.io
import scipy.sparse
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.collections
from matplotlib.colors import ListedColormap
from matplotlib import animation
import matplotlib.cm as cm
from IPython import display

matplotlib.rcParams.update({
    'pgf.rcfonts': False,
})

# Importing shoreline matrix
shoreline = np.array(pd.read_csv("shoreline.txt", header=None))
shoreline = shoreline.reshape(shoreline.size)
landpoints = np.array(scipy.io.loadmat('landpoints.mat')['landpoints'])
landpoints = landpoints.reshape(landpoints.size)

shoreline = np.abs(shoreline - landpoints).reshape(181, 361)
# shoreline = np.flipud(shoreline)
shoreline = np.roll(shoreline, shift=180)
shoreline = shoreline.reshape(shoreline.size)

# Importing shapefiles - Costline and borders
shapename = 'admin_0_countries'
countries_shp = shpreader.Reader(shpreader.natural_earth(resolution='110m',
                                                         category='cultural', name=shapename))
countries_ids = np.array(list(countries_shp.records()))
countries_shp = np.array(list(countries_shp.geometries()))

for i in range(countries_ids.size):
    print(countries_ids[i].attributes['NAME_EN'], countries_ids[i].attributes)
# Creating center points of all nonempty matrix cells
xvals = np.zeros(np.sum(shoreline).astype(int))
yvals = np.array(xvals)
xvalsint = np.array(xvals)
yvalsint = np.array(xvals)
col = np.array(xvals)
name = np.empty(landpoints.shape, dtype='U256')

n_val = 0

for i in range(0, shoreline.size):

    # Determine if a point is a shorepoint
    if not shoreline[i]:

        # Assign x and y values for that point in two arrays
        xvals[n_val] = i % 361 + 0.5 - 180
        xvalsint[n_val] = i % 361
        yvals[n_val] = math.floor(i / 361) + 0.5 - 90
        yvalsint[n_val] = math.floor(i / 361)
        # Initialize distance above possible distance
        dist = 2.5
        name[i] = "NaN"
        col[n_val] = 0

        # Find shortest distance between current shorepoint and polygons and assign name
        # of the country with the shortest distance to that point
        for j in range(countries_shp.size):
            shps = countries_shp[j]

            # Case: country is of type Multipolygon
            if shps.geom_type == 'MultiPolygon':
                shp = np.array(list(shps))
            # Case: country is of type Polygon
            else:
                shp = np.array(list([shps]))

            for k in range(shp.size):
                cdist = shp[k].exterior.distance(Point(xvals[n_val], yvals[n_val]))

                # If distance from point to polygon is lower than current lowest distance to polygon, overwrite
                if cdist < dist:
                    dist = cdist

                    # Assign country name of polygon to cell
                    name[i] = countries_ids[j].attributes['NAME_LONG']
                    col[n_val] = j
        print(name[i])
        # print(dist)
        n_val += 1
"""
plt.scatter(xvals, yvals, s=1, c=col / np.max(col) + 0.1, cmap='prism')
n_val = 0
names_used = np.empty([])
for i in range(shoreline.size):
    if not shoreline[i]:
        if math.floor(i / 361) % 6 == 0:
            if not name[i] == "NaN" and not name[i] in names_used:
                plt.annotate(name[i], (xvals[n_val], yvals[n_val]))
                names_used = np.append(names_used, name[i])
        n_val += 1
plt.show()
plt.scatter(xvals, yvals, s=1, c=col / np.max(col), cmap='gist_ncar')
n_val = 0
names_used = np.empty([])
for i in range(shoreline.size):
    if not shoreline[i]:
        if math.floor(i / 361) % 6 == 0:
            if not name[i] == "NaN" and not name[i] in names_used:
                plt.annotate(name[i], (xvals[n_val], yvals[n_val]))
                names_used = np.append(names_used, name[i])
        n_val += 1
plt.show()
plt.scatter(xvals, yvals, s=1, c=col / np.max(col), cmap='tab20')
plt.show()
plt.scatter(xvals, yvals, s=1, c=col / np.max(col), cmap='flag')
plt.show()
"""

# Counting ocurencies of countrys coastline
count = pd.Series(list(name)).value_counts()

# Importing Jambeck plastic data
Jambeck = pd.read_excel(r'Jambeck_All.xlsx')
Countries = np.array(Jambeck['Country'])
Plastic_2010 = np.array(Jambeck['Mismanaged plastic waste in 2010(ton)'])
Plastic_2025 = np.array(Jambeck['Mismanaged plastic waste in 2025(ton)'])

# Creating Jambeck matrix
J = np.c_[Countries, Plastic_2010, Plastic_2025]

# Defining function to sort in Jambeck matrix
Jambeck_frame = pd.DataFrame(J, columns=['Country', '2010', '2025'])
print(Jambeck_frame)


def Jambeck(c, y):
    # Choosing 2010 or 2025 as the plastic input
    if y == 2010:
        df = Jambeck_frame[Jambeck_frame['Country'] == c]
        try:
            plastic = df['2010'].values[0]
        except IndexError:
            plastic = "NaN"
    elif y == 2025:
        df = Jambeck_frame[Jambeck_frame['Country'] == c]
        try:
            plastic = df['2025'].values[0]
        except IndexError:
            plastic = "NaN"
    else:
        plastic = "Country name or Year is wrong"
    return plastic


s2010 = np.zeros((181, 361), dtype=float)
s2025 = np.zeros((181, 361), dtype=float)

blanks = 0
# Create S matrices for 2010 and 2025

landpoints = np.flipud(np.array(scipy.io.loadmat('landpoints.mat')['landpoints']).reshape((181, 361)))

i = 0

for j in range(xvals.size):
    n = name[j]  # Country name

    # Skip loop iteration if name is NaN
    if n == 'NaN':
        continue

    # Skip loop iteration if name is blank
    if n == '':
        blanks += 1
        continue
    i += 1

    print(n)
    p1 = Jambeck(n, 2010)  # Plastic mismanaged of that country
    p2 = Jambeck(n, 2025)  # Plastic mismanaged of that country

    # Skip loop iteration if country is not present in Jambeck dataset
    if p1 == "NaN" or p2 == "NaN":
        print("No plastic for ", n)
        continue

    csn = count.loc[n]  # Country shore number
    x = int(xvalsint[i] + 181)
    x = x % 361
    y = int(181 - yvalsint[i])
    if x < 181:
        y -= 1

    print(x, y)

    # Create array of adjacent cells
    adj = np.zeros(8, dtype=bool)

    adj[0] = np.roll(landpoints, 1, axis=1)[y - 1, x]
    adj[3] = np.roll(landpoints, 1, axis=1)[y, x]
    adj[5] = np.roll(landpoints, 1, axis=1)[y + 1, x]

    adj[1] = landpoints[y - 1, x]

    landpoints[y, x] = 2

    adj[6] = landpoints[y + 1, x]

    adj[2] = np.roll(landpoints, -1, axis=1)[y - 1, x]
    adj[4] = np.roll(landpoints, -1, axis=1)[y, x]
    adj[7] = np.roll(landpoints, -1, axis=1)[y + 1, x]

    print(adj)
    # Subtract nr. of adjacent land cells from 8 to get total adjacent ocean cells
    aoc = 8 - adj.sum()

    # For each adjacent cell which is ocean, add p/(csn*aoc)
    pc1 = p1 / (csn * aoc)
    pc2 = p2 / (csn * aoc)
    # Seperated for optimization so that less roll operations have to be performed

    # Roll +1
    if not adj[0] or not adj[3] or not adj[5]:
        s2010 = np.roll(s2010, 1, axis=1)
        s2025 = np.roll(s2025, 1, axis=1)

        if not adj[0]:
            s2010[y - 1, x] += pc1
            s2025[y - 1, x] += pc2

        if not adj[3]:
            s2010[y, x] += pc1
            s2025[y, x] += pc2

        if not adj[5]:
            s2010[y + 1, x] = pc1
            s2025[y + 1, x] = pc2

        # Revert roll
        s2010 = np.roll(s2010, -1, axis=1)
        s2025 = np.roll(s2025, -1, axis=1)

    # Roll -1
    if not adj[2] or not adj[4] or not adj[7]:
        s2010 = np.roll(s2010, -1, axis=1)
        s2025 = np.roll(s2025, -1, axis=1)

        if not adj[2]:
            s2010[y - 1, x] += pc1
            s2025[y - 1, x] += pc2

        if not adj[4]:
            s2010[y, x] += pc1
            s2025[y, x] += pc2

        if not adj[7]:
            s2010[y + 1, x] += pc1
            s2025[y + 1, x] += pc2

        # Revert roll
        s2010 = np.roll(s2010, 1, axis=1)
        s2025 = np.roll(s2025, 1, axis=1)

    if not adj[1]:  # No roll
        s2010[y - 1, x] += pc1
        s2025[y - 1, x] += pc2

    if not adj[6]:  # No roll
        s2010[y + 1, x] += pc1
        s2025[y + 1, x] += pc2
"""
plt.matshow(landpoints)
plt.show()

plt.matshow(np.log10(s2010))
plt.show()

plt.matshow(np.log10(s2025))
plt.show()
"""

# P-Matrices
P1 = scipy.io.loadmat('P1.mat')['P1'].tocsr()
P2 = scipy.io.loadmat('P2.mat')['ans'].tocsr()
P3 = scipy.io.loadmat('P3.mat')['ans'].tocsr()
P4 = scipy.io.loadmat('P4.mat')['ans'].tocsr()
P5 = scipy.io.loadmat('P5.mat')['ans'].tocsr()
P6 = scipy.io.loadmat('P6.mat')['ans'].tocsr()
P = [P1, P2, P3, P4, P5, P6]

# Fraction of mismanaged plastic reaching ocean
oceanfactor = 0.15

# Fraction of floating plastic
floatfactor = 0.5

# Factor for plastic entering model
modelfactor = floatfactor * oceanfactor


# Setup for animation
def matrixStep(i, c):
    c = c @ P[i % 6]
    c += (s2010 / 6 + (s2025 - s2010) / (15 * 6) * i) * modelfactor
    return c


# Importing shapefiles - Costline and borders
GSHHS_shp = shapefile.Reader("GSHHS_c_L1.shp")
GSHHS_antarctica5 = shapefile.Reader("GSHHS_c_L5.shp")
GSHHS_antarctica6 = shapefile.Reader("GSHHS_c_L6.shp")

s2010 = np.flipud(s2010)
s2010 = s2010.reshape(65341)
s2010 = scipy.sparse.csr_matrix(s2010)

s2025 = np.flipud(s2025)
s2025 = s2025.reshape(65341)
s2025 = scipy.sparse.csr_matrix(s2025)

mats = np.array([s2010])
its = 800

# Animation
fig = plt.figure()
ax = plt.axes()

ims = []
PFlux = []
PTot = []

norm = colors.SymLogNorm(linthresh=1, vmin=-10000, vmax=10000, )
cmap = plt.cm.get_cmap('RdBu_r', 28)

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
        im = plt.imshow(
            np.roll(np.delete((np.flipud(mats[i].todense()) - np.flipud(mats[i - 1].todense())) * 6, -1, axis=1), 180,
                    axis=1),
            animated=True,
            norm=norm, cmap=cmap)
        ims.append([im])

        # Print progress
        if i % int(its / 10) == 0:
            print(int(i / its * 100), "%")

        if its - i == 1:
            print('100%')
            print('Generating animation...')

# Colorbar for animation
im_ratio = mats[0].shape[0] / mats[0].shape[1]
sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
sm._A = []  # Required for plotting for some reason
cbar = plt.colorbar(sm, fraction=0.047 * im_ratio, pad=0.04, norm=norm)
cbar.set_label("Flux in tonnes pr. year")


# Tickmarks for animation

# Tickmark degree sign cardinal direction
def add_degree_sign(input, axis):
    if axis == 'NS':
        if input < 0:
            output = str(abs(input)) + '$\degree$S'
        elif input > 0:
            output = str(input) + '$\degree$N'
        else:
            output = str(input) + '$\degree$'
    else:
        if int(abs(input)) == 180:
            output = str(abs(input)) + '$\degree$'
        elif input < 0:
            output = str(abs(input)) + '$\degree$W'
        elif input > 0:
            output = str(input) + '$\degree$E'
        else:
            output = str(input) + '$\degree$'
    return output


v_add_d_sign = np.vectorize(add_degree_sign)

ax.xaxis.set_major_locator(plt.MultipleLocator(60))
ax.xaxis.set_minor_locator(plt.MultipleLocator(10))

ax.yaxis.set_major_locator(plt.MultipleLocator(30))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

plt.xticks(np.arange(0, 390, 60), v_add_d_sign(np.arange(-180, 210, 60), 'WE'))
plt.yticks(np.arange(0, 210, 30), v_add_d_sign(np.flip(np.arange(-90, 120, 30)), 'NS'))

# Animating
# anim = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
# anim.save('basic_animation.mp4', fps=25, dpi=100, bitrate=4500)
plt.show()

# Plot first frame of 2015
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()
ax.set_aspect(2/1)

shoreline = shoreline.reshape((181, 361))

img = plt.imshow(
    np.roll(np.delete((np.flipud(mats[30].todense()) - np.flipud(mats[29].todense())) * 6, -1, axis=1), 180, axis=1),
    norm=norm,
    cmap=cmap)
plt.plot([0, 360], [90 - 66, 90 - 66], color='black', alpha=0.7, linestyle='dotted', linewidth=1)

ax.xaxis.set_major_locator(plt.MultipleLocator(60))
ax.xaxis.set_minor_locator(plt.MultipleLocator(10))

ax.yaxis.set_major_locator(plt.MultipleLocator(30))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

plt.xticks(np.arange(0, 390, 60), v_add_d_sign(np.arange(-180, 210, 60), 'WE'))
plt.yticks(np.arange(0, 210, 30), v_add_d_sign(np.flip(np.arange(-90, 120, 30)), 'NS'))

plt.xlim(0, 360)
plt.ylim(180, 0)

cbar = plt.colorbar(img, fraction=0.047 * im_ratio, pad=0.04, norm=norm)
cbar.set_label("Flux in tonnes pr. year")

# Coastline
for shape in GSHHS_shp.shapeRecords():
    x = np.array([i[0] for i in shape.shape.points[:]])
    x = x + 180

    y = np.array([i[1] for i in shape.shape.points[:]])
    y = y + 90

    plt.plot(x, 181 - y, color='black', linewidth=0.5)

# Antarctica
for shape in GSHHS_antarctica6.shapeRecords():
    x = np.array([i[0] for i in shape.shape.points[:]])
    x = x + 180

    y = np.array([i[1] for i in shape.shape.points[:]])
    y = y + 90

    plt.plot(x, 180 - y, color='black', linewidth=0.5)

plt.savefig("flux62015.pdf", bbox_inches='tight', pad_inches=0)
plt.show()

# Plot first frame of 2020
fig = plt.figure(figsize=(8, 4))
ax = plt.axes()

shoreline = shoreline.reshape((181, 361))

img = plt.imshow(
    np.roll(np.delete((np.flipud(mats[60].todense()) - np.flipud(mats[59].todense())) * 6, -1, axis=1), 180, axis=1),
    norm=norm,
    cmap=cmap)
plt.plot([0, 360], [90 - 66, 90 - 66], color='black', alpha=0.5, linestyle='dotted', linewidth=1)

ax.xaxis.set_major_locator(plt.MultipleLocator(60))
ax.xaxis.set_minor_locator(plt.MultipleLocator(10))

ax.yaxis.set_major_locator(plt.MultipleLocator(30))
ax.yaxis.set_minor_locator(plt.MultipleLocator(5))

plt.xticks(np.arange(0, 390, 60), v_add_d_sign(np.arange(-180, 210, 60), 'WE'))
plt.yticks(np.arange(0, 210, 30), v_add_d_sign(np.flip(np.arange(-90, 120, 30)), 'NS'))

plt.xlim(0, 360)
plt.ylim(180, 0)

cbar = plt.colorbar(img, fraction=0.047 * im_ratio, pad=0.04, norm=norm)
cbar.set_label("Flux in tonnes pr. year")

# Coastline
for shape in GSHHS_shp.shapeRecords():
    x = np.array([i[0] for i in shape.shape.points[:]])
    x = x + 180

    y = np.array([i[1] for i in shape.shape.points[:]])
    y = y + 90

    plt.plot(x, 181 - y, color='black', linewidth=0.5)

# Antarctica
for shape in GSHHS_antarctica6.shapeRecords():
    x = np.array([i[0] for i in shape.shape.points[:]])
    x = x + 180

    y = np.array([i[1] for i in shape.shape.points[:]])
    y = y + 90

    plt.plot(x, 180 - y, color='black', linewidth=0.5)

plt.savefig("flux62020.pdf", bbox_inches='tight', pad_inches=0)
plt.show()

PFlux = np.array(PFlux)
PTot = np.array(PFlux)


np.savetxt('Flux15.txt', PFlux, delimiter=',')

deltaFlux = []
sumFlux = []

for i in range(10, its):
    if (i + 1) % 6 == 0:
        sumFlux.append(np.sum(PFlux[i - 6:i]))
        if i > 14:
            deltaFlux.append(sumFlux[int((i + 1) / 6) - 2] - sumFlux[int((i + 1) / 6) - 3])

plt.plot(np.arange(1, 133), sumFlux)
plt.xlim(0, 30)
plt.show()

#
# Delta flux plot
#

fig = plt.figure(figsize=(4, 3))
plt.subplot()
ax = plt.axes()

ax.add_patch(patches.Rectangle((0, -10000), 10, 60000, facecolor='0.8', hatch="//", edgecolor='0.7'))
plt.plot([10, 10], [-10000, 50000], color='gray', linestyle='dotted', linewidth=1)
plt.plot(np.arange(2, 133), deltaFlux, color='0.1')
plt.xlim(0, 30)
plt.xticks(np.arange(0, 40, 10), np.arange(2010, 2050, 10), rotation=45, )
plt.xlabel('Year')
plt.ylabel("Change in flux (T Yr$^{-2}$)")
plt.ylim(-10000, 50000)
plt.minorticks_on()
plt.savefig("Fluxchange.pdf", bbox_inches='tight', pad_inches=0)
plt.show()
