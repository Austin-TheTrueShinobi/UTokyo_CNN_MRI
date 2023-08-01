from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

#This will set the projection type of the rendered plot.
#In order for this to be accurate you need to also specify what
#contour reflection type you want as well as the gradient
ax = plt.figure().add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.05)

# Plot the 3D surface
ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=9, cstride=5,
                alpha=0.8)

# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=-90, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-50, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='Network Response (X)', ylabel='Network Response (Y)', zlabel='Function Depth (Gradient Distance)')

plt.show()              