import matplotlib.pyplot as plt
import numpy as np

def dist(x1,y1,z1,x2,y2,z2):
    dist = np.sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
    return dist
def plot_deltaZ(z1,z2,z3,x,y,z):

    ax = plt.axes(projection='3d')

    x1 = -1
    x2 = 0.5
    x3 = 0.5

    y1 = 0
    y2 = 0.876
    y3 = -0.876


    # Data for a three-dimensional line
    zlin_ac1 = np.linspace(0, z1, 100)
    zlin_ac2 = np.linspace(0, z2, 100)
    zlin_ac3 = np.linspace(0, z3, 100)

    xlin_ac1 = np.ones(zlin_ac1.size)*x1
    xlin_ac2 = np.ones(zlin_ac2.size)*x2
    xlin_ac3 = np.ones(zlin_ac3.size)*x3

    ylin_ac1 = np.ones(zlin_ac1.size)*y1
    ylin_ac2 = np.ones(zlin_ac2.size)*y2
    ylin_ac3 = np.ones(zlin_ac3.size)*y3

    x_conn1 = np.zeros(100)
    y_conn1 = np.zeros(100)
    z_conn1 = np.zeros(100)
    i=0
    for alpha in np.linspace(0,1,100):
        pt = np.array([x1,y1,z1])+alpha*np.array([x-x1, y-y1, z-z1])
        x_conn1[i] = pt[0]
        y_conn1[i] = pt[1]
        z_conn1[i] = pt[2]
        i=i+1


    x_conn2 = np.zeros(100)
    y_conn2 = np.zeros(100)
    z_conn2 = np.zeros(100)
    i=0
    for alpha in np.linspace(0,1,100):
        pt = np.array([x2,y2,z2])+alpha*np.array([x-x2, y-y2, z-z2])
        x_conn2[i] = pt[0]
        y_conn2[i] = pt[1]
        z_conn2[i] = pt[2]
        i=i+1


    x_conn3 = np.zeros(100)
    y_conn3 = np.zeros(100)
    z_conn3 = np.zeros(100)
    i=0
    for alpha in np.linspace(0,1,100):
        pt = np.array([x3,y3,z3])+alpha*np.array([x-x3, y-y3, z-z3])
        x_conn3[i] = pt[0]
        y_conn3[i] = pt[1]
        z_conn3[i] = pt[2]
        i=i+1

    ax.plot3D(xlin_ac1, ylin_ac1, zlin_ac1, 'r')
    ax.plot3D(x_conn1, y_conn1, z_conn1, 'r')

    ax.plot3D(xlin_ac2, ylin_ac2, zlin_ac2, 'g')
    ax.plot3D(x_conn2, y_conn2, z_conn2, 'g')

    ax.plot3D(xlin_ac3, ylin_ac3, zlin_ac3, 'b')
    ax.plot3D(x_conn3, y_conn3, z_conn3, 'b')

    ax.scatter3D(x,y,z,c ='blue', s=50)
    ax.scatter3D(x1,y1,z1, c="limegreen")
    ax.scatter3D(x2,y2,z2, c="limegreen")
    ax.scatter3D(x3,y3,z3, c="limegreen")
    ax.scatter3D(x1,y1,0, c="limegreen")
    ax.scatter3D(x2,y2,0, c="limegreen")
    ax.scatter3D(x3,y3,0, c="limegreen")

    print("dist to 1: ", dist(x1,y1,z1,x,y,z))
    print("dist to 2: ", dist(x2,y2,z2,x,y,z))
    print("dist to 3: ", dist(x3,y3,z3,x,y,z))

    plt.show()

def inv_kin(x, y, z):
    max_circle_radius = 2
    center_dist = np.sqrt(x**2 + y**2)
    if(center_dist<max_circle_radius):
        pass
    else:
        x = x*max_circle_radius/np.sqrt(x**2 + y**2)
        y = y*max_circle_radius/np.sqrt(x**2 + y**2)
    z = z+6
    a = 1
    b = -2*z
    c = z**2 -36 + (-1-x)**2 + (0-y)**2
    z1 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)

    c = z**2 -36 + (0.5-x)**2 + (0.876-y)**2
    z2 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)

    c = z**2 -36 + (0.5-x)**2 + (-0.876-y)**2
    z3 = (-b - np.sqrt(b**2 -  4*a*c))/(2*a)
    print("in_func:",z1,z2,z3)
    z1,z2,z3 = np.clip([z1,z2,z3], 0.02, 9.98)
    return[z1,z2,z3]


if(__name__ == "__main__"):
    x = 0.4
    y = 0.7
    z = 5
    max_circle_radius = 1.5
    center_dist = np.sqrt(x**2 + y**2)
    if(center_dist<max_circle_radius):
        pass
    else:
        x = x*max_circle_radius/np.sqrt(x**2 + y**2)
        y = y*max_circle_radius/np.sqrt(x**2 + y**2)
    
    z1,z2,z3 = inv_kin(x,y,z)

    print(z1,z2,z3)
    print(x,y,z)
    plot_deltaZ(z1,z2,z3,x,y,z+6)
    pass