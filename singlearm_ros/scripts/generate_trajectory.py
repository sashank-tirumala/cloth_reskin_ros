import numpy as np 
import matplotlib.pyplot as plt
def cubic_spline_pt(p0, p1, m0, m1, t):
    """
    p0 - Initial point of trajectory
    p1 - Final point of trajectory
    m0 - Initial Slope of trajectory
    m1 - Final Slope of trajectory
    t - time (Only varies between zero and one)
    """
    if(t<0 or t>1):
        raise ValueError("t does not lie between [0,1]: %d", t)

    tcube = t**3
    tsquare = t**2
    p = (2*tcube-3*tsquare + 1)*p0 + (tcube - 2*tsquare + t)*m0 + (-2*tcube + 3*tsquare)*p1 + (tcube - tsquare)*m1
    return p


if(__name__ == "__main__"):
    
    times = np.arange(0,1,0.001)
    x_vals = []
    y_vals = []
    p0 =np.array([0,0])
    p1= np.array([1,0])
    m0 = -1
    m1 = -1
    for t in times:
        x,y = cubic_spline_pt(p0, p1, m0, m1, t)

        x_vals.append(x)
        y_vals.append(y)
    
    fig = plt.figure()
    plt.plot(x_vals,y_vals,"r", label="Generated trajectory")
    plt.legend()
    plt.show()