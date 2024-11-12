
from matplotlib import cm, pyplot as plt


def plot_multicolor_line(x:np.array,y:np.array,c:np.array, limits):
    '''
    
    :param limits: needs to contain the keys: v_min,v_max,x_min,x_max,y_min,y_max
    '''
    
    col = cm.jet((c-limits['v_min'])/(limits['v_max']-limits['v_min']))
    ax = plt.gca()
    plt.gcf().set_size_inches(10,10)
    ax.set_xlim(limits['x_min'], limits['x_max'])
    ax.set_ylim(limits['y_min'], limits['y_max']) 
    for i in range(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i], y[i+1]], c=col[i])

    im = ax.scatter(x, y, c=c, s=0, cmap=cm.jet)
    return im