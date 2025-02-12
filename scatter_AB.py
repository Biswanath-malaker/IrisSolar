import numpy as np
import matplotlib.pyplot as plt


def scatterAB(A,B,ax):

    # For each row and column
    
    for row in range(s[0]):
        for col in range(s[1]):
            # Create color based on row number
            color = plt.cm.viridis(row / s[0])
            
            # Plot each point
            ax.scatter(A[row, col], B[row, col], 
                    c=[color], 
                    alpha=0.6)

    # Add colorbar
    norm = plt.Normalize(0, s[0])
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Row Number')

def scatterAB_p(A,B,ax,ext_y):

    s = A.shape
    COLOR = np.array(list(np.linspace(ext_y[0],ext_y[1],s[0]))*s[1]).reshape(s[::-1]).T.reshape((s[0]*s[1]))
    A_modified =  A.reshape((s[0]*s[1],))
    B_modified = B.reshape((s[0]*s[1],))


    im = ax.scatter(A_modified,B_modified,c=COLOR)
    return im
    
    
if __name__ == "__main__":
    # Example data generation (you can replace this with your actual arrays A and B)
    A = np.linspace(1,100,100).reshape((25,4))
    B = np.linspace(1,10,100).reshape((25,4))
    # Create a figure and axis with space for colorbar
    fig, ax = plt.subplots(figsize=(12, 8))
    # Add labels and title
    ax.set_xlabel('Array A Values')
    ax.set_ylabel('Array B Values')
    ax.set_title('Scatterplot of Arrays A vs B\nColors represent row numbers')
    
    scatterAB_p(A,B,ax,[10,50])

    # Show the plot
    # plt.tight_layout()
    plt.show()