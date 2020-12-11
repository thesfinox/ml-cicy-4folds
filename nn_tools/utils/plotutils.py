import os
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

def py_to_tex(label):
    '''
    Print in TeX form the label.
    
    Needed argument:
        label: the string containing the Hodge number label to print in TeX format.
    '''
    
    return f'$h^{{ {label[1]}, {label[2]} }}$'


def ratio(x, y, base=(6,5)):
    '''
    Define the ratio of the plots.
    
    Needed arguments:
        x: column ratio,
        y: row ratio.
        
    Optional arguments:
        base: 1x1 default ratio.
    '''
    
    return (base[0] * y, base[1] * x)

    
def savefig(filename, fig, root='./img', show=False):
    '''
    Save a Matplotlib figure to file.
    
    Needed arguments:
        filename: the name of the saved file in the root directory (do not add an extension),
        fig:      the Matplotlib figure object.
        
    Optional arguments:
        root: root directory,
        show: show the plot inline (bool).
    '''
    
    # save the figure to file (PDF and PNG)
    fig.tight_layout()
    os.makedirs(root, exist_ok=True)
    fig.savefig(os.path.join(root, filename + '.pdf'), dpi=144, format='pdf')
    fig.savefig(os.path.join(root, filename + '.png'), dpi=144, format='png')
    
    # show if interactive
    if show:
        plt.show()
    
    # release memory
    fig.clf()
    plt.close(fig)