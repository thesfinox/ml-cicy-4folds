import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()


def ratio(x, y, base=(6,5)):
    '''
    Define the ratio of the plots.
    
    Required arguments:
        x: column ratio,
        y: row ratio.
        
    Optional arguments:
        base: 1x1 default ratio.
    '''
    
    return (base[0] * y, base[1] * x)
    
def savefig(filename, fig, root='./img', show=False, save_pdf=True, save_png=False):
    '''
    Save a Matplotlib figure to file.
    
    Required arguments:
        filename: the name of the saved file in the root directory (do not add an extension),
        fig:      the Matplotlib figure object.
        
    Optional arguments:
        root:     root directory,
        show:     show the plot inline (bool),
        save_pdf: save in PDF format,
        save_png: save in PNG format.
    '''
    
    # save the figure to file (PDF and PNG)
    fig.tight_layout()
    if save_pdf:
        os.makedirs(root, exist_ok=True)
        fig.savefig(os.path.join(root, filename + '.pdf'), dpi=144, format='pdf')
    if save_png:
        os.makedirs(root, exist_ok=True)
        fig.savefig(os.path.join(root, filename + '.png'), dpi=144, format='png')
    
    # show if interactive
    if show:
        plt.show()
    
    # release memory
    fig.clf()
    plt.close(fig)
    
    
def plot_univariate(data,
                    out_name=None,
                    root='.',
                    binwidth=None,
                    title=None,
                    xlabel=None,
                    ylabel=None,
                    yscale=None,
                    colour='C0',
                    stat='count',
                    base_ratio=(1,1),
                    subplots=None,
                    return_ax=False,
                    **kwargs
                   ):
    '''
    Plot a univariate distribution as histogram.
    
    Required arguments:
        data: the data to plot (can be a dict if multiple series on the same plot).
    
    Optional arguments:
        out_name:   name of the ouput (without extension),
        root:       root of the save location,
        binwidth:   width of the bins in the histogram,
        title:      title of the plot,
        xlabel:     x-axis label,
        ylabel:     y-axis label,
        yscale:     y-axis scale,
        colour:     the colour of the distribution,
        stat:       see Seaborn documentation (https://seaborn.pydata.org/generated/seaborn.histplot.html#seaborn.histplot),
        base_ratio: ratio of the output plot (tuple),
        subplots:   pass a tuple of (figure, axis),
        return_ax:  return the axis object if requested,
        **kwargs:   additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
        
    E.g.:
    
        data = {'training': ...,
                'validation': ...,
                'test': ...
               }
               
        The keys of the dictionary will then be used as legend.
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(Y,X))

    # handle the case in which data is divided into training, validation, test
    if isinstance(data, dict):
        
        # ensure colours are correctly mapped
        palette = sns.color_palette(None, n_colors=len(list(data.keys())))
        colour  = {key: palette[n] for n, key in enumerate(data.keys())}
        
        if 'training' in data.keys():
            colour['training'] = 'tab:blue'
        if 'validation' in data.keys():
            colour['validation'] = 'tab:red'
        if 'test' in data.keys():
            colour['test'] = 'tab:green'
            
        # for each dataset in data plot a histogram
        for key in data.keys():
            sns.histplot(data=data[key],
                         binwidth=binwidth,
                         color=colour[key],
                         stat=stat,
                         alpha=0.3,
                         ax=ax
                        )    
        ax.set(title=title,
               xlabel=xlabel,
               ylabel=ylabel,
               yscale=yscale
              )
        ax.legend(list(data.keys()), bbox_to_anchor=(1.0, 0.0), loc='lower left')
        
    else:
        sns.histplot(data=data,
                     binwidth=binwidth,
                     color=colour,
                     stat=stat,
                     alpha=0.5,
                     ax=ax
                    )
        ax.set(title=title,
               xlabel=xlabel,
               ylabel=ylabel,
               yscale=yscale
              )

    if out_name is not None:
        savefig(out_name, fig, root=root, **kwargs)
        
    if return_ax:
        return ax
    

def plot_bivariate(x,
                   y,
                   data=None,
                   hue=None,
                   size=None,
                   out_name=None,
                   root='.',
                   title=None,
                   xlabel=None,
                   ylabel=None,
                   xscale=None,
                   yscale=None,
                   palette='tab10',
                   base_ratio=(1,1),
                   subplots=None,
                   return_ax=False,
                   **kwargs
                  ):
    '''
    Plot a univariate distribution as histogram.
    
    Required arguments:
        x:    series or name of the series in the x-axis,
        y:    series or name of the series in the y-axis.
    
    Optional arguments:
        data:       the dataframe with the data
        hue:        name of the series to use to distinguish colours,
        size:       name of the series to distinguish sizes.
        out_name:   name of the ouput (without extension),
        root:       root of the save location,
        title:      title of the plot,
        xlabel:     x-axis label,
        ylabel:     y-axis label,
        xscale:     x-axis scale,
        yscale:     y-axis scale,
        palette:    name of the colour palette,
        base_ratio: ratio of the output plot (tuple),
        subplots:   pass a tuple of (figure, axis),
        return_ax:  return the axis object if requested,
        **kwargs:   additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
        
    E.g.:
    
        data = {'training': ...,
                'validation': ...,
                'test': ...
               }
               
        The keys of the dictionary will then be used as legend.
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(Y,X))
    
    # handle the case in which data is a dictionary
    if isinstance(data, dict):
        
        # ensure colours are correctly mapped
        palette = sns.color_palette(palette, n_colors=len(list(data.keys())))
        colour  = {key: palette[n] for n, key in enumerate(data.keys())}
        
        if 'training' in data.keys():
            colour['training'] = 'tab:blue'
        if 'validation' in data.keys():
            colour['validation'] = 'tab:red'
        if 'test' in data.keys():
            colour['test'] = 'tab:green'
            
        for key in data.keys():
            sns.scatterplot(data=data[key],
                            x=x,
                            y=y,
                            hue=hue,
                            size=size,
                            color=colour[key],
                            alpha=0.3,
                            ax=ax
                           )
        ax.set(title=title,
               xlabel=xlabel,
               ylabel=ylabel,
              )
        ax.legend(list(data.keys()), bbox_to_anchor=(1.0, 0.0), loc='lower left')
        
    else:
        sns.scatterplot(data=data,
                        x=x,
                        y=y,
                        hue=hue,
                        size=size,
                        alpha=0.5,
                        ax=ax
                       )
        ax.set(title=title,
               xlabel=xlabel,
               ylabel=ylabel,
              )
        ax.legend(bbox_to_anchor=(1.0, 0.0), loc='lower left')

    if out_name is not None:
        savefig(out_name, fig, root=root, **kwargs)
        
    if return_ax:
        return ax
    

def plot_corr(data,
              out_name=None,
              root='.',
              title=None,
              cmap='RdBu_r',
              base_ratio=(1,1),
              subplots=None,
              return_ax=False,
              **kwargs
             ):
    '''
    Plot the correlation matrix.
    
    Required arguments:
        data: the data to plot.
    
    Optional arguments:
        out_name:   name of the ouput (without extension),
        root:       root of the save location,
        title:      title of the plot,
        cmap:       the color of the heatmap,
        base_ratio: ratio of the output plot (tuple),
        subplots:   pass a tuple of (figure, axis),
        return_ax:  return the axis object if requested,
        **kwargs:   additional arguments to pass to savefig.
        
    Returns:
        the axis (if requested).
    '''
    
    # plot the distribution
    if subplots is not None:
        fig, ax = subplots
    else:
        X, Y    = base_ratio
        fig, ax = plt.subplots(1, 1, figsize=ratio(Y,X))

    # compute correlations and plot
    corr_mat = data.corr()
    sns.heatmap(corr_mat,
                center=0.0,
                cmap='RdBu_r',
                ax=ax
               )
    ax.set(title=title,
           xticks=range(len(corr_mat.columns)),
           yticks=range(len(corr_mat.columns))
          )
    ax.set_xticklabels(corr_mat.columns, rotation=90, va='top', ha='left')
    ax.set_yticklabels(corr_mat.columns, va='top', ha='right')

    if out_name is not None:
        savefig(out_name, fig, root=root, **kwargs)
        
    if return_ax:
        return ax

    
def plot_loss(history,
              orient='index',
              out_name=None,
              root='.',
              validation=False,
              title='Loss Function',
              yscale=None,
              base_ratio=(1,1),
              **kwargs
             ):
    '''
    Plot the loss function.
    
    Required arguments:
        history: location of the JSON file containing the history of the training.
    
    Optional arguments:
        orient:     orientation of the JSON file (for Pandas),
        out_name:   name of the ouput (without extension),
        root:       root of the save location,
        validation: plot validation loss,
        title:      title of the plot,
        yscale:     y-axis scale,
        base_ratio: ratio of the output plot (tuple),
        **kwargs:   additional arguments to pass to savefig.
    '''
    
    # open JSON file
    hst = pd.read_json(history, orient=orient)
        
    # select data
    if validation:
        data    = hst[['loss', 'val_loss']]
        palette = ['tab:blue', 'tab:red']
    else:
        data    = hst['loss']
        palette = ['tab:blue']
    
    # plot the loss function
    X, Y    = base_ratio
    fig, ax = plt.subplots(1, 1, figsize=ratio(Y,X))

    sns.lineplot(data=data,
                 palette=palette,
                 ax=ax
                )
    ax.set(title=title,
           xlabel='epochs',
           ylabel='loss'
          )
    if yscale == 'log':
        ax.set_yscale('log')
        
    if validation:
        ax.legend(['training', 'validation'], bbox_to_anchor=(1.0, 0.0), loc='lower left')

    if out_name is not None:
        savefig(out_name, fig, root=root, **kwargs)
        

def plot_lr(history,
            orient='index',
            out_name=None,
            root='.',
            title='Learning Rate',
            yscale='log',
            base_ratio=(1,1),
            **kwargs
           ):
    '''
    Plot the learning rate.
    
    Required arguments:
        history: location of the JSON file containing the history of the training.
    
    Optional arguments:
        orient:     orientation of the JSON file (for Pandas),
        out_name:   name of the ouput (without extension),
        root:       root of the save location,
        title:      title of the plot,
        yscale:     y-axis scale,
        base_ratio: ratio of the output plot (tuple),
        **kwargs:   additional arguments to pass to savefig.
    '''
    
    # open JSON file
    hst = pd.read_json(history, orient=orient)
        
    data    = hst['lr']
    palette = ['tab:blue']
    
    # plot the loss function
    X, Y    = base_ratio
    fig, ax = plt.subplots(1, 1, figsize=ratio(Y,X))

    sns.lineplot(data=data,
                 palette=palette,
                 ax=ax
                )
    ax.set(title=title,
           xlabel='epochs',
           ylabel='loss'
          )
    if yscale == 'log':
        ax.set_yscale('log')
        
    if out_name is not None:
        savefig(out_name, fig, root=root, **kwargs)
        
        
def plot_metric(history,
                metric='mean_squared_error',
                orient='index',
                out_name=None,
                root='.',
                validation=False,
                title='Metric Function',
                ylabel='metric',
                yscale=None,
                base_ratio=(1,1),
                **kwargs
               ):
    '''
    Plot the loss function.
    
    Required arguments:
        history: location of the JSON file containing the history of the training.
    
    Optional arguments:
        metric:     the name of the metric to plot,
        orient:     orientation of the JSON file (for Pandas),
        out_name:   name of the ouput (without extension),
        root:       root of the save location,
        validation: plot validation loss,
        title:      title of the plot,
        yscale:     y-axis scale,
        base_ratio: ratio of the output plot (tuple),
        **kwargs:   additional arguments to pass to savefig.
    '''
    
    # open JSON file
    hst = pd.read_json(history, orient=orient)
        
    # select data
    if validation:
        data    = hst[[metric, 'val_' + metric]]
        palette = ['tab:blue', 'tab:red']
    else:
        data    = hst[metric]
        palette = ['tab:blue']
    
    # plot the loss function
    X, Y    = base_ratio
    fig, ax = plt.subplots(1, 1, figsize=ratio(Y,X))

    sns.lineplot(data=data,
                 palette=palette,
                 ax=ax
                )
    ax.set(title=title,
           xlabel='epochs',
           ylabel=ylabel
          )
    if yscale == 'log':
        ax.set_yscale('log')
        
    if validation:
        ax.legend(['training', 'validation'], bbox_to_anchor=(1.0, 0.0), loc='lower left')

    if out_name is not None:
        savefig(out_name, fig, root=root, **kwargs)