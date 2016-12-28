def plot_palette(pal, n=10):

    for key in pal:
        if isinstance(pal[key], mpl.colors.LinearSegmentedColormap):
            colors = pal[key](np.linspace(0,1,n))
            sns.palplot(sns.color_palette(colors))
        elif isinstance(pal[key], mpl.colors.Colormap):
            colors = pal[key](np.linspace(0,1,n))
            sns.palplot(sns.color_palette(colors))
        elif isinstance(pal[key], str):
            sns.palplot(sns.color_palette([pal[key]]))
        elif isinstance(pal[key], list):
            sns.palplot(sns.color_palette(pal[key]))
        plt.title(str(key))

def figure_palette(pid):
    
#     pal_kws = dict(start=0, light=.7, dark=.2)
#     palettes = dict(palA=sns.cubehelix_palette(4, rot=.2, **pal_kws),
#                     palB=sns.cubehelix_palette(4, rot=-.2, **pal_kws))
#     purple = "#6c4d87"
#     blue = "#476087"
#     brown = "#a6805f"
#     red = "#874858"
    
    # define a dict for each palette
    
    purple = dict(color_light="#9a91c4",
              color_dark="#5f486f",
              color_extra="0.2",
              color_contr1="#af8dc3",
              color_contr2="#7fbf7b",
              cmap=plt.cm.Purples
             )
    
    blue = dict(color_light="#72acd3",
              color_dark="#355d7a",
              color_extra="0.2",
              color_contr1="#67a9cf",
              color_contr2="#ef8a62",
              cmap=plt.cm.Blues,
              petal_cmap=plt.cm.Blues_r
             )
    
    red = dict(color_light="#f06855",
              color_dark="#8b312d",
              color_extra="0.2",
              color_contr1="#ef8a62",
              color_contr2="#999999",
              cmap=plt.cm.Reds
             )
    
    green = dict(color_light="#a6dba0",
              color_dark="#5aae61",
              color_contr1="#7fbf7b",
              color_contr2="#af8dc3",
              cmap=plt.cm.Greens
             )
    
    viridis = dict(color_light="#72acd3",
              color_dark="#355d7a",
              color_extra="0.2",
              cmap=plt.cm.viridis_r
             )
    
    bw = dict(color_light="0.4",
              color_dark="0.1",
              color_extra="0.2",
              color_contr1="0.7",
              color_contr2="0.1",
              cmap=plt.cm.gray_r,
              petal_cmap=plt.cm.gray
             )
    # see http://colorbrewer2.org/#type=diverging&scheme=PRGn&n=3
    final = dict(color_light="#5f486f",
                 color_dark="#355d7a",
                 color_extra="0.5",
                 color_contr1="#67a9cf",
                 color_contr2="#d6604d", # "#ef8a62",
                 color_pastel_green="#58b0a6",
                 color_pastel_blue="#6bacd0",
                 color_pastel_orange="#cfa255",
                 color_pastel_red="#e48065",
                 cmap=plt.cm.Blues,
                 petal_cmap=plt.cm.Spectral
             )
    
    palette = dict(blue=blue,
                   red=red,
                   purple=purple,
                   green=green,
                   viridis=viridis,
                   bw=bw,
                   final=final
              )
    
    return palette[pid]