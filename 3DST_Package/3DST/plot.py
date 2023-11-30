import k3d
import os
from typing import Optional
import numpy as np
import pyvista as pv
import anndata as ad
from scipy.sparse import issparse
import matplotlib.pyplot as plt


############# plot_mesh ####################
def plot_mesh(h5ad_path:str,
            mesh_path:str,
            mesh_anno:list,
            color_anno:str='color_anno',
            scale_size:list=[1.1,1.1,1.1],
            color_map:Optional[dict] = None,  
            
):
    """
    Plot mesh of organization.

    Args:   
        h5ad_path: path of h5ad.
        mesh_path: path of mesh object.
        mesh_anno: list of annotation to plot. 
        scale_size: Enlarge the size of the outer(whole tissue) outline,
        color_anno:'str'. key in .uns[],contain color to annotation.
        color_map:  a dictionary that map group names to group Hexadecimal colors,contain 'all', optional.
        

    Returns:
        Plot mesh of organization.
    """
    ##
    if color_map is None:
        color_map = _get_color_map(h5ad_path,color_anno)
    
    ## plot
    mesh_files = [mesh_path+'/mesh_{}.obj'.format(name).replace(' ', '_') for name in mesh_anno]
    
    plot = k3d.plot()
    for i,(name,mesh_file) in enumerate(zip(mesh_anno, mesh_files)):
        pv_mesh = pv.read(mesh_file).triangulate()
        if name=='all':
            pv_mesh = pv_mesh.scale([1.1, 1.1, 1.1])
        pv_faces = np.asarray(pv_mesh.faces).reshape((pv_mesh.n_faces, -1))[:, 1:]
        plt_mesh = k3d.mesh(vertices=pv_mesh.points,
                            indices=pv_faces,
                            color=int(color_map[name]),     
                            opacity=0.3,
                            name=name,)  
        plot += plt_mesh                 

    return plot
    
    
def _get_color_map(h5ad_path:str,
                  color_anno:str='color_anno',
):
    """
       Get color_map from h5ad.
    """
    file_names = [f for f in os.listdir(h5ad_path) if f.endswith('.h5ad')]
    file_path = os.path.join(h5ad_path, file_names[0])  
    
    adata = ad.read(file_path)
    color_map = adata.uns[color_anno]
    del adata
    annotation = list(color_map.keys())
    colors = np.array(list(color_map.values()))
    color_arr = np.array([int(c[1:], 16) for c in colors])
    color_map = dict(zip(annotation, color_arr))
    color_map['all'] = 14211277
    color_map.values
    
    return color_map

############ plot_points_inside_mesh ############
def plot_points_inside_mesh(points:np.array,
                            mesh_path:str,
                            point_size:float=0.1,
):
    """
    Plot cloud points inside mesh.

    Args:   
        points (np.array): The points to plot. Shape should be (n, 3).
        mesh_path (str): The path to the mesh file. Should be an '.obj'.
        
    Returns:
        Plot mesh and points.
    """
    ## read_mesh
    mesh = pv.read(mesh_path).triangulate()
    faces = np.asarray(mesh.faces).reshape((mesh.n_faces, -1))[:, 1:]
    ## 
    plot = k3d.plot()
    plt_points = k3d.points(positions=points,
                            point_size= point_size,
                            opacity = 0.25,
                            shader='dot',
                            #color = 0,
                            name = 'points')
    
    plt_mesh = k3d.mesh(vertices=mesh.points,
                                indices=faces,
                                color=14211277,
                                opacity=0.3,
                                name='mesh')
    plot += plt_points
    plot += plt_mesh
    return plot

############ plot_gene ################
def plot_gene_exp(adata_list:list, 
                gene_name:str,
                layer=None, 
                spatial_key:str='spatial_regis',
                ps:float=0.75,   
                min_cutoff:float=1,
                mesh:bool=True,  
                mesh_path:Optional[str] = None,               
):  
    """
    Plot gene expression.

    Args:   
        adata_list: a list of adata. output of 'adata_list_input()'
        gene_name: gene name in adata
        layer:the layer of save gene expression, adata.layers[layer]
        spatial_key: The column key in .obsm, default is 'spatial_regis'
        ps:point_size
        min_cutoff:gene expression lower than the threshold are not displayed
        mesh:bool,defult is True, add mesh of all tissue
        mesh_path:path of whole tissue mesh object, (.obj),if mesh=True.s

    Returns:
        plot single gene expression.
    """
    ## 00. get gene expression
    point_cloud = np.empty((0,3), float)
    gene_exp = np.array([], dtype=float)

    for adata in adata_list:
        point_cloud = np.append(point_cloud, adata.obsm[spatial_key], axis=0)
        if gene_name not in adata.var_names:
            continue
        if layer is None:
            gene_exp = np.append(gene_exp, adata[:, gene_name].X.A.toarray().flatten(), axis=0) if issparse(adata.X) else np.append(gene_exp, adata[:, gene_name].X.toarray().flatten(), axis=0)
        else:
            gene_exp = np.append(gene_exp, adata[:, gene_name].layers[layer].A.flatten(), axis=0) if issparse(adata.layers[layer]) else np.append(gene_exp, adata[:,gene_name].layers[layer].flatten(), axis=0)
    gene_exp.max()
    vmin = gene_exp.min()
    vmax = gene_exp.max()

    ## 01. filter some points base on vmin vmax of gene exp
    opacities = []
    for attr in gene_exp:
        if attr <= min_cutoff:
            opacities.append(0)
        else:
            opacities.append(1.0)
    opacities

    ## 03. plot k3d
    plt_points = k3d.points(positions=point_cloud,
                            point_size= ps,
                            shader='3dSpecular',
                            opacities=opacities,
                            attribute=gene_exp,
                            color_range=[vmin, vmax],
                            name=gene_name)
    plot = k3d.plot()
    plot += plt_points
    
    ## 04. add whole mesh
    if mesh:
        pv_mesh = pv.read(mesh_path).triangulate()
        ### scale
        pv_mesh = pv_mesh.scale((1.2, 1.2, 1.2))
        pv_faces = np.asarray(pv_mesh.faces).reshape((pv_mesh.n_faces, -1))[:, 1:]
        ### k3d_plot
        plt_mesh = k3d.mesh(vertices=pv_mesh.points,
                            indices=pv_faces,
                            color=14211277,
                            opacity=0.3,
                            name='tissue_mesh')
        plot += plt_mesh
        # plot.display()
    return plot


############ plot_cloud_point ###############
def plot_cloud_point(adata_list:list, 
                    spatial_key:str='spatial',
                    anno:str='annotation',
                    color_anno:str='color_anno',
                    color_map:Optional[dict] = None,    
                    point_size:float=0.1,
):
    """
    Transform h5ad file into input format required by 3D analysis pipeline.

    Args:   
        adata_list: a list of adata. output of 'adata_list_input()'.
        spatial_key: The column key in .obsm, default is 'spatial'.
        anno: the column key/name that identifies the grouping information(for example, clusters that correspond to different cell types)of spots.
        color_anno: the key in .uns, corresponds to a dictionary that map group names to group colors. 
        color_map:  a dictionary that map group names to group Hexadecimal colors, optional.

    Returns:
        plot cloud point.
    """
    # create cloud point for input for rendering
    if color_map is None:
        color_map = adata_list[0].uns[color_anno]
    ## 01. hexadecimal conversion
    annotation = list(color_map.keys())
    colors = np.array(list(color_map.values()))
    color_arr = np.array([int(c[1:], 16) for c in colors], dtype=np.uint32)
    color_map = dict(zip(annotation, color_arr))

    ## 02.generate  
    pts_map = {}
    #color_list = []
    point_cloud = np.empty((0,3), float)

    for adata in adata_list:
        point_cloud = np.append(point_cloud, adata.obsm[spatial_key], axis=0)
        for i in range(len(adata.obs)):
            annot = adata.obs[anno][i]
            if annot not in pts_map:
                #color_map[anno] = color_id
                pts_map[annot] = np.empty((0,3), float)
                #color_id += 1
            #color_list.append(color_map[annot]) 
            pts_map[annot] = np.append(pts_map[annot], [adata.obsm[spatial_key][i]], axis=0)
    
    ## 03. plot k3d
    plot = k3d.plot()
    for key, val in sorted(pts_map.items()):
        plt_points = k3d.points(positions=val,
                                colors = [color_map[key]]*val.shape[0],
                                point_size=point_size,
                                shader='3dSpecular',
                                opacity=0.7,
                                name = key,
                               )
        plot += plt_points
    return plot

################### plot_fft_score ####################
def plot_fft_score(before:list,
                   regis:list,
                   title='Displacement deviation value',
                   ylabel= 'Dd value',
                   savefig = './dd_value.png',
                   figsize = (20,4),
                   cutoff = 10):
    ##
    x = list(range(0,len(before)))
    ##
    fig = plt.figure()
    #plt.plot(x, ground_truth, color='green',marker='o',label = 'ground_truth' )
    plt.plot(x, before, color='darkgrey',marker='o',label = 'before' )
    plt.plot(x, regis, color='red',marker='o',label = 'regis' )
    plt.axhline(y=cutoff, color='g', linestyle='--')

    ##
    plt.title(title)
    plt.xlabel('slice pair')
    plt.ylabel(ylabel)
    plt.xticks([i for i in range(0,len(x))], [i for i in range(0,len(x))])
    plt.legend(loc = (1.1,0.5))
    fig.set_size_inches(figsize)
    ##
    #plt.show()
    plt.savefig(savefig)