import k3d
import os
from typing import Optional,Union
import numpy as np
# import pyvista as pv
import anndata as ad
# from scipy.sparse import issparse
# import matplotlib.pyplot as plt
import pandas as pd
from k3d.colormaps import matplotlib_color_maps
import scanpy as sc

############ plot_cloud_point ###############
def plot_cloud_point(adata: ad.AnnData, 
                     spatial_key: str='3d_align_spatial',
                     anno: str='region',
                     color_anno: str='color_anno',
                     color_map: Optional[dict] = None,
                     point_size: Union[float, list] = 20,
                     save_path: Optional[str] = None
):
    """
    Transform h5ad file into input format required by 3D analysis pipeline.

    Args:   
        adata_list: adata list which have been aligned.
        spatial_key: The column key in .obsm, default is '3d_align_spatial'.
        anno: gene name or obs name that identifies the grouping information(for example, clusters that correspond to different cell types)of spots.
        color_anno: the key in .uns, corresponds to a dictionary that map group names to group colors. 
        color_map:  a dictionary that map group names to group Hexadecimal colors, optional.

    Returns:
        plot cloud point.
    """
    
    position_s = adata.obsm['3d_align_spatial']
    color_s = [color_map[i] for i in adata.obs[anno].tolist()]
    point_size_s = [i**0.5 for i in adata.obs['area'].tolist()]
    
    plot = k3d.plot()
    # for i in range(len(point_size_s)):
    #     # c =
    #     plt_points = k3d.points(positions = position_s[i],
    #                             color = int(color_s[i][1:], 16),
    #                             point_size  = point_size_s[i],
    #                             shader='3dSpecular',
    #                             opacity = 1,
    #                             name = adata.obs[anno].tolist()[i],
    #                             )
    #     plot+=plt_points
    
    plt_points = k3d.points(positions = position_s,
                        # color = int(color_s[i][1:], 16),
                        colors = [int(i[1:], 16) for i in color_s],
                        point_sizes  = point_size_s,
                        shader='3dSpecular',
                        opacity = 1,
                        # name = adata.obs[anno].tolist()[i],
                        )
    plot+=plt_points
    
    with open(save_path,'w') as fp:
        fp.write(plot.get_snapshot())
    return plot


def plot_gene_cloud_point(adata: ad.AnnData, 
                          spatial_key: str='3d_align_spatial',
                          gene_name: str='region',
                          point_size: Union[float, list] = 20,
                          save_path: Optional[str] = None
):
    """
    Transform h5ad file into input format required by 3D analysis pipeline.

    Args:   
        adata_list: adata list which have been aligned.
        spatial_key: The column key in .obsm, default is '3d_align_spatial'.
        anno: gene name or obs name that identifies the grouping information(for example, clusters that correspond to different cell types)of spots.
        color_anno: the key in .uns, corresponds to a dictionary that map group names to group colors. 
        color_map:  a dictionary that map group names to group Hexadecimal colors, optional.

    Returns:
        plot cloud point.
    """
    adata = adata[:,adata.var_names==gene_name]
    adata = adata[adata.obs['region'] != 'meninges']
    sc.pp.filter_genes(adata,min_cells = 5)
    vals = adata.obsm[spatial_key]
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    f = adata.X.toarray().reshape(422035 )
    point_size_s = [i**0.5 for i in adata.obs['area'].tolist()]
    plt_points = k3d.points(positions=vals,
                            color_map=matplotlib_color_maps.Coolwarm,
                            point_sizes = point_size_s,
                            shader='3dSpecular',
                            # opacity=0.7,
                            opacities =  f + 0.3,
                            attribute=f,
                            )
    plot = k3d.plot()
    plot += plt_points
    with open(save_path,'w') as fp:
        fp.write(plot.get_snapshot())
    return plot


if __name__ == '__main__':
    # adata = ad.read_h5ad('C:/七鳃鳗/3d/lamprey_spateo.h5ad')
    adata = ad.read_h5ad('C:/七鳃鳗/3d/lamprey_3d.h5ad')
    # sregion = adata.obs['region'].tolist()
    # sregion = ['motor_nucleus_of_VII' if x=='motor_nucleus_of_Ⅶ' else x for x in region]
    # sregion = ['nucleus_of_X' if x=='nucleus_of_Ⅹ' else x for x in region]
    # adata.obs['region'] = region
    csv = pd.read_excel('C:/Users/zepoch/Downloads/brainRegion.xlsx')
    spot_color = {}
    for i in range(50):
        Brain_Region = csv['Brain_Region'][i].replace(' ','_')
        spot_color[Brain_Region] = '#'+str(csv['color'][i])
    
    # adata_all = []
    # for i in range(1,41):
    #     temp = adata[adata.obs['slices'] == str(i)]
    #     adata_all.append(temp)

    # adata_all[18].obsm['3d_align_spatial'] = np.array([adata_all[18].obsm['3d_align_spatial'][:,0], 
    #                                                       adata_all[18].obsm['3d_align_spatial'][:,1] - 200,
    #                                                       adata_all[18].obsm['3d_align_spatial'][:,2]]).T
    # adata_all[19].obsm['3d_align_spatial'] = np.array([adata_all[19].obsm['3d_align_spatial'][:,0], 
    #                                                       adata_all[19].obsm['3d_align_spatial'][:,1] - 200,
    #                                                       adata_all[19].obsm['3d_align_spatial'][:,2]]).T
    # for i in range(21, 40):
    #     adata_all[i].obsm['3d_align_spatial'] = np.array([adata_all[i].obsm['3d_align_spatial'][:,0], 
    #                                                       adata_all[i].obsm['3d_align_spatial'][:,1] - (40 * i),
    #                                                       adata_all[i].obsm['3d_align_spatial'][:,2]]).T
    # adata = sc.AnnData.concatenate(*adata_all)
    # adata.write('C:/七鳃鳗/3d/lamprey_3d.h5ad')
    # plot_gene_cloud_point(adata, gene_name = 'nbisL1-mrna-16506', save_path = 'C:/七鳃鳗/3d/gene_plt.html')
    # plot_cloud_point(adata, color_map = spot_color, save_path = 'C:/七鳃鳗/3d/plot_size.html')
    # csv = pd.read_excel('C:/七鳃鳗/3d/lamprey_brain_marker.xlsx')
    # for i in csv.geneID:
    #     try:
    #         plot_gene_cloud_point(adata, gene_name = i, save_path = 'C:/七鳃鳗/3d/lamprey_brain_marker/'+ i + '.html')
    #     except:
    #         continue
    # plot_gene_cloud_point(adata, gene_name = i, save_path = 'C:/七鳃鳗/3d/MSTRG.2359.html')
    import pandas as pd
    csv = pd.read_csv('C:/Users/zepoch/Downloads/hs_results.csv')
    gene_list = csv.Gene[:30].tolist()
    for i in gene_list:
        plot_gene_cloud_point(adata, gene_name = i, save_path = 'C:/七鳃鳗/SVG/' + i +'.html')