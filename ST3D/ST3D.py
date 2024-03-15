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

class ST_3D:
    def __init__(self,
                 adata: ad.AnnData,):
        self.adata = adata
        pass
    
    def _get_position_label(self, position_label: str = '3d_align_spatial'):
        self.postion = self.adata.obsm[position_label]
        
    def _get_annotation_label(self, annotation_label: str = 'region'):
        self.annotation = self.adata.obs[annotation_label].tolist()
    
    def _get_color_annotation_label(self, color_annotation_label: str = 'color_anno'):
        self.color_annotation = self.adata.obs[color_annotation_label].tolist()
    
    def _get_color_annotation(self, color_annotation: dict):
        self.color_annotation = color_annotation
    
    def _get_point_size(self, point_size: Union[float, list] = 20):
        self.point_size = point_size
    
    def _get_save_path(self, save_path: str):
        self.save_path = save_path
    
    def plot_cloud_point(self,
                         color_annotation: Optional[dict] = None,
                         save_path: str = './cloud_point.html',
                         position_label: str = '3d_align_spatial',
                         annotation_label: str = 'region',
                         color_annotation_label: str = 'color_anno',
                         point_size: Union[float, list] = 20,
                         ):
        
        position = self._get_position_label(position_label)
        
        

        color_s = [self.color_annotation[i] for i in self.adata.obs[annotation_label].tolist()]
        point_size_s = [i**0.5 for i in self.adata.obs['area'].tolist()]
        
        plot = k3d.plot()
        plt_points = k3d.points(positions = position_s,
                                colors = [int(i[1:], 16) for i in color_s],
                                point_sizes  = point_size_s,
                                shader='3dSpecular',
                                opacity = 1,
                                )
        plot+=plt_points
        with open(self.save_path,'w') as fp:
            fp.write(plot.get_snapshot())
        return plot

    def plot_gene_cloud_point(self, gene_name: str):
        adata = self.adata[:,self.adata.var_names==gene_name]
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
        with open(self.save_path,'w') as fp:
            fp.write(plot.get_snapshot())
        return plot
    
    def plot_gene_mesh(self, 
                       gene_name: str):
        pass
    
    def rotation():
        
        '''
        
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from mpl_toolkits.mplot3d import Axes3D

        # 创建一个3D图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 创建初始点云
        N = 100
        x = np.random.rand(N)
        y = np.random.rand(N)
        z = np.random.rand(N)
        sc = ax.scatter(x, y, z)

        # 更新函数
        def update(frame):
            ax.cla()  # 清除当前的轴
            x = np.random.rand(N)
            y = np.random.rand(N)
            z = np.random.rand(N)
            sc = ax.scatter(x, y, z)
            return sc,

        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2), blit=True)
        ani.save('animation.mp4', writer='ffmpeg', fps=30)
        # 请注意，保存动画需要ffmpeg库。如果你没有安装这个库，你可以使用以下命令来安装
        # sudo apt-get install ffmpeg
        plt.show()
        '''
        
        '''
        import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import trimesh

# 加载mesh文件
mesh = trimesh.load_mesh('your_mesh_file.stl')

# 创建一个3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 创建初始点云
N = 100
x = np.random.rand(N)
y = np.random.rand(N)
z = np.random.rand(N)
sc = ax.scatter(x, y, z)

# 添加mesh到图形
ax.add_collection3d(plt.mesh(mesh.vertices, mesh.faces))

# 更新函数
def update(frame):
    ax.cla()  # 清除当前的轴

    # 旋转点云
    x = np.cos(frame/10) * x - np.sin(frame/10) * y
    y = np.sin(frame/10) * x + np.cos(frame/10) * y
    sc = ax.scatter(x, y, z)

    # 旋转mesh
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(frame), (0, 0, 1)))
    ax.add_collection3d(plt.mesh(mesh.vertices, mesh.faces))

    return sc,

# 创建动画
ani = animation.FuncAnimation(fig, update, frames=range(100), blit=True)

plt.show()
        
        
        '''
        
        '''
        
        colors = np.random.rand(N, 3)  # 创建一个N x 3的数组来表示N个RGB颜色值
        sc = ax.scatter(x, y, z, c=colors)
        
        '''