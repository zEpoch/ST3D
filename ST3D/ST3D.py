import k3d
from k3d import plot
import os
from typing import Optional,Union
import numpy as np
import anndata as ad
# from scipy.sparse import issparse
# import matplotlib.pyplot as plt
import pandas as pd
from k3d.colormaps import matplotlib_color_maps
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

class ST_3D:
    def __init__(self,
                 adata: ad.AnnData,):
        self.adata = adata
        pass
    
    def _get_position_label(self, position_label: str = '3d_align_spatial'):
        self.position_label = position_label
        assert position_label in self.adata.obsm.keys(), f'position_label {position_label} not in adata.obsm.keys'
        assert self.adata.obsm[position_label].shape[1] == 3, f'position_label {position_label} shape[1] != 3'
        assert self.adata.obsm[position_label].shape[0] == self.adata.shape[0], f'position_label {position_label} shape[0] != adata.shape[0]'
        
    def _get_annotation_label(self, annotation_label: str = 'region'):
        assert annotation_label in self.adata.obs.columns, f'annotation_label {annotation_label} not in adata.obs.columns'
        self.annotation_label = annotation_label
    
    def _get_color_annotation_dict_label(self, color_annotation_dict_label: str = 'color_anno'):
        assert color_annotation_dict_label in self.adata.uns.keys(), f'color_annotation_label {color_annotation_dict_label} not in adata.uns.keys'
        self.color_annotation_dict = self.adata.uns[color_annotation_dict_label].tolist()
    
    def _get_color_annotation_dict(self, color_annotation_dict: dict):
        self.color_annotation_dict = color_annotation_dict
    
    def _get_point_size(self, point_size: Union[float, list] = 20):
        self.point_size = point_size
    
    def _get_save_path(self, save_path: str):
        self.save_path = save_path
    

    def _generate_random_colors(self, n: int):
        return ['#' + ''.join(random.choices('0123456789ABCDEF', k=6)) for _ in range(n)]
    
    
    def plot_cloud_point(self,
                         color_annotation_dict: Optional[dict] = None,
                         save_path: str = './cloud_point.html',
                         position_label: str = '3d_align_spatial',
                         annotation_label: Optional[str] = None,
                         color_annotation_dict_label: Optional[str] = None,
                         point_size: Union[float, list] = 20,
                         ):
        
        if position_label:
            self._get_position_label(position_label)
        else:
            assert False, 'position_label is None'
        

        if annotation_label:
            self._get_annotation_label(annotation_label)
        else:
            self.annotation_label = None
        
        if color_annotation_dict:
            self._get_color_annotation_dict(color_annotation_dict)
        elif color_annotation_dict_label:
            self._get_color_annotation_dict_label(color_annotation_dict_label)
        else:
            self.color_annotation_dict = None
        
        if self.annotation_label and self.color_annotation_dict:

            annotation_labels = list(set(self.adata.obs[self.annotation_label]))
            plots = plot()
            for i in annotation_labels:
                adata_i = self.adata[self.adata.obs[self.annotation_label] == i].copy()
                color_i = self.color_annotation_dict[i]
                plt_points = k3d.points(positions = adata_i.obsm[self.position_label].astype('float32'),
                                        colors = [int(color_i[1:], 16)] * adata_i.shape[0],
                                        point_sizes  = point_size,
                                        shader='3dSpecular',
                                        opacity = 1,
                                        name = i)
                plots += plt_points
            
            with open(save_path,'w') as fp:
                fp.write(plots.get_snapshot())
        elif self.annotation_label:
            
            annotation_labels = list(set(self.adata.obs[self.annotation_label]))
            
            colors = self._generate_random_colors(len(annotation_labels)) 
            plots = plot()
            for i in annotation_labels:
                adata_i = self.adata[self.adata.obs[self.annotation_label] == i].copy()
                color_i = colors[annotation_labels.index(i)]
                print(color_i)
                plt_points = k3d.points(positions = adata_i.obsm[self.position_label].astype('float32'),
                                        colors = [int(color_i[1:], 16) for i in range(adata_i.shape[0])],
                                        point_sizes  = point_size,
                                        shader='3dSpecular',
                                        opacity = 1,
                                        name = i)
                plots += plt_points
            
            with open(save_path,'w') as fp:
                fp.write(plots.get_snapshot())
            
        else:
            plots = plot()

            plt_points = k3d.points(positions = self.adata.obsm[self.position_label].astype('float32'),
                                    colors = 8421504,
                                    point_sizes  = point_size,
                                    shader='3dSpecular',
                                    opacity = 1,)
            plots += plt_points
            with open(self.save_path,'w') as fp:
                fp.write(plots.get_snapshot())
        return plots
        

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
        with open(save_path,'w') as fp:
            fp.write(plot.get_snapshot())
        return plot
    
    
    
    def plot_gene_mesh(self, 
                       gene_name: str):
        pass
    
    def points_rotation(self, 
                        save_path: str, 
                        position_label: str = '3d_align_spatial',
                        annotation_label: Optional[str] = None,
                        color_annotation_dict_label: Optional[str] = None,
                        color_annotation_dict: Optional[dict] = None,
                        ):
        '''
        
        '''
        if position_label:
            self._get_position_label(position_label)
        else:
            assert False, 'position_label is None'
            
        if annotation_label:
            self._get_annotation_label(annotation_label)
        else:
            self.annotation_label = None
        
        if color_annotation_dict:
            self._get_color_annotation_dict(color_annotation_dict)
        elif color_annotation_dict_label:
            self._get_color_annotation_dict_label(color_annotation_dict_label)
        else:
            self.color_annotation_dict = None
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        points = self.adata.obsm[self.position_label]
        annotation_labels = self.adata.obs[self.annotation_label].tolist()
        
        if self.annotation_label and self.color_annotation_dict:
            colors = [self.color_annotation_dict[i] for i in annotation_labels]
        elif self.annotation_label:
            colors = self._generate_random_colors(len(annotation_labels))
        else:
            colors = None
        
        
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c = colors)

        x_max = max(points[:, 0])+ 0.5 * max(points[:, 0])
        y_max = max(points[:, 1])+ 0.5 * max(points[:, 1])
        z_max = max(points[:, 2])+ 0.5 * max(points[:, 2])
        
        # 更新函数
        def update(num, points, sc):
            ax.cla()  # 清除当前的轴

            # 计算旋转矩阵
            theta = np.radians(num)
            c, s = np.cos(theta), np.sin(theta)
            R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))

            # 旋转点云
            points_rot = np.dot(points, R.T)

            # 绘制旋转后的点云
            sc = ax.scatter(points_rot[:, 0], points_rot[:, 1], points_rot[:, 2])

            ax.set_xlim([-x_max, x_max])
            ax.set_ylim([-y_max, y_max])
            ax.set_zlim([-z_max, z_max])
            
            ax.set_axis_off()
            
            return sc

        # 创建动画
        ani = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 1), fargs=(points, sc), blit=True)
        writervideo = animation.PillowWriter(fps=180) 
        ani.save('increasingStraightLine.gif', writer=writervideo) 
        # ani.save('animation.mp4', writer='ffmpeg', fps=60)
        # 请注意，保存动画需要ffmpeg库。如果你没有安装这个库，你可以使用以下命令来安装
        # sudo apt-get install ffmpeg
        plt.show()

    def caculate_mesh(self,
                      mesh_file: str,
                      position_label: str = '3d_align_spatial',
                      annotation_label: Optional[str] = None,
                      color_annotation_dict_label: Optional[str] = None,
                      color_annotation_dict: Optional[dict] = None,
                      ):
        '''
        caculate mesh from 3d points
        
        '''
        
        
        
        
        pass
    
    
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