import os
import gc
import re
import itertools
import math
import numpy as np
import pyvista as pv
import pymeshfix as mf
import pyacvd
import anndata
from typing import Optional
import mcubes

from sklearn.cluster import DBSCAN
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from .data_process import sort_file_names

def calculate_mesh(
              mesh_anno:list,
              arg:list,
              save_path:str,
              color_anno:str,
              spatial_key:str,
              celltype:str,  
              method:str='march_cubes',
              mesh_all:bool=False,
              h5ad_path:Optional[list] = None,
              adata_list:Optional[list] = None,
              sc_xyz:Optional[list] = None,
              color_map:Optional[dict] = None,       
):
    """
    Construct mesh model of inner or all tissue.

    Args:   
        h5ad_path:h5ad path.
        spatial_key: The column key in .obsm, default is 'spatial'.
        celltype: cell annotation in .obs.
        method: calculate mesh method,defult is march_cubes.
        mesh_anno: list,contain annotation. len(mesh_anno) = len(arg).
        arg:[[eps_val, min_samples, thresh_num,mc_scale_factor]]. This can be a mesh of one organization or multiple organizations,
                len(args)= number of organization want to show.
        mesh_all: defult is True, plot all organizations including the whole tissue.
        color_anno:'str'. key in .uns[],contain color to annotation.
        color_map:  a dictionary that map group names to group Hexadecimal colors, optional.
        

    Returns:
       .obj
    """
    if not os.path.exists(save_path):
            os.makedirs(save_path)
    if len(arg) != len(mesh_anno):
        raise ValueError(f"Each organ or cell type corresponds to a set of parameters. Please ensure the correspondence and equal length between arg and mesh_anno")

    ## 01. get all point cloud.
    xli, yli, zli, tyli,sc_xyz = read_and_parse(spatial_regis=spatial_key,anno=celltype,h5ad_path=h5ad_path,adata_list=adata_list,sc_xyz=sc_xyz)

    ## 02. calculate mesh of each or given celltype.
    tdg_li = {}
    for i,(x,y) in enumerate(zip(mesh_anno, arg)):
        ## calcute mesh
        # print(y[])
        tdg_li[x] = Mesh(xli, yli, zli, tyli, ty_name=x, eps_val=y[0], min_samples=y[1], thresh_num=y[2])
        mesh,_ = tdg_li[x].create_mesh_of_type(sc_xyz=sc_xyz,method=method, mc_scale_factor=y[3])
        ## uniform and smooth mesh
        mesh= tdg_li[x].uniform_re_mesh_and_smooth(mesh)
        ## add mesh
        pl = pv.Plotter()
        _ = pl.add_mesh(mesh)
        ## save .obj
        pl.export_obj(save_path + '/mesh_' + str(x).replace(' ', '_') + '.obj')
        #pl.export_obj(save_path+'/mesh_'+ str(x) +'.obj')
        

class Mesh():

    """ Reconstruct the mesh model based on the 3D point cloud."""

    def __init__(self, xli, yli, zli, tyli, ty_name, eps_val=1.5, min_samples=8, thresh_num=10):
        """
        Args:

            xli: List of x, with the same order of cells / bins as other iterable inputs
            yli: List of y, with the same order of cells / bins as other iterable inputs
            zli: List of z, with the same order of cells / bins as other iterable inputs
            tyli: List of type, with the same order of cells / bins as other iterable inputs
            ty_name: string of type being selected
            eps_val: In DBSCAN(Density-Based Spatial Clustering of Applications with Noise), the maximum distance between 
                    two samples for one to be considered as in the neighborhood of the other. Try increase this value and 
                    min_samples if a cell type region is splitted into too many subregions.
            min_samples: In DBSCAN, the number of samples (or total weight) in a neighborhood for a point to be
                            considered as a core point. This includes the point itself. Try increase this value and
                             eps_val if a cell type region is splitted into too many subregions. Try increase this value if
                             too many cells are excluded as outliers.
            thresh_num: minimum number of cells in a group to be considered to display a mesh

        """
        self.xli = xli
        self.yli = yli
        self.zli = zli
        self.tyli = tyli
        self.ty_name = ty_name
        self.eps_val = eps_val
        self.min_samples = min_samples
        self.thresh_num = thresh_num
        if ty_name == 'all':
            scatter_li = [{'x': np.array(self.xli), 'y': np.array(self.yli), 'z': np.array(self.zli), 'ty': 'all'}]
        else:
            scatter_li = self._select_split_and_remove_outlier(eps_val, min_samples, thresh_num)  # list, length equals to number of spatial clusters of this type

        self.scatter_li = scatter_li

    def create_mesh_of_type(self, 
                            sc_xyz:list,
                            alpha=None,
                            mc_scale_factor=None, 
                            levelset=None,
                            tol=None,
                            method: Literal["march_cubes","delaunay_3d"] = "march_cubes",):
        """
            Generate mesh for each element in self.scatter_li
        Args:
            self,
            method:  march_cubes, delaunay_3d
               'march', 'march_cubes'

               Method parameter among 'march' and 'march_cubes' indicates using Marching Cubes Algorithm to create mesh. Marching Cubes
               voxelize a point cloud and assigns 0 / 1 to each voxel/cube, then finds the iso face of each cube that
               intersects the edges of the cube into opposite classification. A mesh is formed by the iso faces.

               It is robust to noises and lackage of points, but may add extra mesh that doesn't exist physically.

                We recommend you to browse https://www.cs.carleton.edu/cs_comps/0405/shape/marching_cubes.html for visualization
                of this algorithm. Refer to its publication for more details: https://dl.acm.org/doi/pdf/10.1145/37402.37422

                -------------------------
                'delaunay', 'delaunay_3d'

                Method parameter value among 'delaunay' and 'delaunay_3d' indicates using Delaunay 3d Mesh Reconstruction
                to create mesh. It iteratively finds out the 3D triangulation of points with tedrahedra close
                to regular tedrahedra, using Bowyer-Watson algorithm, then extracts the surface to form a mesh. The
                output of the Delaunay triangulation is supposedly a convex hull, hence may surafce that 'overly wrap'
                points clouds, which idealy forms concave shapes.

                It is most idealized to be used in situations where the convex shape is known as prior knowledge.

                Refer to its publication for more details:
                https://www.kiv.zcu.cz/site/documents/verejne/vyzkum/publikace/technicke-zpravy/2002/tr-2002-02.pdf

            mc_scale_factor: scale_factor adpoted in the Marching Cubes method, so that mc_scale_factor times of
                            maximum neighbor-wise distance equals to width of one voxel
            levelset: If method in 'march' and 'march_cubes', this is the iso value when generating the iso surfaces.
            tol: If method in 'delaunay', 'delaunay_3d', cells smaller than this will be degenerated and merged.

        Returns: 
            mesh
        
        """

        mesh_li = []
        for scatter in self.scatter_li:
            if method in ['delaunay', 'delaunay_3d']:
                _args = {'alpha': 0, 'tol': 1.5}
                if alpha:
                    _args['alpha'] = alpha
                if tol:
                    _args['tol'] = tol
                mesh = self._create_mesh_delaunay(scatter, alpha=_args['alpha'], tol=_args['tol'])  # alpha=2, rel_tol=0.01

            elif method in ['march', 'march_cubes']:
                _args = {"levelset": 0, "mc_scale_factor": 1}
                if levelset:
                    _args['levelset'] = 0
                if mc_scale_factor:
                    _args['mc_scale_factor'] = mc_scale_factor

                mesh = self._create_mesh_march_cubes(scatter,
                                                     sc_xyz=sc_xyz,
                                                     mc_scale_factor=_args['mc_scale_factor'],
                                                     levelset=_args['levelset'])
            mesh_li.append(mesh)

        # post-process
        for i, mesh in enumerate(mesh_li):
            mesh = self._remove_duplicated_mesh(mesh)
            mesh = self._fix_mesh(mesh)
            mesh_li[i] = mesh

        mesh_all = self._merge_models(mesh_li)
        return mesh_all, mesh_li

    def uniform_re_mesh_and_smooth(self, mesh):
        """Get a smooth, uniformly meshed surface using voronoi clustering"""

        def unstruc_grid2polydata(grid):
            """
            e.g. grid = mesh.split_bodies(), polydata (mesh) = unstruc_grid2polydata(grid)
            :param grid:
            :return:
            """
            mesh = pv.wrap(grid.extract_surface())
            v = mesh.points

            # 修改face的底层数据,使之完全由三角形组成
            faces = mesh.faces
            if not mesh.is_all_triangles:
                tri_mesh = mesh.triangulate()
                faces = tri_mesh.faces
            f = np.ascontiguousarray(faces.reshape(-1, 4)[:, 1:])
            triangles = np.empty((f.shape[0], 4), dtype=pv.ID_TYPE)
            triangles[:, -3:] = f
            triangles[:, 0] = 3

            mesh = pv.PolyData(v, triangles, deep=False)
            return mesh

        def smooth_mesh(mesh, n_iter=100, **kwargs):
            """
            Adjust point coordinates using Laplacian smoothing.
            https://docs.pyvista.org/api/core/_autosummary/pyvista.PolyData.smooth.html#pyvista.PolyData.smooth

            Args:
                mesh: A mesh model.
                n_iter: Number of iterations for Laplacian smoothing.
                **kwargs: The rest of the parameters in pyvista.PolyData.smooth.

            Returns:
                smoothed_mesh: A smoothed mesh model.
            """

            smoothed_mesh = mesh.smooth(n_iter=n_iter, **kwargs)

            return smoothed_mesh

        uniform_surfs = []
        for sub_surf in mesh.split_bodies():
            sub_surf = unstruc_grid2polydata(sub_surf).triangulate().clean()
            # Get a smooth, uniformly meshed surface using voronoi clustering.
            sub_uniform_surf = self._uniform_re_mesh_single_fixed_voroni(
                mesh=sub_surf.extract_surface())  # sub_fix_surf.extract_surface()
            uniform_surfs.append(sub_uniform_surf)
        uniform_surf = self._merge_models(models=uniform_surfs)
        uniform_surf = uniform_surf.extract_surface().triangulate().clean()

        # post-process
        mesh = smooth_mesh(uniform_surf)
        return mesh

    @staticmethod
    def _create_mesh_delaunay(scatter, alpha, tol):
        xyz = np.concatenate([np.expand_dims(scatter['x'], axis=1),
                              np.expand_dims(scatter['y'], axis=1),
                              np.expand_dims(scatter['z'], axis=1)], axis=1)

        pdata = pv.PolyData(xyz)
        # if uniform_pc:
        #     pcd = self._uniform_larger_pc(pcd, type='o3d')

        mesh = pdata.delaunay_3d(alpha=alpha).extract_surface().triangulate().clean(tolerance=tol)
        if mesh.n_points == 0:
            raise ValueError(
                f"\nThe point cloud cannot generate a surface mesh with `pyvista` method and alpha == {alpha}."
            )
        return mesh

    def _create_mesh_march_cubes(self, scatter,sc_xyz, mc_scale_factor, levelset):

        def _scale_model_by_distance(
                model,
                distance=1,
                scale_center=None,
        ):

            # Check the distance.
            distance = distance if isinstance(distance, (tuple, list)) else [distance] * 3
            if len(distance) != 3:
                raise ValueError(
                    "`distance` value is wrong. \nWhen `distance` is a list or tuple, it can only contain three elements."
                )

            # Check the scaling center.
            scale_center = model.center if scale_center is None else scale_center
            if len(scale_center) != 3:
                raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

            # Scale the model based on the distance.
            for i, (d, c) in enumerate(zip(distance, scale_center)):
                p2c_bool = np.asarray(model.points[:, i] - c) > 0
                model.points[:, i][p2c_bool] += d
                model.points[:, i][~p2c_bool] -= d

            return model

        def _scale_model_by_scale_factor(
                model,
                scale_factor=1,
                scale_center=None,
        ):

            # Check the scaling factor.
            scale_factor = scale_factor if isinstance(scale_factor, (tuple, list)) else [scale_factor] * 3
            if len(scale_factor) != 3:
                raise ValueError(
                    "`scale_factor` value is wrong."
                    "\nWhen `scale_factor` is a list or tuple, it can only contain three elements."
                )

            # Check the scaling center.
            scale_center = model.center if scale_center is None else scale_center
            if len(scale_center) != 3:
                raise ValueError("`scale_center` value is wrong." "\n`scale_center` can only contain three elements.")

            # Scale the model based on the scale center.
            for i, (f, c) in enumerate(zip(scale_factor, scale_center)):
                model.points[:, i] = (model.points[:, i] - c) * f + c

            return model

        def scale_model(
                model,
                distance=None,
                scale_factor=1,
                scale_center=None,
                inplace=False,
        ):
            """
            Scale the model around the center of the model.

            Args:
                model: A 3D reconstructed model.
                distance: The distance by which the model is scaled. If `distance` is float, the model is scaled same distance
                          along the xyz axis; when the `scale factor` is list, the model is scaled along the xyz axis at
                          different distance. If `distance` is None, there will be no scaling based on distance.
                scale_factor: The scale by which the model is scaled. If `scale factor` is float, the model is scaled along the
                              xyz axis at the same scale; when the `scale factor` is list, the model is scaled along the xyz
                              axis at different scales. If `scale_factor` is None, there will be no scaling based on scale factor.
                scale_center: Scaling center. If `scale factor` is None, the `scale_center` will default to the center of the model.
                inplace: Updates model in-place.

            Returns:
                model_s: The scaled model.
            """

            model_s = model.copy() if not inplace else model

            if not (distance is None):
                model_s = _scale_model_by_distance(model=model_s, distance=distance, scale_center=scale_center)

            if not (scale_factor is None):
                model_s = _scale_model_by_scale_factor(model=model_s, scale_factor=scale_factor,
                                                       scale_center=scale_center)

            model_s = model_s.triangulate()

            return model_s if not inplace else None

        # def rigid_transform(
        #         coords,
        #         coords_refA,
        #         coords_refB,
        # ):
        #     """
        #     Compute optimal transformation based on the two sets of points and apply the transformation to other points.

        #     Args:
        #         coords: Coordinate matrix needed to be transformed.
        #         coords_refA: Referential coordinate matrix before transformation.
        #         coords_refB: Referential coordinate matrix after transformation.

        #     Returns:
        #         The coordinate matrix after transformation
        #     """
        #     # Check the spatial coordinates

        #     coords, coords_refA, coords_refB = coords.copy(), coords_refA.copy(), coords_refB.copy()
        #     assert (
        #             coords.shape[1] == coords_refA.shape[1] == coords_refA.shape[1]
        #     ), "The dimensions of the input coordinates must be uniform, 2D or 3D."
        #     coords_dim = coords.shape[1]
        #     if coords_dim == 2:
        #         coords = np.c_[coords, np.zeros(shape=(coords.shape[0], 1))]
        #         coords_refA = np.c_[coords_refA, np.zeros(shape=(coords_refA.shape[0], 1))]
        #         coords_refB = np.c_[coords_refB, np.zeros(shape=(coords_refB.shape[0], 1))]

        #     # Compute optimal transformation based on the two sets of points.
        #     coords_refA = coords_refA.T
        #     coords_refB = coords_refB.T

        #     centroid_A = np.mean(coords_refA, axis=1).reshape(-1, 1)
        #     centroid_B = np.mean(coords_refB, axis=1).reshape(-1, 1)

        #     Am = coords_refA - centroid_A
        #     Bm = coords_refB - centroid_B
        #     H = Am @ np.transpose(Bm)

        #     U, S, Vt = np.linalg.svd(H)
        #     R = Vt.T @ U.T

        #     if np.linalg.det(R) < 0:
        #         Vt[2, :] *= -1
        #         R = Vt.T @ U.T

        #     t = -R @ centroid_A + centroid_B

        #     # Apply the transformation to other points
        #     new_coords = (R @ coords.T) + t
        #     new_coords = np.asarray(new_coords.T)
        #     return new_coords[:, :2] if coords_dim == 2 else new_coords

        try:
            from scipy.spatial.distance import cdist
        except ImportError:
            raise ImportError('Need to install scipy')
        try:
            import mcubes
        except ImportError:
            raise ImportError('Need to install mcubes')

        # 1. data api: np.array to point cloud
        xyz = np.concatenate([np.expand_dims(scatter['x'], axis=1),
                              np.expand_dims(scatter['y'], axis=1),
                              np.expand_dims(scatter['z'], axis=1)], axis=1)
        print(xyz.shape)
        pc = pv.PolyData(xyz)

        # if uniform_pc:
        #     pc = self._uniform_larger_pc(pc, type='pv')

        raw_points = np.asarray(pc.points)
        print("raw",np.min(raw_points, axis=0))
        print("raw",np.max(raw_points, axis=0))
        new_points = raw_points - np.min(raw_points, axis=0)
        pc.points = new_points
        print("shift",np.min(new_points, axis=0))
        print("shift",np.max(new_points, axis=0))
        # 2. Preprocess: calculate, then apply scale_factor to the points
        dist = cdist(XA=new_points, XB=new_points, metric="euclidean")
        row, col = np.diag_indices_from(dist)
        dist[row, col] = None
        max_dist = np.nanmin(dist, axis=1).max()  # the maximum neighbor-wise distance
        mc_sf = max_dist * mc_scale_factor  # so that mc_scale_factor times of maximum neighbor-wise distance equals to width of one voxel
        print(max_dist,mc_sf)
        scale_pc = scale_model(model=pc, scale_factor=1/mc_sf)
        if np.any(np.min(np.asarray(scale_pc.points), axis=0)< 0):
            scale_pc.points = scale_pc.points - np.min(scale_pc.points, axis=0)

        scale_pc_points = np.ceil(np.asarray(scale_pc.points)).astype(np.int64)
        scale_pc.points = scale_pc_points

        print(np.min(np.asarray(scale_pc.points), axis=0))
        print(np.max(np.asarray(scale_pc.points), axis=0))
        # 3. Preprocess: generate volume for mesh generation, based on the points
        volume_array = np.zeros(
            shape=[
                scale_pc_points[:, 0].max() + 1,
                scale_pc_points[:, 1].max() + 1,
                scale_pc_points[:, 2].max() + 1,
            ]
        )
        volume_array[scale_pc_points[:, 0], scale_pc_points[:, 1], scale_pc_points[:, 2]] = 1
        print("shape:" ,volume_array.shape)
        # 4. Process: extract the iso-surface based on marching cubes algorithm.
        volume_array = mcubes.smooth(volume_array)
        vertices, triangles = mcubes.marching_cubes(volume_array, levelset)
        if len(vertices) == 0:
            raise ValueError(f"The point cloud cannot generate a surface mesh with `marching_cube` method.")
        v = np.asarray(vertices).astype(np.float64)
        f = np.asarray(triangles).astype(np.int64)
        f = np.c_[np.full(len(f), 3), f]
        mesh = pv.PolyData(v, f.ravel()).extract_surface().triangulate()
        mesh.clean(inplace=True)
        print("mesh0",np.min(np.asarray(mesh.points), axis=0))
        print("mesh0",np.max(np.asarray(mesh.points), axis=0))
        # 5. Post-process: scale the mesh model back to the points' original coordinates
        scale_pc_ori =scale_model(model=pc, scale_factor=1/mc_sf)
        if np.any(np.min(np.asarray(scale_pc_ori.points), axis=0)< 0):
            mesh.points = mesh.points + np.min(scale_pc_ori.points, axis=0)
            scale_pc.points = scale_pc.points + np.min(scale_pc_ori.points, axis=0)
        mesh = scale_model(model=mesh, scale_factor=mc_sf)
        scale_pc = scale_model(model=scale_pc, scale_factor=mc_sf)
        print("mesh1",np.min(np.asarray(mesh.points), axis=0))
        print("mesh1",np.max(np.asarray(mesh.points), axis=0))
        # 6. offset the mesh model back to the points' original coordinates
        points=mesh.points
        translation=np.min(raw_points, axis=0)
        mesh.points=points+translation
        print("mesh2",np.min(np.asarray(mesh.points), axis=0))
        print("mesh2",np.max(np.asarray(mesh.points), axis=0))
        # mesh.points = rigid_transform(
        #     coords=np.asarray(mesh.points), coords_refA=np.asarray(scale_pc.points), coords_refB=raw_points
        # )
        #mesh = scale_model(model=mesh, scale_factor=[1 / x for x in sc_xyz])
        points=mesh.points
        mesh.points[:,0]=points[:,0]*(1/sc_xyz[0])
        mesh.points[:,1]=points[:,1]*(1/sc_xyz[1])
        mesh.points[:,2]=points[:,2]*(1/sc_xyz[2])
        print("mesh3",np.max(np.asarray(mesh.points), axis=0))
        return mesh

    @staticmethod
    def _uniform_re_mesh_single_fixed_voroni(mesh, nsub=3, nclus=20000):
        """
        Generate a uniformly meshed surface using voronoi clustering.

        Args:
            mesh: A mesh model.
            nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
                  nface*4**nsub where nface is the current number of faces.
            nclus: Number of voronoi clustering.

        Returns:
            new_mesh: A uniform mesh model.
        """

        # if mesh is not dense enough for uniform remeshing, increase the number of triangles in a mesh.
        if not (nsub is None):
            mesh.subdivide(nsub=nsub, subfilter="butterfly", inplace=True)

        # Uniformly remeshing.
        clustered = pyacvd.Clustering(mesh)

        if not (nsub is None):
            clustered.subdivide(nsub=nsub)

        clustered.cluster(nclus)

        new_mesh = clustered.create_mesh().triangulate().clean()
        return new_mesh

    # fixme: not adopted yet for its high consumption of time and memory
    def _uniform_larger_pc(self, pc, alpha=0, nsub=3, nclus=20000, type='pv'):
        """
        Generates a uniform point cloud with a larger number of points.
        If the number of points in the original point cloud is too small or the distribution of the original point cloud is
        not uniform, making it difficult to construct the surface, this method can be used for preprocessing.

        Args:
            pc: A point cloud model.
            alpha: Specify alpha (or distance) value to control output of this filter.
                   For a non-zero alpha value, only edges or triangles contained within a sphere centered at mesh vertices
                   will be output. Otherwise, only triangles will be output.
            nsub: Number of subdivisions. Each subdivision creates 4 new triangles, so the number of resulting triangles is
                  nface*4**nsub where nface is the current number of faces.
            nclus: Number of voronoi clustering.

        Returns:
            new_pc: A uniform point cloud with a larger number of points.
        """
        coords = np.asarray(pc.points)
        coords_z = np.unique(coords[:, 2])

        slices = []
        for z in coords_z:
            slice_coords = coords[coords[:, 2] == z]
            slice_cloud = pv.PolyData(slice_coords)
            if len(slice_coords) >= 3:
                slice_plane = slice_cloud.delaunay_2d(alpha=alpha).triangulate().clean()
                uniform_plane = self._uniform_re_mesh_single_fixed_voroni(mesh=slice_plane, nsub=nsub, nclus=nclus)
                slices.append(uniform_plane)
            else:
                slices.append(slice_cloud)

        slices_mesh = self._merge_models(models=slices)
        if type in ['pv']:
            new_pc = pv.PolyData(slices_mesh.points).clean()
        return new_pc

    def _remove_duplicated_mesh(self, mesh):
        """Removes unused points and degenerate cells."""
        """Remove replicated meshes, can accept multiple bodies in a mesh, named as 'clean_mesh' in Spateo"""

        sub_meshes = mesh.split_bodies()
        n_mesh = len(sub_meshes)

        if n_mesh == 1:
            return mesh
        else:
            inside_number = []
            for i, main_mesh in enumerate(sub_meshes[:-1]):
                main_mesh = pv.PolyData(main_mesh.points, main_mesh.cells)
                for j, check_mesh in enumerate(sub_meshes[i + 1:]):
                    check_mesh = pv.PolyData(check_mesh.points, check_mesh.cells)
                    inside = check_mesh.select_enclosed_points(main_mesh, check_surface=False).threshold(0.5)
                    inside = pv.PolyData(inside.points, inside.cells)
                    if check_mesh == inside:
                        inside_number.append(i + 1 + j)

            cm_number = list(set([i for i in range(n_mesh)]).difference(set(inside_number)))
            if len(cm_number) == 1:
                cmesh = sub_meshes[cm_number[0]]
            else:
                cmesh = self._merge_models([sub_meshes[i] for i in cm_number])

            return pv.PolyData(cmesh.points, cmesh.cells)

    def _fix_mesh(self, mesh):
        """
        Repair the mesh where it was extracted and subtle holes along complex parts of the mesh.
        Example in https://pymeshfix.pyvista.org/examples/index.html
        While filling subtles holes in mesh, this process may over fill some of the areas, hence
        produce overly coarse results in non-CAD meshes.

        """

        def fix_single_mesh(mesh):
            meshfix = mf.MeshFix(mesh)
            meshfix.repair(verbose=False)
            fixed_mesh = meshfix.mesh.triangulate().clean()

            if fixed_mesh.n_points == 0:
                raise ValueError(
                    f"The surface cannot be Repaired. " f"\nPlease change the method or parameters of surface reconstruction."
                )
            return fixed_mesh

        fixed_surfs = []
        for sub_surf in mesh.split_bodies():
            # Repair the surface mesh where it was extracted and subtle holes along complex parts of the mesh
            sub_fix_surf = fix_single_mesh(sub_surf.extract_surface())
            fixed_surfs.append(sub_fix_surf)
        fixed_surfs = self._merge_models(models=fixed_surfs)
        # fixed_surfs = fixed_surfs.extract_surface().triangulate().clean()
        return fixed_surfs

    @staticmethod
    def _merge_models(models):
        """Merge all models in the `models` list. The format of all models must be the same."""

        merged_model = models[0]
        for model in models[1:]:
            merged_model = merged_model.merge(model)

        return merged_model

    @staticmethod
    def _split_and_remove_outlier_append(scatter_li, x_arr, y_arr, z_arr, ty,
                                         eps_val, min_samples, thresh_num):

        X = np.concatenate([np.expand_dims(x_arr, axis=1),
                            np.expand_dims(y_arr, axis=1),
                            np.expand_dims(z_arr, axis=1)], axis=1)  # (n,3)

        # 2. re-grouping based on spatial coordinates
        dbscan_clus = DBSCAN(eps=eps_val, min_samples=min_samples).fit(X)
        dbscan_labels_arr = dbscan_clus.labels_  # (n_spot_in_this_cluster,)

        # 3. process data into input required by volume visualization, based on re-grouping results
        for dbscan_label in set(dbscan_labels_arr):
            # get rid of outliers
            if dbscan_label == -1:
                continue
            grp_idx = np.where(dbscan_labels_arr == dbscan_label)
            # print(dbscan_label, grp_idx[0].shape)
            if grp_idx[0].shape[0] < thresh_num:
                continue
            sin_vol = {'x': x_arr[grp_idx], 'y': y_arr[grp_idx], 'z': z_arr[grp_idx], 'ty': ty}
            scatter_li.append(sin_vol)

        return scatter_li

    def _select_split_and_remove_outlier(self,  eps_val, min_samples, thresh_num=10):
        """
        Generate inner volumes of the selected type to be visualized.
        Args:
            ty_name: List of types
            xli: List of x, with the same order of cells / bins as other iterable inputs
            yli: List of y, with the same order of cells / bins as other iterable inputs
            zli: List of z, with the same order of cells / bins as other iterable inputs
            yli: List of type, with the same order of cells / bins as other iterable inputs
            eps_val: In DBSCAN, the maximum distance between two samples for one to be considered as in the neighborhood
                        of the other. Try increase this value and min_samples if a cell type region is splitted into too
                        many subregions.
            min_samples: In DBSCAN, the number of samples (or total weight) in a neighborhood for a point to be
                            considered as a core point. This includes the point itself. Try increase this value and
                             eps_val if a cell type region is splitted into too many subregions. Try increase this value if
                             too many cells are excluded as outliers.
            thresh_num: minimum number of cells in a group to be considered to display a mesh
        
        Returns: 
            List of dictionary of inner volumes. Each dictionary includes keys of 'x', 'y', 'z', with values in NumPy.NdArray, and 'ty' with values of datatype str.
        """

        # scatter_li = []
        # for ty in list(dict.fromkeys(self.tyli)):
        #     if not ty == self.ty_name:
        #         continue
        #
        #     # print(ty)

        # 1. prepare data for analysis and processing
        ty_idx = np.where(np.array(self.tyli) == self.ty_name)

        x_arr = np.array(self.xli)[ty_idx]  # (n,)
        y_arr = np.array(self.yli)[ty_idx]
        z_arr = np.array(self.zli)[ty_idx]
        ty_arr = np.array(self.tyli)[ty_idx]

        scatter_li = self._split_and_remove_outlier_append([], x_arr, y_arr, z_arr, ty_arr[0], eps_val=eps_val, min_samples=min_samples, thresh_num=thresh_num)
        return scatter_li


def read_and_parse(spatial_regis:str,anno:str, h5ad_path:Optional[str] = None, adata_list:Optional[list] = None, sc_xyz:Optional[list] = None,):
    """
    Get x,y,z,anno columns as mesh input.

    Args:   
        h5ad_path: path of input of .h5ad files.
        spatial_regis: The column key in .obsm, default to be 'spatial_regis'. note that x,y,z 
        anno: The column key/name that identifies the grouping information(for example, clusters that correspond to different cell types)of spots.
        sc_xyz:The scale by which the spatial points in h5ad is scaled. when the `sc_xyz` is list, the model is scaled along the xyz
                axis at different scales. If `sc_xyz` is None, there will be by scale by defult parameter.
        

    Returns:
        a list of adata. and update adata to outpath which format that meets the requirements of 3D flow analysis.
    """
    xli = []
    yli = []
    zli = []
    tyli = []
    if h5ad_path:
        fnames =sort_file_names(file_path=h5ad_path,suffix='.h5ad')
        adata_list = []
        for fname in fnames:
            path = os.path.join(h5ad_path, fname)
            adata = anndata.read(path)
            adata_list.append(adata)
    elif adata_list:
        adata_list = adata_list
    else:  
        raise ValueError(f"h5ad_path and adata_list should have at least one that is not None.")
    for adata in adata_list:
        if adata.uns['data_unit']['binsize'] == 'cellbin':
            binsize = 10
        else:
            binsize = adata.uns['data_unit']['binsize']
        z_size = adata.uns['data_unit']['z_size']
        match = re.search(r'(\d+)um', z_size)
        z_size = int(match.group(1))
        if sc_xyz is None:
            sc_xyz = [None] * 3
            sc_xyz[0] = 1000/(binsize*0.5)
            sc_xyz[1] = 1000/(binsize*0.5)
            sc_xyz[2] = 1000/z_size
        x = (adata.obsm[spatial_regis][:, 0]*sc_xyz[0]).tolist()
        y = (adata.obsm[spatial_regis][:, 1]*sc_xyz[1]).tolist()
        z = (adata.obsm[spatial_regis][:, 2]*sc_xyz[2]).tolist()
        ty = adata.obs[anno].tolist()
        del adata
        gc.collect()

        xli = xli + x
        del x
        gc.collect()

        yli = yli + y
        del y
        gc.collect()

        zli = zli + z
        del z
        gc.collect()

        tyli = tyli + ty
        del ty
        gc.collect()
    
    return xli,yli,zli,tyli,sc_xyz