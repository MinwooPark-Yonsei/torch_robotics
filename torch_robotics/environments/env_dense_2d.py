import numpy as np
import torch
from matplotlib import pyplot as plt
import time

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField, MultiHollowBoxField
from torch_robotics.environments.utils import create_grid_spheres
from torch_robotics.robots import RobotPointMass
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvDense2D(EnvBase):

    def __init__(self,
                 name='EnvDense2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 seed=None,
                 circle_num=10,
                 box_num=10,
                 hollow_box_num=0,
                 addtional_circles=None,
                 additional_boxes=None,
                 additional_hollow_boxes=None,
                 **kwargs
                 ):
        
        if seed is None:
            seed = int(np.random.rand() * 2**32 - 1)
        np.random.seed(seed)        
        print("seed:", seed)

        circle_loc = np.random.uniform(-1, 1, (circle_num, 2))
        circle_r = np.array([0.125] * circle_num)
        box_loc = np.random.uniform(-1, 1, (box_num, 2))
        box_wh = np.array([[0.2, 0.2]] * box_num)
        hollow_box_loc = np.random.uniform(-1, 1, (hollow_box_num, 2))
        hollow_box_wh = np.array([[0.5, 0.5]] * hollow_box_num)
        wall_thickness = np.array([[0.1, 0.1]])

        if addtional_circles:
            circle_loc = np.concatenate((circle_loc, addtional_circles[0]))
            circle_r = np.concatenate((circle_r, addtional_circles[1]))
        if additional_boxes:
            box_loc = np.concatenate((box_loc, additional_boxes[0]))
            box_wh = np.concatenate((box_wh, additional_boxes[1]))
        if additional_hollow_boxes:
            hollow_box_loc = np.concatenate((hollow_box_loc, additional_hollow_boxes[0]))
            hollow_box_wh = np.concatenate((hollow_box_wh, additional_hollow_boxes[1]))
        
        print("circle pos:\n", circle_loc)
        print("box pos:\n", box_loc)
        print("hollow box pos:\n", hollow_box_loc)

        obj_list = []
        if circle_num or addtional_circles:
            obj_list.append(
                MultiSphereField(
                    circle_loc,
                    circle_r,
                    tensor_args=tensor_args
                )
            )
        if box_num or additional_boxes:
            obj_list.append(
                MultiBoxField(
                    box_loc,
                    box_wh,
                    tensor_args=tensor_args
                )
            )
        if hollow_box_num or additional_hollow_boxes:
            obj_list.append(
                MultiHollowBoxField(
                    hollow_box_loc,
                    hollow_box_wh,
                    wall_thickness,
                    tensor_args=tensor_args
                )
            )

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),  # environments limits
            obj_fixed_list=[ObjectField(obj_list, 'dense2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.3,
            n_pre_samples=50000,
            max_time=50
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            n_interpolated_points=None,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        if isinstance(robot, RobotPointMass):
            return params
        else:
            raise NotImplementedError


if __name__ == '__main__':
    env = EnvDense2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=DEFAULT_TENSOR_ARGS
    )
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
