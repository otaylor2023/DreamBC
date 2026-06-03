import torch
import os
import json
from dataclasses import dataclass

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def repo_path(*parts):
    return os.path.join(REPO_ROOT, *parts)


def _resolve_repo_path(path):
    if path is None or os.path.isabs(path):
        return path
    if path.startswith("Ctrl-World/"):
        return os.path.join(os.path.dirname(REPO_ROOT), path)
    return repo_path(path)


def resolve_paths(args):
    for attr in (
        "ckpt_path",
        "pi_ckpt",
        "dataset_root_path",
        "dataset_meta_info_path",
        "output_dir",
        "action_adapter",
        "data_stat_path",
        "val_model_path",
        "save_dir",
        "val_dataset_dir",
    ):
        if hasattr(args, attr):
            setattr(args, attr, _resolve_repo_path(getattr(args, attr)))
    return args


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    svd_model_path = "stabilityai/stable-video-diffusion-img2vid"
    clip_model_path = "openai/clip-vit-base-patch32"
    ckpt_path = repo_path("checkpoint/checkpoint-10000.pt")
    pi_ckpt = repo_path("openpi/checkpoint/pi05_droid")

    # dataset parameters
    # raw data
    dataset_root_path = repo_path("dataset_example")
    dataset_names = 'droid_subset'
    # meta info
    dataset_meta_info_path = repo_path('dataset_meta_info') #'/cephfs/cjyyj/code/video_evaluation/exp_cfg'#'dataset_meta_info'
    dataset_cfgs = dataset_names
    prob=[1.0]
    annotation_name='annotation' #'annotation_all_skip1'
    num_workers=4
    down_sample=3 # downsample 15hz to 5hz
    skip_step = 1
    

    # logs parameters
    debug = False
    tag = 'doird_subset'
    output_dir = repo_path("model_ckpt", tag)
    wandb_run_name = tag
    wandb_project_name = "droid_example"


    # training parameters
    learning_rate= 1e-5 # 5e-6
    gradient_accumulation_steps = 1
    mixed_precision = 'fp16'
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 500000
    checkpointing_steps = 20000
    validation_steps = 2500
    max_grad_norm = 1.0
    # for val
    video_num= 10

    ############################ model args ##############################

    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 7.5 #2.0 #7.5 #7.5 #7.5 #3.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    # num history and num future predictions
    num_frames= 5
    num_history = 6
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16 # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory



    ########################### rollout args ############################
    # policy
    task_type: str = "pickplace" # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    gripper_max_dict = {'replay':1.0, 'pickplace':0.75, 'towel_fold':0.95, 'wipe_table':0.95, 'tissue':0.97, 'close_laptop':0.95,'drawer':0.6,'stack':0.75,}
    z_min_dict = {'pickplace':0.23}
    ##############################################################################
    policy_type = 'pi05' # choose from ['pi05', 'pi0', 'pi0fast']
    action_adapter = repo_path('models/action_adapter/model2_15_9.pth') # adapat action from joint vel to cartesian pose
    pred_step = 5 # predict 5 steps (1s) action each time
    policy_skip_step = 2 # horizon = (pred_step-1) * policy_skip_step
    interact_num = 12 # number of interactions (each interaction contains pred_step steps)

    # wm
    data_stat_path = repo_path('dataset_meta_info/droid/stat.json')
    val_model_path = ckpt_path
    history_idx = [0,0,-12,-9,-6,-3]

    # save
    save_dir = repo_path('synthetic_traj')

    # select different traj for different tasks
    def __post_init__(self):
        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        self.z_min = self.z_min_dict.get(self.task_type, 0.18)
        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        if self.task_type == "replay":
            self.val_dataset_dir = repo_path("dataset_example/droid_subset")
            self.val_id = ["899", "18599","199",]
            self.start_idx = [8, 14, 8] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_replay"

        elif self.task_type == "keyboard":
            self.val_dataset_dir = repo_path("dataset_example/droid_subset")
            self.val_id = ["1799"]
            self.start_idx = [23] * len(self.val_id)
            self.instruction = [""] * len(self.val_id)
            self.task_name = "Rollouts_keyboard"

        # elif self.task_type == "keyboard2":
        #     self.val_dataset_dir = "/cephfs/shared/droid_hf/droid_svd_v2"
        #     self.val_id = ["1499"]*100
        #     self.start_idx = [8] * len(self.val_id) # 2599 8 #9499 10
        #     self.instruction = [""] * len(self.val_id)
        #     self.task_name = "Rollouts_keyboard_1499"
        #     self.ineraction_num = 7

        elif self.task_type == "pickplace":
            self.interact_num = 15
            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup_full/pickplace")
            self.val_id = ['0000','0001','0002','0003','0004','0005','0006','0007','0008','0009']
            self.start_idx = [0]*len(self.val_id)
            self.instruction = ['pick the blue block and place it in plate']*5 + ['pick the green block and place it in plate']*3 + ['pick the red block and place it in plate']*2

        elif self.task_type == "towel_fold":
            self.interact_num = 15
            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup")
            self.val_id =['0004','0005']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["fold the towel"] * len(self.val_id)

            self.val_dataset_dir = repo_path('dataset_example/droid_new_setup_eval/towel_fold')
            self.val_id = ['000018', '000044', '000120', '000228', '000255', '000336', '000403', '000427', '000453', '000643', '000739', '000803', '000833', '000902', '235555', '235713', '235826', '235933']
            self.start_idx = [0]*len(self.val_id)
            self.instruction = ['fold the towel']*len(self.val_id)

        elif self.task_type == "wipe_table":
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0006','0007']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = [
            #     "move the towel from left to right",
            #     "move the towel from left to right"
            # ]
            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup_full/wipe_table")
            self.val_id = ['0000','0001','0002','0003','0004']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['moving the towel from left to right', 'moving the towel from right to left', 'moving the towel from left to right','moving the towel from left to right','moving the towel from left to right']

        elif self.task_type == "tissue":
            # self.interact_num = 10
            # self.val_dataset_dir = "dataset_example/droid_new_setup"
            # self.val_id = ['0008','0009']
            # self.start_idx = [0] * len(self.val_id)
            # self.instruction = ["pull one tissue out of the box"] * len(self.val_id)
            # self.policy_skip_step = 3

            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup_full/tissue")
            self.val_id = ['0000', '0001', '0002', '0003', '0004']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['pull one tissue out of the box']*len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup_full/tissue")
            self.val_id = ['0000', '0001', '0002', '0003', '0004']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['pull one tissue out of the box']*len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup_full/tissue")
            self.val_id = ['0000', '0001', '0002', '0003', '0004']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ['pull one tissue out of the box']*len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "close_laptop":
            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup")
            self.val_id = ['0010','0011']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup_eval/laptop")
            self.val_id = ['135749','135849','135931','175856','175930','180035']
            self.start_idx = [0] * len(self.val_id)
            self.instruction = ["close the laptop"] * len(self.val_id)
            self.policy_skip_step = 3

        elif self.task_type == "stack":
            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup")
            self.val_id = ['0012','0013']
            self.start_idx = [5] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block"] * len(self.val_id)

            self.val_dataset_dir = repo_path("dataset_example/droid_new_setup_eval/stack")
            self.val_id = ['163907','164016','164350','232817','233512','234632','234823']
            self.start_idx = [10] * len(self.val_id)
            self.instruction = ["stack the blue block on the red block","stack the blue block on the red block","stack the blue block on the red block","stack the blue block on the red block","stack the green block on the red block","stack the blue block on the green block","stack the blue block on the green block"]
        
        elif self.task_type == 'drawer':
            self.val_dataset_dir = repo_path('dataset_example/droid_new_setup_full/drawer')
            self.val_id = ['0000', '0001', '0002', '0003', '0004']
            self.start_idx = [10]*len(self.val_id)
            self.instruction = ['pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer', 'pick up the sponge and place in the drawer']
            self.policy_skip_step = 3
        
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")
        resolve_paths(self)
