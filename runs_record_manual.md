============================================================
OpenCabinet - Diffusion UNet Low-Dim Training
============================================================
Device: cuda

============================================================
Loading augmented dataset
============================================================
Dataset root: /content/augmented_data/robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819
Train samples: 33855
Val samples: 3637
Obs dim: 13
Action dim: 7

============================================================
Building diffusion policy
============================================================
Policy parameters: 4,233,983
EMA enabled (decay=0.999)

============================================================
Training
============================================================
Epochs: 100
Batch size: 128
LR: 0.0001
Gradient accum: 1
Checkpoint dir: /content/drive/MyDrive/cabinet_checkpoints

Epoch 1 train_loss=0.596689 val_loss=nan lr=1.00e-04
Epoch 10 train_loss=0.075011 val_loss=0.098281 lr=9.77e-05
Epoch 20 train_loss=0.060781 val_loss=0.063878 lr=9.07e-05
Epoch 30 train_loss=0.054927 val_loss=0.057780 lr=7.97e-05
Epoch 40 train_loss=0.053271 val_loss=0.054257 lr=6.58e-05
Epoch 50 train_loss=0.048833 val_loss=0.055282 lr=5.03e-05
Epoch 60 train_loss=0.046503 val_loss=0.053696 lr=3.48e-05
Epoch 70 train_loss=0.045018 val_loss=0.055179 lr=2.08e-05
Epoch 80 train_loss=0.042442 val_loss=0.060251 lr=9.62e-06
Epoch 90 train_loss=0.041309 val_loss=0.060744 lr=2.47e-06
Epoch 100 train_loss=0.041315 val_loss=0.059244 lr=0.00e+00

============================================================
Training complete
============================================================
Best val loss: 0.053696
Final checkpoint: /content/drive/MyDrive/cabinet_checkpoints/final_diffusion_policy.pt
Best checkpoint: /content/drive/MyDrive/cabinet_checkpoints/best_diffusion_policy.pt

Evaluate with:
python 07_evaluate_policy.py --checkpoint /content/drive/MyDrive/cabinet_checkpoints/best_diffusion_policy.pt

[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly (e.g. pip install mink==0.0.5), otherwise you will not be able to use the default IK controller setting for GR1 robot. (**init**.py:40)
WARNING: mimicgen environments not imported since mimicgen is not installed!
============================================================
OpenCabinet - Policy Evaluation (Diffusion UNet)
============================================================
Device: mps
Loaded diffusion policy from: tmp/cabinet_policy_checkpoints/checkpoint_epoch_00100.pt
Epoch 100, train_loss=0.041315, val_loss=0.059244
obs_dim=13, action_dim=7, horizon=16, n_obs_steps=2, n_action_steps=8

============================================================
Evaluating on pretrain split (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /Users/zanelabute/Downloads/CS_188_Robotics/Final_Project/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
Episode 1/20: FAIL (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
Episode 2/20: FAIL (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
Episode 3/20: FAIL (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
Episode 4/20: SUCCESS (steps= 144, reward=0.0) layout=55, style=46, task="Open the cabinet door."
Episode 5/20: SUCCESS (steps= 193, reward=0.0) layout=59, style=38, task="Open the cabinet door."
Episode 6/20: FAIL (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
Episode 7/20: SUCCESS (steps= 82, reward=0.0) layout=45, style=60, task="Open the cabinet doors."
Episode 8/20: SUCCESS (steps= 97, reward=0.0) layout=22, style=17, task="Open the cabinet doors."
Episode 9/20: SUCCESS (steps= 390, reward=0.0) layout=22, style=59, task="Open the cabinet doors."
Episode 10/20: SUCCESS (steps= 105, reward=0.0) layout=31, style=34, task="Open the cabinet doors."
Episode 11/20: SUCCESS (steps= 92, reward=0.0) layout=28, style=23, task="Open the cabinet doors."
Episode 12/20: SUCCESS (steps= 169, reward=0.0) layout=43, style=57, task="Open the cabinet doors."
Episode 13/20: FAIL (steps= 500, reward=0.0) layout=43, style=38, task="Open the cabinet doors."
Episode 14/20: FAIL (steps= 500, reward=0.0) layout=44, style=37, task="Open the cabinet door."
Episode 15/20: SUCCESS (steps= 118, reward=0.0) layout=20, style=53, task="Open the cabinet doors."
Episode 16/20: FAIL (steps= 500, reward=0.0) layout=20, style=40, task="Open the cabinet door."
Episode 17/20: SUCCESS (steps= 397, reward=0.0) layout=21, style=13, task="Open the cabinet door."
Episode 18/20: SUCCESS (steps= 421, reward=0.0) layout=25, style=49, task="Open the cabinet doors."
Episode 19/20: FAIL (steps= 500, reward=0.0) layout=12, style=37, task="Open the cabinet doors."
Episode 20/20: FAIL (steps= 500, reward=0.0) layout=13, style=35, task="Open the cabinet door."

============================================================
Evaluation Results
============================================================
Split: pretrain
Episodes: 20
Successes: 11/20
Success rate: 55.0%
Avg ep length: 335.4 steps
Avg reward: 0.000

============================================================
Performance Context
============================================================
Expected success rates from the RoboCasa benchmark:

| Method           | Pretrain | Target  |
| ---------------- | -------- | ------- |
| Random actions   | ~0%      | ~0%     |
| Diffusion Policy | ~30-60%  | ~20-50% |
| pi-0             | ~40-70%  | ~30-60% |
| GR00T N1.5       | ~35-65%  | ~25-55% |




============================================================
  OpenCabinet - Policy Evaluation (Diffusion UNet)
============================================================
Device: mps
Loaded diffusion policy from: test_epochs/checkpoint_epoch_00025.pt
  Epoch 25, train_loss=0.059052, val_loss=0.058860
  obs_dim=13, action_dim=7, horizon=16, n_obs_steps=2, n_action_steps=8

============================================================
  Evaluating on pretrain split (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /Users/zanelabute/Downloads/CS_188_Robotics/Final_Project/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: SUCCESS (steps= 240, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=25, style=57, task="Open the cabinet doors."
  Episode   4/20: SUCCESS (steps= 110, reward=0.0) layout=60, style=44, task="Open the cabinet doors."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=15, style=34, task="Open the cabinet doors."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=44, style=24, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=23, style=29, task="Open the cabinet doors."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=49, style=35, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=51, style=54, task="Open the cabinet doors."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=21, style=49, task="Open the cabinet door."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=21, style=38, task="Open the cabinet doors."
  Episode  12/20: FAIL    (steps= 500, reward=0.0) layout=49, style=24, task="Open the cabinet doors."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=60, style=48, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=18, style=12, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=19, style=27, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=38, style=41, task="Open the cabinet doors."
  Episode  17/20: SUCCESS (steps= 263, reward=0.0) layout=27, style=26, task="Open the cabinet doors."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=14, style=21, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=26, style=54, task="Open the cabinet doors."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=28, style=55, task="Open the cabinet door."

============================================================
  Evaluation Results
============================================================
  Split:          pretrain
  Episodes:       20
  Successes:      3/20
  Success rate:   15.0%
  Avg ep length:  455.6 steps
  Avg reward:     0.000

============================================================
  Performance Context
============================================================
Expected success rates from the RoboCasa benchmark:

  Method            | Pretrain | Target
  ------------------|----------|-------
  Random actions    |    ~0%   |   ~0%
  Diffusion Policy  |  ~30-60% | ~20-50%
  pi-0              |  ~40-70% | ~30-60%
  GR00T N1.5        |  ~35-65% | ~25-55%

===========================================================
  OpenCabinet - Policy Evaluation (Diffusion UNet)
============================================================
Device: mps
Loaded diffusion policy from: test_epochs/checkpoint_epoch_00050.pt
  Epoch 50, train_loss=0.048833, val_loss=0.055282
  obs_dim=13, action_dim=7, horizon=16, n_obs_steps=2, n_action_steps=8

============================================================
  Evaluating on pretrain split (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /Users/zanelabute/Downloads/CS_188_Robotics/Final_Project/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: SUCCESS (steps= 168, reward=0.0) layout=54, style=37, task="Open the cabinet door."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=29, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=30, style=27, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=56, style=39, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=15, style=12, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=60, style=40, task="Open the cabinet doors."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=35, style=13, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=38, style=30, task="Open the cabinet doors."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=12, style=44, task="Open the cabinet doors."

============================================================
  Evaluation Results
============================================================
  Split:          pretrain
  Episodes:       20
  Successes:      1/20
  Success rate:   5.0%
  Avg ep length:  483.4 steps
  Avg reward:     0.000

============================================================
  Performance Context
============================================================
Expected success rates from the RoboCasa benchmark:

============================================================
  OpenCabinet - Policy Evaluation (Diffusion UNet)
============================================================
Device: mps
Loaded diffusion policy from: test_epochs/checkpoint_epoch_00050.pt
  Epoch 50, train_loss=0.048833, val_loss=0.055282
  obs_dim=13, action_dim=7, horizon=16, n_obs_steps=2, n_action_steps=8

============================================================
  Evaluating on pretrain split (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /Users/zanelabute/Downloads/CS_188_Robotics/Final_Project/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: FAIL    (steps= 500, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: SUCCESS (steps= 168, reward=0.0) layout=54, style=37, task="Open the cabinet door."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=29, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=30, style=27, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=56, style=39, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=15, style=12, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=60, style=40, task="Open the cabinet doors."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=35, style=13, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=38, style=30, task="Open the cabinet doors."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=12, style=44, task="Open the cabinet doors."

============================================================
  Evaluation Results
============================================================
  Split:          pretrain
  Episodes:       20
  Successes:      1/20
  Success rate:   5.0%
  Avg ep length:  483.4 steps
  Avg reward:     0.000

============================================================
  Performance Context
============================================================
Expected success rates from the RoboCasa benchmark:

  Method            | Pretrain | Target
  ------------------|----------|-------
  Random actions    |    ~0%   |   ~0%
  Diffusion Policy  |  ~30-60% | ~20-50%
  pi-0              |  ~40-70% | ~30-60%
  GR00T N1.5        |  ~35-65% | ~25-55%

(.venv) zanelabute@Zanes-Air cabinet_door_project % git status
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   ../.gitignore
        modified:   ../runs_record_manual.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        figures/
        plot_generators/

no changes added to commit (use "git add" and/or "git commit -a")
(.venv) zanelabute@Zanes-Air cabinet_door_project % cd ..
(.venv) zanelabute@Zanes-Air cs188-cabinet-door-project % git status
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   .gitignore
        modified:   runs_record_manual.md

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        cabinet_door_project/figures/
        cabinet_door_project/plot_generators/

no changes added to commit (use "git add" and/or "git commit -a")
(.venv) zanelabute@Zanes-Air cs188-cabinet-door-project % git restore runs_record_m
anual.md
(.venv) zanelabute@Zanes-Air cs188-cabinet-door-project % git status
On branch main
Your branch is up to date with 'origin/main'.

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   .gitignore

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        cabinet_door_project/figures/
        cabinet_door_project/plot_generators/

no changes added to commit (use "git add" and/or "git commit -a")
(.venv) zanelabute@Zanes-Air cs188-cabinet-door-project % cd cabinet_door_project
(.venv) zanelabute@Zanes-Air cabinet_door_project % python 07_evaluate_policy.py --checkpoint test_epochs/checkpoint_epoch_00075.pt
[robosuite WARNING] No private macro file found! (macros.py:57)
[robosuite WARNING] It is recommended to use a private macro file (macros.py:58)
[robosuite WARNING] To setup, run: python /Users/zanelabute/Downloads/CS_188_Robotics/Final_Project/cs188-cabinet-door-project/robosuite/robosuite/scripts/setup_macros.py (macros.py:59)
[robosuite WARNING] Could not import robosuite_models. Some robots may not be available. If you want to use these robots, please install robosuite_models from source (https://github.com/ARISE-Initiative/robosuite_models) or through pip install. (__init__.py:30)
[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly (e.g. pip install mink==0.0.5), otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
WARNING: mimicgen environments not imported since mimicgen is not installed!
============================================================
  OpenCabinet - Policy Evaluation (Diffusion UNet)
============================================================
Device: mps
Loaded diffusion policy from: test_epochs/checkpoint_epoch_00075.pt
  Epoch 75, train_loss=0.043175, val_loss=0.057806
  obs_dim=13, action_dim=7, horizon=16, n_obs_steps=2, n_action_steps=8

============================================================
  Evaluating on pretrain split (20 episodes)
============================================================
[robosuite INFO] Loading controller configuration from: /Users/zanelabute/Downloads/CS_188_Robotics/Final_Project/cs188-cabinet-door-project/robosuite/robosuite/controllers/config/robots/default_pandaomron.json (composite_controller_factory.py:121)
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: FAIL    (steps= 500, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: FAIL    (steps= 500, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: FAIL    (steps= 500, reward=0.0) layout=59, style=26, task="Open the cabinet door."
  Episode   8/20: FAIL    (steps= 500, reward=0.0) layout=50, style=47, task="Open the cabinet doors."
  Episode   9/20: FAIL    (steps= 500, reward=0.0) layout=27, style=21, task="Open the cabinet door."
  Episode  10/20: FAIL    (steps= 500, reward=0.0) layout=29, style=17, task="Open the cabinet doors."
  Episode  11/20: SUCCESS (steps= 272, reward=0.0) layout=42, style=31, task="Open the cabinet doors."
  Episode  12/20: SUCCESS (steps= 113, reward=0.0) layout=11, style=40, task="Open the cabinet doors."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=30, style=22, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=55, style=14, task="Open the cabinet doors."
  Episode  15/20: FAIL    (steps= 500, reward=0.0) layout=24, style=18, task="Open the cabinet door."
  Episode  16/20: SUCCESS (steps= 148, reward=0.0) layout=45, style=19, task="Open the cabinet doors."
  Episode  17/20: FAIL    (steps= 500, reward=0.0) layout=37, style=45, task="Open the cabinet doors."
  Episode  18/20: FAIL    (steps= 500, reward=0.0) layout=31, style=43, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=47, style=36, task="Open the cabinet doors."
  Episode  20/20: SUCCESS (steps=  99, reward=0.0) layout=20, style=31, task="Open the cabinet doors."

============================================================
  Evaluation Results
============================================================
  Split:          pretrain
  Episodes:       20
  Successes:      4/20
  Success rate:   20.0%
  Avg ep length:  431.6 steps
  Avg reward:     0.000

============================================================
  Performance Context
============================================================
Expected success rates from the RoboCasa benchmark:

  Method            | Pretrain | Target
  ------------------|----------|-------
  Random actions    |    ~0%   |   ~0%
  Diffusion Policy  |  ~30-60% | ~20-50%
  pi-0              |  ~40-70% | ~30-60%
  GR00T N1.5        |  ~35-65% | ~25-55%