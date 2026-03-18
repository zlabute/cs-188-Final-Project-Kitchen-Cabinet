============================================================
  OpenCabinet - Diffusion UNet Low-Dim Training
============================================================
Device: cuda

============================================================
  Loading augmented dataset
============================================================
Dataset root: /content/augmented_data/robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819
Train samples: 33855
Val samples:   3637
Obs dim:       13
Action dim:    7

============================================================
  Building diffusion policy
============================================================
Policy parameters: 4,233,983
EMA enabled (decay=0.999)

============================================================
  Training
============================================================
Epochs:           100
Batch size:       128
LR:               0.0001
Gradient accum:   1
Checkpoint dir:   /content/drive/MyDrive/cabinet_checkpoints

  Epoch     1  train_loss=0.596689  val_loss=nan  lr=1.00e-04
  Epoch    10  train_loss=0.075011  val_loss=0.098281  lr=9.77e-05
  Epoch    20  train_loss=0.060781  val_loss=0.063878  lr=9.07e-05
  Epoch    30  train_loss=0.054927  val_loss=0.057780  lr=7.97e-05
  Epoch    40  train_loss=0.053271  val_loss=0.054257  lr=6.58e-05
  Epoch    50  train_loss=0.048833  val_loss=0.055282  lr=5.03e-05
  Epoch    60  train_loss=0.046503  val_loss=0.053696  lr=3.48e-05
  Epoch    70  train_loss=0.045018  val_loss=0.055179  lr=2.08e-05
  Epoch    80  train_loss=0.042442  val_loss=0.060251  lr=9.62e-06
  Epoch    90  train_loss=0.041309  val_loss=0.060744  lr=2.47e-06
  Epoch   100  train_loss=0.041315  val_loss=0.059244  lr=0.00e+00

============================================================
  Training complete
============================================================
  Best val loss:      0.053696
  Final checkpoint:   /content/drive/MyDrive/cabinet_checkpoints/final_diffusion_policy.pt
  Best checkpoint:    /content/drive/MyDrive/cabinet_checkpoints/best_diffusion_policy.pt

  Evaluate with:
    python 07_evaluate_policy.py --checkpoint /content/drive/MyDrive/cabinet_checkpoints/best_diffusion_policy.pt


[robosuite WARNING] Could not load the mink-based whole-body IK. Make sure you install related import properly (e.g. pip install mink==0.0.5), otherwise you will not be able to use the default IK controller setting for GR1 robot. (__init__.py:40)
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
  Episode   1/20: FAIL    (steps= 500, reward=0.0) layout=53, style=37, task="Open the cabinet door."
  Episode   2/20: FAIL    (steps= 500, reward=0.0) layout=53, style=20, task="Open the cabinet door."
  Episode   3/20: FAIL    (steps= 500, reward=0.0) layout=26, style=13, task="Open the cabinet doors."
  Episode   4/20: SUCCESS (steps= 144, reward=0.0) layout=55, style=46, task="Open the cabinet door."
  Episode   5/20: SUCCESS (steps= 193, reward=0.0) layout=59, style=38, task="Open the cabinet door."
  Episode   6/20: FAIL    (steps= 500, reward=0.0) layout=47, style=43, task="Open the cabinet doors."
  Episode   7/20: SUCCESS (steps=  82, reward=0.0) layout=45, style=60, task="Open the cabinet doors."
  Episode   8/20: SUCCESS (steps=  97, reward=0.0) layout=22, style=17, task="Open the cabinet doors."
  Episode   9/20: SUCCESS (steps= 390, reward=0.0) layout=22, style=59, task="Open the cabinet doors."
  Episode  10/20: SUCCESS (steps= 105, reward=0.0) layout=31, style=34, task="Open the cabinet doors."
  Episode  11/20: SUCCESS (steps=  92, reward=0.0) layout=28, style=23, task="Open the cabinet doors."
  Episode  12/20: SUCCESS (steps= 169, reward=0.0) layout=43, style=57, task="Open the cabinet doors."
  Episode  13/20: FAIL    (steps= 500, reward=0.0) layout=43, style=38, task="Open the cabinet doors."
  Episode  14/20: FAIL    (steps= 500, reward=0.0) layout=44, style=37, task="Open the cabinet door."
  Episode  15/20: SUCCESS (steps= 118, reward=0.0) layout=20, style=53, task="Open the cabinet doors."
  Episode  16/20: FAIL    (steps= 500, reward=0.0) layout=20, style=40, task="Open the cabinet door."
  Episode  17/20: SUCCESS (steps= 397, reward=0.0) layout=21, style=13, task="Open the cabinet door."
  Episode  18/20: SUCCESS (steps= 421, reward=0.0) layout=25, style=49, task="Open the cabinet doors."
  Episode  19/20: FAIL    (steps= 500, reward=0.0) layout=12, style=37, task="Open the cabinet doors."
  Episode  20/20: FAIL    (steps= 500, reward=0.0) layout=13, style=35, task="Open the cabinet door."

============================================================
  Evaluation Results
============================================================
  Split:          pretrain
  Episodes:       20
  Successes:      11/20
  Success rate:   55.0%
  Avg ep length:  335.4 steps
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