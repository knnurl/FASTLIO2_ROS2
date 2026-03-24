import torch

ckpt_path = '/home/local/ISDADS/ses634/fastlio2_ws/src/lidar_semantic/weights/semantickitti_tsunghan.tar'
out_path = '/home/local/ISDADS/ses634/fastlio2_ws/src/lidar_semantic/weights/semantickitti_tsunghan_1d.tar'

print(f"Loading weights from {ckpt_path}...")
ckpt = torch.load(ckpt_path, map_location='cpu')

# Handle the state dict
state_dict = ckpt.get('model_state_dict', ckpt)
new_state_dict = {}

for key, tensor in state_dict.items():
    # If it's a 4D tensor with 1x1 spatial dimensions (like fc1, fc2, fc3)
    # Squeeze the last dimension to make it a 3D tensor (Conv1d compatible)
    if tensor.dim() == 4 and tensor.shape[-1] == 1 and tensor.shape[-2] == 1:
        new_state_dict[key] = tensor.squeeze(-1)
    else:
        new_state_dict[key] = tensor

# Pack it back up
if 'model_state_dict' in ckpt:
    ckpt['model_state_dict'] = new_state_dict
else:
    ckpt = new_state_dict

torch.save(ckpt, out_path)
print(f"Successfully saved converted weights to {out_path}")
