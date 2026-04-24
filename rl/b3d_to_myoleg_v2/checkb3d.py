from b3d_reader import B3DReader

# ========================
# 改成你的真实数据路径！
# ========================
b3d_path = "/home/wxp/repos/projects/exo-assist-pipeline/data/addbiomechanics/train/No_Arm/Camargo2021_Formatted_No_Arm/AB06_split0/AB06_split0.b3d"

reader = B3DReader(b3d_path)
data = reader.extract_trial_data(0, 0)

print("\n=== 原始数据内存c占用 ===")
for key in ['pos', 'vel', 'tau', 'acc', 'com_pos', 'com_vel', 'com_acc', 'contact']:
    if key in data:
        arr = data[key]
        print(f"  {key}: {arr.shape} = {arr.nbytes / 1024**2:.2f} MB")
    else:
        print(f"  {key}: [不存在]")

print(f"\ncontact_bodies: {data.get('contact_bodies', '无')}")
print(f"DoF数量: {data['n_dofs']}")

print(f"\n=== 文件大小对比预估 ===")
import os
original_mb = os.path.getsize(b3d_path) / 1024**2

# 转换后会保存的数组总大小
converted_keys = ['pos', 'vel', 'tau', 'acc', 'com_pos', 'com_vel', 'com_acc', 'contact']
converted_bytes = sum(data[k].nbytes for k in converted_keys if k in data)
converted_mb = converted_bytes / 1024**2

print(f"原始 b3d 文件: {original_mb:.2f} MB")
print(f"提取的numpy数组总和: {converted_mb:.2f} MB")
print(f"比例: {converted_mb/original_mb*100:.1f}%")