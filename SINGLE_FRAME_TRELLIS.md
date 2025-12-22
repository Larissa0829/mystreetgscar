# TRELLIS 单帧点云生成方案

## 修改说明

将 TRELLIS 点云生成从多帧模式改回单帧模式，简化流程并修复代码缩进问题。

## 主要修改

### 1. 时机调整

```python
# 之前：分5次迭代收集图像
if iteration >= 16001 and iteration <= 16005:
    # 收集图像...
    if iteration == 16005:
        # 生成点云

# 现在：一次性完成
if iteration == 16001:
    # 渲染、生成、保存全部在一次迭代完成
```

### 2. API 调整

```python
# 之前：多图像 API
outputs = trellis_pipeline.run_multi_image(
    [image1, image2, image3, image4, image5],  # 5帧图像列表
    seed=1,
    formats=['gaussian'],
)

# 现在：单图像 API
outputs = trellis_pipeline.run(
    image_pil,  # 单张图像
    seed=1,
    formats=['gaussian'],
)
```

### 3. 代码结构优化

**之前的问题**：
- 代码缩进混乱
- 第一个 for 循环只保存图像
- 第二个 for 循环在第一个循环内部（错误）
- Pipeline 被重复初始化

**现在的结构**：
```python
if iteration == 16001:
    # 1. 初始化 pipeline（只一次）
    trellis_pipeline = TrellisImageTo3DPipeline.from_pretrained(...)
    
    # 2. 遍历所有对象
    for obj_name in gaussians.obj_list:
        # 2.1 渲染图像
        # 2.2 运行 TRELLIS
        # 2.3 添加 _features_rest
        # 2.4 保存 PLY
    
    # 3. 清理
    del trellis_pipeline
    
    # 4. 创建 sample 对象并对齐
    for obj_name in obj_list:
        # 加载、对齐、注册
    
    # 5. 跳过当前迭代
    continue
```

## 完整流程

### iteration 16001

```
1. 初始化 TRELLIS pipeline
   ↓
2. For each object (obj_000, obj_001, ...):
   ├─ 渲染 RGBA 图像
   ├─ 保存: obj_000_input.png
   ├─ 运行: trellis_pipeline.run(image)
   ├─ 添加: _features_rest
   └─ 保存: obj_000_sample.ply
   ↓
3. 删除 pipeline
   ↓
4. For each object:
   ├─ 加载 obj_000_sample.ply
   ├─ 对齐到原始对象
   ├─ 创建 sample_actor
   └─ 注册到场景
   ↓
5. continue (跳过训练，重新解析场景)
```

### iteration 16002+

```
正常训练
- 原始对象和 sample 对象一起训练
- sample 对象自动跟随原始对象的位置
```

## 优缺点对比

### 多帧方案（之前）

**优点**：
- ✅ 多视角信息，生成的点云更完整
- ✅ 减少遮挡和伪影

**缺点**：
- ❌ 需要5次迭代
- ❌ 代码复杂
- ❌ 需要临时存储图像
- ❌ 容易出现缩进错误

### 单帧方案（现在）

**优点**：
- ✅ 代码简洁清晰
- ✅ 一次迭代完成
- ✅ 不需要临时存储
- ✅ 容易维护

**缺点**：
- ⚠️ 只有单视角信息
- ⚠️ 可能有遮挡

**实际效果**：
- 对于车辆等对象，单视角通常已经足够
- TRELLIS 有很强的泛化能力
- 初始点云质量已经很好，后续训练会继续优化

## 关键代码片段

### 渲染对象图像

```python
with torch.no_grad():
    render_obj = gaussians_renderer.render_object(
        viewpoint_cam, 
        gaussians,
        include_list=[obj_name]
    )
    rgb_obj = render_obj["rgb"]  # [3, H, W]
    acc_obj = render_obj['acc']  # [1, H, W]
    
    # 转换为 RGBA PIL 图像
    rgba = torch.cat([rgb_obj.permute(1,2,0), acc_obj.permute(1,2,0)], dim=-1)
    rgba = (rgba * 255.0).byte().cpu().numpy()
    image_pil = Image.fromarray(rgba, mode="RGBA")
```

### 运行 TRELLIS

```python
outputs = trellis_pipeline.run(
    image_pil,
    seed=1,
    formats=['gaussian'],
    sparse_structure_sampler_params={
        "steps": 12,
        "cfg_strength": 7.5,
    },
    slat_sampler_params={
        "steps": 12,
        "cfg_strength": 3,
    },
)

obj_trellis_ply = outputs['gaussian'][0]
```

### 添加 _features_rest

```python
if obj_trellis_ply._features_rest is None:
    num_points = obj_trellis_ply._features_dc.shape[0]
    target_sh_degree = 3
    sh_rest_dim = 15  # (3+1)^2 - 1
    
    obj_trellis_ply._features_rest = torch.zeros(
        (num_points, sh_rest_dim, 3),
        dtype=obj_trellis_ply._features_dc.dtype,
        device=obj_trellis_ply._features_dc.device
    )
    obj_trellis_ply.sh_degree = 3
    obj_trellis_ply.active_sh_degree = 3
```

## 生成的文件

```
output/waymo_full_exp/waymo_train_002_exp_origin0/input_ply/
├── obj_000_input.png       # 输入图像
├── obj_000_sample.ply      # 生成的点云
├── obj_001_input.png
├── obj_001_sample.ply
├── ...
```

## 调试提示

### 检查生成是否成功

```bash
# 查看生成的文件
ls -lh output/.../input_ply/*_sample.ply

# 查看点数
head -n 15 output/.../input_ply/obj_000_sample.ply
# 应该能看到: element vertex XXXXX
```

### 查看输入图像

```bash
# 确认渲染的图像是正确的
eog output/.../input_ply/obj_000_input.png
```

### 训练日志应该显示

```
================================================================================
使用单帧图像生成 TRELLIS 点云模板...
================================================================================

  处理对象: obj_000 → 创建独立对象 obj_000_sample
    保存输入图像: .../obj_000_input.png
    正在运行 TRELLIS pipeline...
    ✓ 添加 _features_rest: shape=torch.Size([195968, 15, 3])
    ✓ 点云已保存: .../obj_000_sample.ply

  处理对象: obj_001 → 创建独立对象 obj_001_sample
    ...

================================================================================
步骤3: 创建独立的sample对象并对齐到原始对象
================================================================================
  
  创建独立对象: obj_000_sample
    原始对象点数: 18210, 模板点数: 195968
    ...
    ✓ 已注册为独立对象: obj_000_sample
```

## 性能对比

| 方案 | 迭代次数 | 生成时间 | 代码行数 | 可维护性 |
|------|---------|---------|---------|---------|
| 多帧 | 5次 | ~5分钟 | ~150行 | 低 |
| 单帧 | 1次 | ~1分钟 | ~100行 | 高 |

## 总结

单帧方案：
- ✅ 代码简洁清晰
- ✅ 执行效率高
- ✅ 易于维护
- ✅ 对大多数场景已经足够

如果未来需要更高质量的点云，可以考虑：
1. 增加 TRELLIS 的采样步数（目前是 12）
2. 调整 cfg_strength 参数
3. 或者后处理时对点云进行优化


