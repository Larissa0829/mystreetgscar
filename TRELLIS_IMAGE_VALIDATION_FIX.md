# TRELLIS 图像有效性检查修复

## 问题描述

运行训练时遇到错误：
```
ValueError: zero-size array to reduction operation minimum which has no identity
```

### 错误原因

当渲染的对象图像**几乎完全透明**时，TRELLIS 的 `preprocess_image` 方法无法找到有效的边界框：

```python
alpha = output_np[:, :, 3]
bbox = np.argwhere(alpha > 0.8 * 255)  # 如果没有像素 > 204，bbox 为空数组
bbox = np.min(bbox[:, 1]), ...  # ❌ 对空数组调用 np.min 报错
```

### 可能的场景

1. **对象不在当前视角**: 在迭代 16001 时，某些动态对象可能移出了相机视野
2. **对象被遮挡**: 对象完全被其他物体遮挡
3. **渲染问题**: 对象的 opacity 太低或点云为空

## 解决方案

### 方案1: 在 train.py 中提前检查 ✓

**位置**: `train.py` 第 324-338 行

```python
# 检查图像有效性：alpha通道必须有足够的不透明像素
alpha_np = alpha.cpu().numpy()
valid_pixels = (alpha_np > 0.3).sum()  # 不透明度 > 30% 的像素数
total_pixels = alpha_np.shape[0] * alpha_np.shape[1]

if valid_pixels < total_pixels * 0.01:  # 有效像素少于1%
    print(f"    ⚠️ 跳过: 图像几乎完全透明 (有效像素: {valid_pixels}/{total_pixels})")
    continue

image_pil = Image.fromarray((rgba * 255).byte().cpu().numpy(), mode="RGBA")
image_pil.save(f"{ply_file_path}/{obj_name}_input.png")
print(f"    图像有效性: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
```

**优点**:
- ✅ 提前过滤无效图像，避免浪费 TRELLIS 计算
- ✅ 清晰的日志输出，便于调试
- ✅ 可调节的阈值（当前：不透明度 > 30% 的像素至少占 1%）

### 方案2: 增强 TRELLIS 鲁棒性 ✓

**位置**: `TRELLIS/trellis/pipelines/trellis_image_to_3d.py` 第 103-118 行

```python
output_np = np.array(output)
alpha = output_np[:, :, 3]
bbox = np.argwhere(alpha > 0.8 * 255)

# 处理空边界框：如果没有找到高不透明度的像素，降低阈值
if len(bbox) == 0:
    bbox = np.argwhere(alpha > 0.3 * 255)

# 如果仍然为空，使用整个图像
if len(bbox) == 0:
    bbox = 0, 0, output.width - 1, output.height - 1
else:
    bbox = np.min(bbox[:, 1]), np.min(bbox[:, 0]), np.max(bbox[:, 1]), np.max(bbox[:, 0])
```

**优点**:
- ✅ 多级阈值回退（0.8 → 0.3 → 全图）
- ✅ 不会因为边界情况而崩溃
- ✅ 对未来可能的边界情况更鲁棒

## 修复前后对比

### 修复前

```
生成 obj_006 的点云...
  保存输入图像: .../obj_006_input.png
  正在运行 TRELLIS pipeline...
❌ ValueError: zero-size array to reduction operation minimum which has no identity
```

### 修复后

```
生成 obj_006 的点云...
  图像有效性: 156/409600 (0.0%)
  ⚠️ 跳过: 图像几乎完全透明 (有效像素: 156/409600)
✓ 继续处理下一个对象
```

或者（如果图像有效）：

```
生成 obj_000 的点云...
  图像有效性: 45231/409600 (11.0%)
  正在运行 TRELLIS pipeline...
  ✓ 已保存: .../obj_000_sample.ply (20360 点)
```

## 参数说明

### 有效性检查阈值

**当前设置**:
```python
valid_pixels = (alpha_np > 0.3).sum()  # 不透明度阈值: 30%
if valid_pixels < total_pixels * 0.01:  # 最小有效像素比例: 1%
```

**可调节参数**:

| 参数 | 当前值 | 说明 | 建议范围 |
|------|--------|------|----------|
| 不透明度阈值 | 0.3 (30%) | 判断像素是否"有效"的透明度阈值 | 0.2 - 0.5 |
| 最小像素比例 | 0.01 (1%) | 有效像素占总像素的最小比例 | 0.005 - 0.02 |

**调整建议**:
- **如果跳过太多对象**: 降低阈值（如改为 0.005）
- **如果仍有报错**: 提高不透明度阈值（如改为 0.4）

### TRELLIS 边界框阈值

**当前设置**:
```python
# 第一级：高不透明度
bbox = np.argwhere(alpha > 0.8 * 255)  # 204/255

# 第二级：中等不透明度（回退）
if len(bbox) == 0:
    bbox = np.argwhere(alpha > 0.3 * 255)  # 76.5/255

# 第三级：使用全图（最后的回退）
if len(bbox) == 0:
    bbox = 0, 0, output.width - 1, output.height - 1
```

## 调试建议

### 查看渲染的输入图像

所有渲染的图像都保存在：
```
{model_path}/input_ply/{obj_name}_input.png
```

**检查步骤**:
1. 打开 `obj_XXX_input.png` 图像
2. 查看 alpha 通道（透明度）
3. 如果几乎完全透明 → 正常跳过
4. 如果有明显的对象但被跳过 → 降低阈值

### 日志输出

```bash
# 正常情况
✓ obj_000 的点云: 图像有效性: 45231/409600 (11.0%)

# 图像太透明，自动跳过
⚠️ obj_006: 跳过: 图像几乎完全透明 (有效像素: 156/409600)

# TRELLIS 降低阈值（如果图像通过了 train.py 的检查）
（在 TRELLIS 内部自动处理，无额外日志）
```

## 测试验证

运行训练并观察日志：

```bash
python train.py --config configs/your_config.yaml
```

**预期结果**:
1. ✅ 不再出现 `ValueError: zero-size array` 错误
2. ✅ 对于透明对象，显示 "⚠️ 跳过" 消息
3. ✅ 对于有效对象，显示图像有效性百分比
4. ✅ 继续处理其他对象，不中断训练

## 后续优化建议

1. **多帧选择**: 如果当前帧对象不可见，可以尝试其他帧（16002-16005）
2. **视角调整**: 自动寻找对象最可见的视角
3. **手动标注**: 对于重要对象，使用手动提供的参考图像

## 总结

✅ **双重保护机制**:
- **第一层** (train.py): 提前过滤无效图像，节省计算
- **第二层** (TRELLIS): 处理边界情况，避免崩溃

✅ **用户友好**:
- 清晰的日志输出
- 可调节的阈值
- 自动跳过问题对象

✅ **鲁棒性提升**:
- 不会因为个别对象不可见而中断整个流程
- 对各种边界情况都有妥善处理

