### 给另一个AI的“实验实现需求清单”（可直接复制粘贴）
你让它按这个清单去改 `zju3dv/street_gaussians`（动静分离、每车一套 Gaussians、跨帧共享、每帧位姿变换渲染）框架即可：[repo](https://github.com/zju3dv/street_gaussians)，论文 [arXiv](https://arxiv.org/abs/2401.01339)。

---

## 0. 实验目标（必须写在 README/实验脚本顶部）
- **任务**：Waymo 上做 3DGS 重建后进行动态车辆编辑（move/replace）。编辑会产生两类空洞：  
  - **A 静态背景空洞**：车移走后地面/背景暴露洞（A 已有 SBR-GS 模块处理，本文不重写，仅作为 pipeline 组件）。  
  - **B 车辆不可见面空洞**：车体 never-observed surfaces（背面/侧面）需要补全（本文主贡献）。
- **核心实验要证明**：B 能让车辆不可见面在多视角下几何稳定（不纸片、不漂浮），并在系统上与 A 形成完整可编辑 3DGS 世界。

---

## 1. 数据与Case选择（代码层面要固化为配置文件/清单）
### 1.1 Case清单
- **2 个 demo case**：用于快速迭代（跑得快）
- **8 个 hard cases**：用于对比/消融（最终结果）
- 每个 case 要包含：
  - segment id / 场景路径
  - 目标车辆 track_id
  - 选择 TRELLIS 输入的 frame_id（“可见视角最多的一帧”）
  - 编辑操作参数（move 的平移/旋转；replace 可选）

### 1.2 “never-observed”判定（实验协议）
实现一个简单函数（可近似，但要可复现）：
- 输入：相机外参、车辆位姿、车辆朝向 yaw
- 输出：某个车辆面（例如右侧 90° 扇区/后侧 120° 扇区）在训练序列中是否从未被任何相机观测
- 目的：确保 B 的实验不是“换个视角就能看到”的情况

---

## 2. 基线与对比组（必须能一键跑）
要提供统一脚本跑四组（相同 case、相同渲染轨迹、相同输出目录结构）：

1. **baseline**：Street Gaussians 原生编辑（无 A、无 B）
2. **+A**：只加静态背景补洞模块（SBR-GS）；B 不启用  
3. **+B**：只加车辆补全模块；A 不启用  
4. **ours(A+B)**：A 与 B 同时启用

输出要求：
- RGB 渲染视频（固定 3 条轨迹：front_orbit、side_low、top_high）
- depth 渲染（至少导出关键帧序列）
- 车辆点云可视化（车体 Gaussians/采样点云）

---

## 3. B模块：TRELLIS点云支架（需要实现的接口与产物）
### 3.1 输入与生成
- 输入：车辆裁剪图（来自选定 frame_id），裁剪框来自 Waymo 2D box 或投影 3D box（任选一种，但要固定）
- 调用 TRELLIS：输出 point cloud \(P_{trellis}\)
- 点云后处理：去噪/下采样（固定参数并记录）
- 保存：`P_trellis_raw.ply`、`P_trellis_denoised.ply`

### 3.2 配准（从“手动朝向”升级为可复现）
目标：得到相似变换 \((s,R,t)\) 把 trellis 点云对齐到 Waymo 车辆点云/车辆坐标系，并输出置信度。

- **初始化**：
  - yaw：用 Waymo 3D box yaw
  - 尺度：用 box 尺寸比值（或点云 bbox 比值）
  - 平移：用 box 中心
- **精配准**：ICP（点到点即可）对齐到 Waymo 车辆点云（Waymo自带）
- **置信度输出**：
  - ICP fitness / inlier ratio / residual（至少两个）
- 保存：`T_sim3.json`（s,R,t）、`icp_metrics.json`

注意：配准不必非常准，但必须可复现、可保存指标，用于论文里解释“配准不确定→软约束”。

---

## 4. B模块：软约束（loss 三件套 + 调度）
要在训练中只对“目标车辆实例 Gaussians”施加约束（动静分离、每车一套 Gaussians）。

### 4.1 三个loss（最小必须）
1. **L_scaffold**：车辆高斯中心到配准后 trellis 点云最近邻距离  
   - 用鲁棒核 Huber/Tukey  
   - 乘配准置信度权重 \(w\)（配准差就弱约束）
2. **L_box**：车辆高斯中心必须落在 Waymo 车辆 3D box 内（或超出惩罚）
3. **L_sym**：对称先验（最快可做的版本）  
   - 将 trellis 点云在车辆坐标系做镜像增强，再一起用于 L_scaffold（不必先做“成对高斯”，先把实验做出来）

### 4.2 调度策略（解决“前期好后期差”）
- 对 L_scaffold 做退火（例如训练后半程逐步减弱）或相反（可配置，做消融）
- 动态车 densification 后期冻结/阈值提高（避免后期漂浮点）
- 车辆分支学习率单独 schedule（后期强制降 LR）

---

## 5. 消融实验（必须能自动跑，至少4项）
在 2–3 个 hard cases 上跑即可：

- **-Trellis**：不使用 trellis 点云（B off）
- **-ICP**：只用 box yaw/尺度初始化，不做 ICP 精配准
- **-L_sym**：去掉对称先验
- **-L_box**：去掉 box 可行域约束（或替换成“无置信度加权/无退火”二选一）

输出同对比组：RGB+depth+点云可视化。

---

## 6. 指标与可视化（最小要求）
### 6.1 常规图像指标
- PSNR/SSIM/LPIPS：在 held-out views（或固定评估视角）上计算

### 6.2 几何proxy（至少一个）
利用 Waymo 车自带点云：
- 采样车辆 3DGS（或直接用车辆高斯中心）形成点集 \(P_{gs}\)
- 计算与 Waymo 车辆点云 \(P_{waymo}\) 的：
  - ICP residual（对齐后）或 Chamfer distance（proxy）
- 目的：给审稿人一个“几何数值”，证明不是只会画纹理

---

## 7. 交界处理（车体-背景缝，先工程化也行）
提供一个可复现的边界融合策略（mask feather / alpha blending），并输出“处理前后”对比图。

---

## 8. 输出目录规范（方便写论文与复现）
建议固定成：
- `results/{case}/{method}/rgb/`、`depth/`、`pcd/`、`metrics.json`、`videos/`
其中 `method ∈ {baseline, A_only, B_only, AB, ablation_xxx}`

---

### 一句话总结给对方AI
“请在 `street_gaussians` 动静分离框架里实现**车辆实例级**的 TRELLIS 点云支架：单视图生成点云→box init+ICP 配准→(L_scaffold鲁棒+L_box+L_sym)软约束+调度；并提供四组对比与四个消融的自动化脚本，输出 RGB/depth/点云/指标，证明 never-observed 车体面几何稳定。”
