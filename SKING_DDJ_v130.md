# SKING_DDJ_v130

SKING_DDJ 系列的新 UV 提取版本。输入标准的 **512×512 edited 前后视组合图**，左侧为 `front_left`，右侧为 `back_left`；输出 Minecraft classic **64×64 RGBA** 皮肤。

模型通过图像语义编码、分部位特征路由和头部外层校准预测 UV。头部外层阈值固定为 **0.25**，其余外层按 0.5 导出；内层必须不透明，无效 UV 区域和透明像素 RGB 归零。身体 RGB 保持独立预测，头部校准只调整头部外层 Alpha。

## worker 接口

沿用 `worker_tasks.task_render_to_uv` 的原任务签名。新增版本的三个参数为：

```json
{
  "model_version": "SKING_DDJ_v130",
  "dense_uv_checkpoint_file": "SKING_DDJ_v130/parser.pt",
  "DMR_mappings_dir": "SKING_DDJ_v130/mappings"
}
```

这些逻辑路径会解析到当前 worker 仓库的 `releases/SKING_DDJ_v130/`。旧版本继续使用 `SKING_ROOT_DIR` 和 `DMR_ROOT_DIR`，不受此规则影响。三个参数必须成套使用；版本或映射不匹配会报错，不会回退到旧模型。

默认使用已有的 `DENSE_UV_EXECUTION_MODE=resident`：一个专属子进程加载语义编码器及 UV 模型，连续请求复用。沿用单版本缓存，切换版本先释放上一版；超时/异常会清理本次进程组，父进程退出时通过 pidfd 清理子进程。`oneshot` 模式可用于回归或诊断。

2026-11-01 已将生产专用 UV worker 的默认预加载切为本版本，并设置 `DENSE_UV_ALLOWED_MODEL_VERSIONS=SKING_DDJ_v130`；旧版请求会被拒绝。旧权重和历史记录保留，旧 image-to-skin 常驻进程已从 PM2 和开机脚本中移除。前端模型目录通过后端切换为 v130，图片转皮肤已恢复接收新任务；文字转皮肤、图片编辑转皮肤及旧模型继续下线。配置示例见 `SKING_DDJ_v130.env.example`。

## 目录与依赖

```text
releases/SKING_DDJ_v130/
  parser.pt                  # 纯推理权重：身体、头部、外层校准
  semantic_encoder.pth       # 本地语义编码器权重
  mappings/                  # 固定镜头映射和预计算几何
  runtime/                   # 独立 UV 推理代码
  vendor/dinov2/             # 编码器源码及其许可证
  pipeline.json              # 固定输入/输出与阈值
  weights_audit.json          # 权重来源哈希及导出核验
  release.json               # 全部发布资产哈希
```

发布包约 1.22 GiB，权重已复制到本仓库目录，不依赖原训练目录或用户级模型缓存。导出只包含推理张量，不携带优化器、训练数据清单或训练日志；导出前后逐张量相等。

本机默认使用 `/home/ds/miniconda3/envs/qwen-image-train/bin/python`，与回归基线环境一致；也可通过 `SKING_DDJ_V130_PYTHON` 指定兼容环境。验证环境为 Torch 2.8.0+cu128、torchvision 0.23.0+cu128、NumPy 2.4.1、Pillow 12.1.0。仓库原 Dockerfile 的 Torch 2.2 镜像不是该版本的验证环境；容器部署需要挂载完整发布包，并提供相应 CUDA/Python 环境。父 worker 无需导入新模型依赖。

权重和映射属于本地部署资产，不纳入普通 Git 大文件提交；迁移时复制整个发布目录，启动时按 `release.json` 校验。

## 本机独立调用

```bash
cd /home/ds/llms/github/entropydrop_gpu_worker
/home/ds/miniconda3/envs/qwen-image-train/bin/python \
  releases/SKING_DDJ_v130/runtime/run.py \
  --input /absolute/path/edited.png \
  --output /absolute/path/SKING_DDJ_v130.png
```

输出旁会生成 `.png.json` 审计，记录输入、输出、发布包和 checkpoint 哈希。模型只读取 edited 图，不需要原始 source、参考 UV 或个例可见性。输入必须符合上述尺寸和视角；不会静默缩放不兼容的图像。

## 验证

- 旧版与新版 worker 回归测试共 48 项通过。
- 三个固定案例与已选校准提取器输出逐字节一致。
- 连续四次请求复用同一个模型进程；单次推理和常驻推理输出一致。
- 实测常驻显存约 2026 MiB（1.98 GiB）；关闭运行时后释放。
- 完整记录位于 `verification/SKING_DDJ_v130/verification.json`。

本次属于已选模型的独立封装与版本接入，不是重新训练。三个回归案例用于保证迁移一致性，不能视为新的泛化评估。
