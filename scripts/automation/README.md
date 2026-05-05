# Training Orchestrator

自动化多阶段 RL 训练：过拟合自动停止 → 保存 best checkpoint → 自动启动下一阶段。

## 文件结构

```
scripts/automation/       # 核心代码（不修改现有文件）
training_plans/           # YAML 训练计划
```

## 使用方法

```bash
cd ~/magiclab_rl_lab/scripts

# 查看计划（不执行）
python -m automation.orchestrator \
    --plan ../training_plans/<plan>.yaml \
    --project-root ~/magiclab_rl_lab \
    --device cuda:4 --dry-run

# 正式执行
nohup python -m automation.orchestrator \
    --plan ../training_plans/<plan>.yaml \
    --project-root ~/magiclab_rl_lab \
    --device cuda:4 \
    --start-from <stage_id> \
    --poll-interval 120 \
    > /tmp/orchestrator.log 2>&1 &
```

## CLI 参数

| 参数 | 说明 |
|------|------|
| `--plan` | YAML 训练计划路径（必填） |
| `--project-root` | magiclab_rl_lab 根目录 |
| `--start-from` | 从哪个 stage 开始 |
| `--fresh` | 忽略已有 state 重新开始 |
| `--dry-run` | 只打印计划不执行 |
| `--device` | GPU 设备（默认 cuda:0） |
| `--poll-interval` | 轮询间隔秒数（默认 120） |

## 查看日志

```bash
tail -f ~/magiclab_rl_lab/logs/orchestrator.log       # orchestrator
tail -f ~/magiclab_rl_lab/logs/train_<stage_id>.log   # 当前阶段训练
cat ~/magiclab_rl_lab/orchestrator_state.json | python -m json.tool  # 状态
```

## 崩溃恢复

重启时不加 `--fresh`，自动恢复：PID 存活→继续监控，PID 已死→重试。

## Smoke Test 结果 (2026-05-04)

RTX 6000D cuda:4 上通过。64 envs, 50 iters × 2 stages，~8 分钟。

- stage1 启动 → monitor 轮询 HEALTHY → 进程退出 → 保存 best checkpoint
- stage2 自动启动 (带 --checkpoint resume) → 完成 → All stages complete!

### 修复的 bug

1. **SameFileError**: config_swapper 同文件跳过
2. **Wrong run directory**: 按 `--run_name` 后缀精确匹配
3. **Zombie process**: `/proc/{pid}/stat` 检测 zombie 状态
4. **CWD mismatch**: launcher 指定 `cwd=project_root`
5. **Isaac Sim 启动慢**: 8×30s 循环等待（最多 4 分钟）
