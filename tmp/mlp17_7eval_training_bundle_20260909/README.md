# 17站训练、7站验证动态MLP训练包

## 数据口径

- 训练：1224例。10个非评估站各86例，7个评估站各52例。
- 验证：119例。GOB、DRA、IZA、TBL、SEL、GWN、CAB各17例。
- 测试：119例。训练与验证阶段不读取。
- 训练批次采用站点均衡抽样。

## 执行顺序

`launch_mlp17_7eval_training.sh`依次完成：

1. 合并原开发集与确认集索引，不复制约5 GB张量；
2. 检查1462个案例及标签；
3. 对卫星、NWP和固定时效融合执行单批前向/反向预检；
4. 用新训练集重新训练共享专家和时效基线；
5. 冻结专家，提取1224/119例训练与验证状态；
6. 按原动态MLP架构训练42、43、44三个路由种子；
7. 停在验证完成状态，不读取119例测试集。

远端默认路径为`/root/autodl-tmp`。解压目录应命名为`guizhou_mlp17_7eval_code_20260909`，然后运行：

```bash
nohup bash /root/autodl-tmp/guizhou_mlp17_7eval_code_20260909/scripts/launch_mlp17_7eval_training.sh \
  > /root/autodl-tmp/guizhou_mlp17_7eval_launch.log 2>&1 &
```

进度文件：`/root/autodl-tmp/guizhou_mlp17_7eval_run_20260909/status.txt`。
