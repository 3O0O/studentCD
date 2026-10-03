# 聚合协作结果包

[aggregate-results.v1.json](aggregate-results.v1.json) 是按显式字段从已验收数值报告新建的聚合投影，未原样复制报告。阅读入口为 [RESULTS.md](../docs/RESULTS.md)。每个 section 的 `source`／`sources` 与 `fields` 标明来源；`sources` 保存原字节 SHA，不含私人位置。

学生宏先在学生内平均再对学生等权；submission mean 按记录等权。expected 使用完整候选分布，argmax 使用最高概率代码。descriptive CI 为冻结数据的学生 bootstrap 区间，不是确认性结论。DIAGNOSTIC 使用未来标签，只能分析。

q 的 AUC、误改／漏改带覆盖，空可靠性箱保留 null。主 feedback_numeric 736条在最高箱、1条在次高箱，ECE0.001612303059918436；任务频率才是737条全在最高箱。五折来自 train；A 的候选分析复用同一 dev。

无个体记录、学生正文、模型、权重、私人机器信息或凭据。不能从该包独立重抽 bootstrap、重拟合 q、重算模型 likelihood 或核每条程序。copy likelihood／排名及强提示 oracle 未提供；没有补造。正文验证、数值验证与资源释放验收分别处理。

数据和模型遵守上游许可，此包不重新授予第三方数据许可。代码许可遵守仓库已明确声明；未选择时不代替用户决定。
