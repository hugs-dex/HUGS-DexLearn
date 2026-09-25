# 数据与模型接口

## 数据 producer → DexLearn

`ANYSCALEGRASP_DATA_ROOT` 指向具体版本的 bundle。外层 `archives/`、`extracted/`、`bundles/` 是归档组织，不是 loader 根目录。
HumanGraspData 产生 `OurHumanGraspFormat/grasp/`、`object/`、`metadata.csv`；物体资产位于 `object/DGN_2k/`。
具体目录及缺失项必须按数据版本记录，不能用研发数据补齐。

人手 grasp 包含左右手 `trans`、`rot`、MANO 参数和可选 `index_mcp_pos`；位置为米，旋转向量为弧度。
`index_mcp_pos` 来自 MANO joint 5，MANO 输出毫米转换为米后加 wrist translation。
`hand_pos_source=index_mcp` 读取预计算字段，不通过 wrist_pos 冒充 index-MCP。
`metadata.csv` 的 scene / pose_index 用于 posed-object soft labels；split 保持 producer 给定值。
人手数据自己的 scene frame 与 DGN scene frame 分别由其 producer 定义，不做隐式跨数据集旋转。

机器人训练保持原配置的 `human_DGN2k_full/{shadow,leap_sp}` 与兼容 Leap 数据路径，可由 CLI 改写。
关节顺序由对应 URDF/训练 metadata 决定；Shadow joint_num=44、Leap/Leap-SP joint_num=32 是双手总数。
不要只按 joint_num 推断顺序，也不要把 Leap 静默替换为 Leap-SP。
输出 grasp_pose 保持原模型的轨迹布局，左右手各自使用 xyz + wxyz + joint 顺序；单/双手和轨迹长度由模型配置确定。

## DexLearn → HUGS-BODex

五类固定顺序：

| ID | 名称 | Active hand 顺序 `[right, left]` |
| --- | --- | --- |
| 1 | 1_right_two | true, false |
| 2 | 2_right_three | true, false |
| 3 | 3_right_full | true, false |
| 4 | 4_both_three | true, true |
| 5 | 5_both_full | true, true |

`0_any` 是评分/采样请求占位符，不是第六个真实类别。

| 字段 | 形状/意义 |
| --- | --- |
| budget_scores | `(5,)`，语义由 score_semantics / factorization 记录 |
| index_mcp_pos | `(5, K, 2, 3)`，物理 scene frame 中的米单位位置 |
| wrist_quat | `(5, K, 2, 4)`，wxyz，active hand 的四元数必须归一化 |
| active_hand_mask | `(5, K, 2)` bool，手顺序固定 right、left |
| grasp_type_ids/names | 固定五类顺序，不重新编号 |
| scene_id / object_id | 保持物体与场景标识；scene_id 对应相对路径，无 `.npy` 后缀 |
| robot_name / robot_size | 输出命名空间及相对于人手的尺度比；robot_size 必须为正 |
| pc_runtime_scale | `1 / robot_size`；导出位置再映射回物理场景并恢复点云中心 |

配置为 wrist 时可导出 `wrist_pos`，但现有 HUGS-BODex consumer 要求 `index_mcp_pos`；不能不经转换互换。
单手记录的 inactive left 是占位姿态，consumer 必须遵守 mask，不把它作为真实左手抓取。
`scene_path` 和 `pc_path` 是运行时来源路径；跨机器移动产物应显式配置根目录/路径映射，不改写单位或 ID。
每个场景保存 NPY，另有 `manifest.json`、`scene_index.json`、`scene_budget_scores.jsonl`。

## Checkpoint 与复现元数据

manifest 保存 checkpoint/score/pose 的 SHA-256、模型配置、采样候选数与选择策略、seed、数据配置、scene whitelist hash、
代码 commit/dirty 状态与环境。不得将 dirty 状态隐藏为已验证发布 tuple。
输出中新加的 coordinate_contract 是解释性 metadata；现有必需字段及数组语义保持兼容。

首发用临时合成数据通过真实 exporter 与独立 BODex reader，证明格式兼容；它不证明模型质量或真实 human 初始化成功。
