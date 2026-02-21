# Stage 2E: CLI/Server 入口合流 - Pipeline 统一实现

## TL;DR

> **Quick Summary**: 将 deepseek-ocr-pipeline 的 API 骨架接入真实调用链路，让 CLI 和 Server 两个 crate 不再直接依赖 deepseek-ocr-core/config/assets/infer-*，而是只依赖 pipeline 这一层。实现"一个入口，统一语义"。
> 
> **Deliverables**:
> - OcrRuntimeBuilder::build() 完整实现（配置解析 + 资源准备）
> - OcrModelManager::load() 完整实现（模型加载 + 缓存）
> - OcrPipeline::generate() 完整实现（统一推理入口）
> - CLI 迁移到 pipeline（Cargo.toml + app.rs）
> - Server 迁移到 pipeline（Cargo.toml + state.rs/generation.rs）
> - 依赖边界 CI 门禁（禁止 cli/server 直连底层 crates）
> - 行为契约测试（stdout/JSON schema 不变）
> 
> **Estimated Effort**: Large (预计 8-12 小时)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: Task 1 → Task 3 → Task 5 → Task 7 → Task 9 → Final Verification

---

## Context

### Original Request
用户要求实现 Stage 2E（CLI/Server 入口合流），具体目标：
- 在 pipeline 内完成最小可用实现（OcrConfigResolver/OcrModelManager/OcrPipeline）
- 迁移 CLI 到 pipeline（只依赖 deepseek-ocr-pipeline）
- 迁移 Server 到 pipeline（只依赖 deepseek-ocr-pipeline）
- 保持现有行为不变（门禁依赖 stdout/JSON output 格式）
- 通过 cargo check/clippy 和 benchsuite failfast 门禁

### Interview Summary
**Key Discussions**:
- 当前 pipeline 已有 API 骨架，但大部分是 todo!()/bail!() 占位
- config.rs 的 resolve() 已有真实实现（走 deepseek-ocr-config）
- CLI/Server 当前直接依赖 core/config/assets/infer-* 等底层 crates
- 用户极度反感复制粘贴、薄封装与"写死流程"

**Research Findings**:
- Metis 识别出 6 大类 gaps：未问问题、guardrails、scope creep、假设验证、缺失验收标准、边界场景
- 关键风险：依赖边界破坏、行为契约漂移、并发缓存竞态、门禁失败
- 必须锁死 scope：2E 只做"入口合流，不改语义"，增强项列入 2F

### Metis Review
**Identified Gaps** (addressed):
- **依赖边界硬约束**: 已加入 CI 门禁任务（Task 10）
- **行为契约测试**: 已加入 CLI/Server 回归验证任务（Task 8/9）
- **并发缓存 singleflight**: 已在 OcrModelManager 实现要求中明确
- **配置优先级边缘 case**: 已在验收标准中补充
- **scope creep 锁死**: 已在 Must NOT Have 明确列出 6 项禁止

---

## Work Objectives

### Core Objective
实现 pipeline 作为 OCR 推理的唯一高层入口，让 CLI/Server 应用层不再感知底层 config/assets/core/infer-* 的细节，同时保持现有行为完全不变（门禁无感知迁移）。

### Concrete Deliverables
- OcrRuntimeBuilder::build() → 返回 OcrRuntime（含 OcrModelManager）
- OcrModelManager::load() → 返回 OcrPipelineHandle（带缓存）
- OcrPipeline::generate() → 返回 OcrResponse（统一语义）
- CLI Cargo.toml → 移除 core/config/assets/infer-*，新增 pipeline
- Server Cargo.toml → 移除 core/config/assets/infer-*，新增 pipeline
- CLI app.rs → 使用 pipeline API，移除 match ModelKind -> load_*_model
- Server state.rs → 使用 pipeline API，移除重复加载逻辑
- CI 门禁 → 禁止 cli/server 直接 import 底层 crates
- 行为契约测试 → stdout/JSON schema/exit code 不变

### Definition of Done
- [ ] `cargo check --workspace --all-targets --all-features` 全绿
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` 全绿
- [ ] CLI 同一输入下输出 JSON schema 完全一致
- [ ] Server 关键 API 响应体/状态码/错误码一致
- [ ] benchsuite failfast 通过（单代表长 case + accelerate + 8192）
- [ ] 依赖边界 CI 门禁生效（grep 断言无违禁 import）

### Must Have
- 配置优先级：CLI > config file > defaults（已实现）
- 模型缓存：并发 N 请求同模型仅加载一次（singleflight）
- 错误映射：pipeline 统一错误类型，CLI/Server 对外错误码兼容
- 观测稳定：日志/trace span 名称和关键字段不变
- 线程安全：OcrModelManager/OcrPipeline Send + Sync

### Must NOT Have (Guardrails)
- **禁止**在 2E 重构底层 infer/core API（scope creep）
- **禁止**引入新配置项或重命名现有配置键（schema drift）
- **禁止**统一 CLI/Server 全部业务语义差异（只统一入口）
- **禁止**缓存淘汰策略复杂化（无 LRU/TTL 大改）
- **禁止**错误体系大一统重构（先兼容映射）
- **禁止**输出格式"清理"或日志美化（破坏门禁）
- **禁止**用 `#[allow(...)]` 掩盖 clippy 问题

---

## Verification Strategy (MANDATORY)

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.
> Acceptance criteria requiring "user manually tests/confirms" are FORBIDDEN.

### Test Decision
- **Infrastructure exists**: YES (cargo check/clippy + benchsuite)
- **Automated tests**: YES (tests-after) - 测试放在 crate 的 tests/ 目录
- **Framework**: cargo test (workspace 自带) + benchsuite (Python)
- **Agent-Executed QA**: ALWAYS (mandatory for all tasks)

### QA Policy
Every task MUST include agent-executed QA scenarios (see TODO template below).
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Frontend/UI**: N/A (CLI/Server 后端服务)
- **TUI/CLI**: interactive_bash (tmux) — Run command, validate output
- **API/Backend**: Bash (curl) — Send requests, assert status + response fields
- **Library/Module**: Bash (cargo test/cargo check) — Compile + test assertions

---

## Execution Strategy

### Parallel Execution Waves

> Maximize throughput by grouping independent tasks into parallel waves.
> Each wave completes before the next begins.
> Target: 5-8 tasks per wave. Fewer than 3 tasks per wave (except final) = under-splitting.

```
Wave 1 (Start Immediately — foundation + config):
├── Task 1: OcrConfigResolver 完善 + 测试 [quick]
├── Task 2: OcrFsOptions 工具函数 [quick]
├── Task 3: OcrModelId 验证逻辑 [quick]
├── Task 4: OcrPipelineEvent 序列化 [quick]
└── Task 5: 依赖边界 CI 脚本 [quick]

Wave 2 (After Wave 1 — core pipeline impl):
├── Task 6: OcrRuntimeBuilder::build() [deep]
├── Task 7: OcrModelManager::load() [deep]
├── Task 8: OcrPipeline::generate() [deep]
└── Task 9: OcrPipelineHandle 缓存包装 [unspecified-high]

Wave 3 (After Wave 2 — CLI migration):
├── Task 10: CLI Cargo.toml 依赖迁移 [quick]
├── Task 11: CLI app.rs 主链路迁移 [deep]
├── Task 12: CLI debug/bench 兼容 [unspecified-high]
└── Task 13: CLI 行为契约测试 [deep]

Wave 4 (After Wave 3 — Server migration):
├── Task 14: Server Cargo.toml 依赖迁移 [quick]
├── Task 15: Server state.rs 迁移 [deep]
├── Task 16: Server generation.rs 迁移 [deep]
└── Task 17: Server API 契约测试 [deep]

Wave 5 (After Wave 4 — integration + verification):
├── Task 18: workspace clippy --all-features [deep]
├── Task 19: benchsuite failfast 单长 case [unspecified-high]
├── Task 20: 依赖边界 CI 门禁验证 [quick]
└── Task 21: Git cleanup + tagging [git]

Wave FINAL (After ALL tasks — independent review, 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: Task 1 → Task 6 → Task 7 → Task 8 → Task 11 → Task 15 → Task 19 → F1-F4
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 4 (Wave 1)
```

### Dependency Matrix (abbreviated — show ALL tasks in your generated plan)

- **1-5**: — — 6-9, 1
- **6**: 1, 2, 3 — 7-9, 2
- **7**: 6 — 9, 11, 15, 2
- **8**: 6, 7 — 9, 11, 15, 2
- **9**: 6, 7, 8 — 11, 15, 2
- **10**: 9 — 11-13, 3
- **11**: 9, 10 — 13, 18, 19, 3
- **14**: 9 — 15-17, 4
- **15**: 9, 14 — 17, 18, 19, 4
- **18**: 11, 13, 15, 17 — 19-21, 5
- **19**: 18 — 21, 5
- **21**: 19 — F1-F4, FINAL

### Agent Dispatch Summary

- **1**: **4** — T1-T5 → `quick`
- **2**: **4** — T6-T9 → `deep`/`unspecified-high`
- **3**: **4** — T10 → `quick`, T11 → `deep`, T12 → `unspecified-high`, T13 → `deep`
- **4**: **4** — T14 → `quick`, T15-T17 → `deep`, T18 → `deep`
- **5**: **4** — T19 → `unspecified-high`, T20 → `quick`, T21 → `git`
- **FINAL**: **4** — F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep`

---

## TODOs

- [x] 1. OcrConfigResolver 完善 + 测试

  **What to do**:
  - 完善 config.rs 的 OcrConfigResolver::resolve() 错误处理
  - 添加 OcrConfigSource 的 Display/Serialize 实现（便于日志）
  - 新增 tests/config_resolver_test.rs: 测试 defaults < config_file < cli_args 优先级
  - 补充边界 case：空 patch、部分 patch、冲突 patch 的合并行为

  **Must NOT do**:
  - 不修改 deepseek-ocr-config 的内部逻辑
  - 不引入新的配置字段
  - 不改变现有 ConfigOverrides 的语义

  **Recommended Agent Profile**:
  > quick 类别适合这种单一关注点的任务：完善已有实现 + 测试
  - **Category**: `quick`
    - Reason: 任务聚焦在单一模块（config.rs），逻辑清晰，无跨模块依赖
  - **Skills**: [`git-master`]
    - `git-master`: 需要查看 config.rs 的修改历史，确保不破坏已有约定
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法，按现有模式实现即可
    - `ultrabrain`: 逻辑不复杂，无需深度推理

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2-5)
  - **Blocks**: [Tasks 6-9, 10-21]
  - **Blocked By**: None (can start immediately)

  **References** (CRITICAL - Be Exhaustive):

  **Pattern References** (existing code to follow):
  - `crates/pipeline/src/config.rs:195-268` - OcrConfigResolver/OcrConfigLayer/OcrPatchLayer 现有实现
  - `crates/pipeline/src/ext.rs:16-52` - IntoConfigOverrides trait + patch -> overrides 转换

  **API/Type References** (contracts to implement against):
  - `crates/config/src/lib.rs:AppConfig` - load_or_init/apply_overrides/normalise 签名
  - `crates/config/src/config.rs:ConfigOverrides` - 配置覆盖结构定义

  **Test References** (testing patterns to follow):
  - `crates/config/tests/app_config_test.rs` - AppConfig 加载/覆盖测试模式
  - `crates/pipeline/tests/` (新建) - pipeline crate 测试目录结构

  **External References** (libraries and frameworks):
  - N/A (纯内部逻辑)

  **WHY Each Reference Matters** (explain the relevance):
  - config.rs:195-268 - 理解现有 Layer/Resolver 架构，确保扩展不破坏设计
  - ext.rs:16-52 - 理解 patch -> overrides 转换逻辑，确保 resolve() 正确调用
  - AppConfig - 明确 resolve() 最终要返回的类型和行为

  **Acceptance Criteria**:

  > **AGENT-EXECUTABLE VERIFICATION ONLY** — No human action permitted.
  > Every criterion MUST be verifiable by running a command or using a tool.

  **If TDD (tests enabled):**
  - [ ] Test file created: crates/pipeline/tests/config_resolver_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test config_resolver_test → PASS (5 tests, 0 failures)

  **QA Scenarios (MANDATORY — task is INCOMPLETE without these):**

  ```
  Scenario: 配置优先级验证 - defaults < config_file < cli_args
    Tool: Bash (cargo test)
    Preconditions: 创建测试 fixture（临时 config.toml + CLI patch）
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test config_resolver_test -- priority_defaults_less_than_config_file
      2. cargo test -p deepseek-ocr-pipeline --test config_resolver_test -- priority_config_file_less_than_cli_args
      3. 验证测试输出包含 "test result: ok"
    Expected Result: 5 tests all pass, 0 failures
    Failure Indicators: 任何 test failure 或 panic
    Evidence: .sisyphus/evidence/task-1-config-priority-test.txt

  Scenario: clippy 检查通过
    Tool: Bash (cargo clippy)
    Preconditions: 修改后的 config.rs/ext.rs
    Steps:
      1. cargo clippy -p deepseek-ocr-pipeline --tests -- -D warnings
      2. 验证输出不包含 "error" 或 "warning"
    Expected Result: 0 errors, 0 warnings
    Failure Indicators: 任何 clippy diagnostic
    Evidence: .sisyphus/evidence/task-1-clippy-check.txt
  ```

  **Evidence to Capture:**
  - [ ] 测试输出：task-1-config-priority-test.txt
  - [ ] clippy 输出：task-1-clippy-check.txt

  **Commit**: YES (groups with 2-5)
  - Message: `refactor(pipeline): OcrConfigResolver 完善 + 工具函数`
  - Files: `crates/pipeline/src/config.rs`, `crates/pipeline/src/ext.rs`, `crates/pipeline/src/model.rs`, `crates/pipeline/src/observer.rs`, `crates/pipeline/src/fs.rs`, `crates/pipeline/tests/config_resolver_test.rs`
  - Pre-commit: `cargo check -p deepseek-ocr-pipeline`

- [x] 2. OcrFsOptions 工具函数

  **What to do**:
  - 在 fs.rs 添加 OcrFsOptions::build_local_fs() 静态方法（复用 config.rs 的 build_local_fs）
  - 添加 OcrFsOptions::default_app_name() 常量
  - 新增 tests/fs_options_test.rs: 测试路径构建逻辑
  - 确保与 deepseek-ocr-config::LocalFileSystem 兼容

  **Must NOT do**:
  - 不修改 LocalFileSystem 的内部实现
  - 不引入新的环境变量处理逻辑
  - 不改变默认目录名（"deepseek-ocr"）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 工具函数封装，逻辑简单，无复杂依赖
  - **Skills**: [`git-master`]
    - `git-master`: 查看 fs.rs 修改历史，确保与现有约定一致
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3-5)
  - **Blocks**: [Tasks 6-9]
  - **Blocked By**: None

  **References**:
  - `crates/pipeline/src/config.rs:241-252` - build_local_fs 现有实现
  - `crates/pipeline/src/fs.rs:1-28` - OcrFsOptions 定义
  - `crates/config/src/fs.rs:LocalFileSystem` - 目标兼容类型

  **Acceptance Criteria**:
  - [ ] Test file created: crates/pipeline/tests/fs_options_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test fs_options_test → PASS

  **QA Scenarios**:
  ```
  Scenario: OcrFsOptions::build_local_fs() 路径构建
    Tool: Bash (cargo test)
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test fs_options_test -- build_local_fs_with_custom_roots
      2. 验证生成的 config_dir/cache_dir 符合预期
    Expected Result: 3 tests pass
    Evidence: .sisyphus/evidence/task-2-fs-test.txt
  ```

  **Commit**: YES (groups with 1, 3-5)
  - Message: `refactor(pipeline): OcrConfigResolver 完善 + 工具函数`

- [x] 3. OcrModelId 验证逻辑

  **What to do**:
  - 在 model.rs 添加 OcrModelId::validate() 私有方法（已在 from_model_id 内联）
  - 添加 OcrModelId::known_models() 返回已知 model_id 列表（deepseek-ocr/paddleocr-vl/dots-ocr/glm-ocr）
  - 新增 tests/model_id_test.rs: 测试验证逻辑（空字符串/空白符/非法字符）
  - 实现 Display/TryFrom<String>/TryFrom<&str> (已有部分)

  **Must NOT do**:
  - 不改变现有验证规则（非空 + 无空白符）
  - 不引入正则表达式验证（保持简单）
  - 不修改 known_models 列表（与 config.toml 默认 entries 一致）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 类型验证逻辑，规则清晰
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要创造性解法

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-2, 4-5)
  - **Blocks**: [Tasks 6-9]
  - **Blocked By**: None

  **References**:
  - `crates/pipeline/src/model.rs:1-51` - OcrModelId 现有定义
  - `crates/config/src/config.rs:AppConfig::models.entries` - 配置文件中的 model_id 使用

  **Acceptance Criteria**:
  - [ ] Test file created: crates/pipeline/tests/model_id_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test model_id_test → PASS

  **QA Scenarios**:
  ```
  Scenario: OcrModelId 验证 - 非法输入拒绝
    Tool: Bash (cargo test)
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test model_id_test -- reject_empty_string
      2. cargo test -p deepseek-ocr-pipeline --test model_id_test -- reject_whitespace
      3. 验证返回 Err 而非 panic
    Expected Result: 4 tests pass (empty, whitespace, valid, known_models)
    Evidence: .sisyphus/evidence/task-3-model-id-test.txt
  ```

  **Commit**: YES (groups with 1-2, 4-5)
  - Message: `refactor(pipeline): OcrConfigResolver 完善 + 工具函数`

- [x] 4. OcrPipelineEvent 序列化

  **What to do**:
  - 在 observer.rs 为 OcrPipelineEvent 实现 serde::Serialize（便于日志/调试输出）
  - 为 OcrPipelineEvent 实现 Display（便于人类可读日志）
  - 新增 tests/observer_event_test.rs: 测试序列化输出格式
  - 确保 Duration 字段序列化为人类可读格式（如 "1.234s"）

  **Must NOT do**:
  - 不改变 OcrPipelineEvent 的字段结构
  - 不引入新的依赖（serde 已在 workspace）
  - 不修改 NoopObserver 行为

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: 派生 trait 实现，模式固定
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-3, 5)
  - **Blocks**: [Tasks 6-9]
  - **Blocked By**: None

  **References**:
  - `crates/pipeline/src/observer.rs:1-60` - OcrPipelineEvent/OcrPipelineObserver/NoopObserver 定义
  - `crates/core/src/benchmark.rs:Timer/Event` - 类似事件的序列化模式（如有）

  **Acceptance Criteria**:
  - [ ] Test file created: crates/pipeline/tests/observer_event_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test observer_event_test → PASS

  **QA Scenarios**:
  ```
  Scenario: OcrPipelineEvent 序列化 - JSON 格式
    Tool: Bash (cargo test)
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test observer_event_test -- serialize_model_load_finished
      2. 验证 JSON 输出包含所有字段
      3. 验证 duration 字段为 "X.XXXs" 格式
    Expected Result: 3 tests pass (serialize, display, duration_format)
    Evidence: .sisyphus/evidence/task-4-observer-serialize-test.txt
  ```

  **Commit**: YES (groups with 1-3, 5)
  - Message: `refactor(pipeline): OcrConfigResolver 完善 + 工具函数`

- [x] 5. 依赖边界 CI 脚本

  **What to do**:
  - 创建 .github/workflows/dependency-boundary.yml（CI 门禁）
  - 添加 grep/ast-grep 规则：禁止 cli/server 直接 import deepseek-ocr-core/config/assets/infer-*
  - 添加 bash 脚本 scripts/check-dependency-boundary.sh（本地可运行）
  - 在 workflow 中集成到 CI 流程（cargo check 之后）

  **Must NOT do**:
  - 不修改现有 CI 流程（新增独立 job）
  - 不引入新的外部工具（只用 grep/ast-grep）
  - 不改变 workspace 依赖结构

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: CI 脚本，模式固定
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1-4)
  - **Blocks**: [Tasks 18-21]
  - **Blocked By**: None

  **References**:
  - `.github/workflows/` (现有) - CI workflow 结构
  - `crates/cli/Cargo.toml` - 当前依赖列表（用于 grep 白名单/黑名单）
  - `crates/server/Cargo.toml` - 当前依赖列表

  **Acceptance Criteria**:
  - [ ] CI workflow created: .github/workflows/dependency-boundary.yml
  - [ ] Script created: scripts/check-dependency-boundary.sh
  - [ ] 本地运行脚本通过（当前会失败，因为还没迁移）
  - [ ] 脚本输出清晰的错误信息（列出违禁 import）

  **QA Scenarios**:
  ```
  Scenario: 依赖边界检查 - 检测违禁 import
    Tool: Bash (scripts/check-dependency-boundary.sh)
    Preconditions: CLI/Server 仍依赖底层 crates（当前状态）
    Steps:
      1. chmod +x scripts/check-dependency-boundary.sh
      2. ./scripts/check-dependency-boundary.sh
      3. 验证输出包含 "FAIL: deepseek-ocr-cli imports deepseek-ocr-core"
    Expected Result: 脚本退出码 non-zero，输出清晰错误列表
    Failure Indicators: 脚本退出码 0（应该失败）
    Evidence: .sisyphus/evidence/task-5-dependency-check-fail.txt

  Scenario: CI workflow YAML 语法验证
    Tool: Bash (yamllint 或 actionlint)
    Steps:
      1. actionlint .github/workflows/dependency-boundary.yml
      2. 验证无语法错误
    Expected Result: actionlint 退出码 0
    Evidence: .sisyphus/evidence/task-5-ci-yaml-valid.txt
  ```

  **Commit**: YES (groups with 1-4)
  - Message: `refactor(pipeline): OcrConfigResolver 完善 + 工具函数`

- [x] 6. OcrRuntimeBuilder::build() 完整实现

  **What to do**:
  - 实现 runtime.rs 的 OcrRuntimeBuilder::build() 方法
  - 调用 OcrConfigResolver::resolve() 获取最终 AppConfig
  - 从 AppConfig 提取 device/precision/observer 等运行时参数
  - 创建 OcrModelManager（传入 device/precision/observer）
  - 返回 OcrRuntime { manager }
  - 错误处理：config resolve 失败、device 初始化失败的错误映射
  - 新增 tests/runtime_builder_test.rs: 测试 build 流程（mock config）

  **Must NOT do**:
  - 不修改 AppConfig 的内部结构
  - 不引入新的设备初始化逻辑（复用 core::runtime::prepare_device_and_dtype）
  - 不改变 OcrRuntime 的公开 API

  **Recommended Agent Profile**:
  > deep 类别适合这种核心实现：需要理解整个配置解析链 + 错误处理 + 资源初始化
  - **Category**: `deep`
    - Reason: 核心实现任务，需要理解多层抽象（Builder -> Resolver -> AppConfig -> Manager）
  - **Skills**: [`git-master`]
    - `git-master`: 查看 runtime.rs 修改历史，确保与现有设计一致
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法，按现有架构实现

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Wave 1)
  - **Parallel Group**: Wave 2 (with Tasks 7-9)
  - **Blocks**: [Tasks 10-21]
  - **Blocked By**: [Tasks 1-5]

  **References**:
  - `crates/pipeline/src/runtime.rs:36-125` - OcrRuntimeBuilder/OcrRuntime 现有骨架
  - `crates/pipeline/src/config.rs:195-268` - OcrConfigResolver::resolve() 实现
  - `crates/core/src/runtime.rs:prepare_device_and_dtype` - 设备初始化函数
  - `crates/config/src/config.rs:AppConfig` - 最终解析结果类型

  **Acceptance Criteria**:
  - [ ] OcrRuntimeBuilder::build() 返回 Result<OcrRuntime>
  - [ ] Test file created: crates/pipeline/tests/runtime_builder_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test runtime_builder_test → PASS

  **QA Scenarios**:
  ```
  Scenario: OcrRuntimeBuilder 正常 build 流程
    Tool: Bash (cargo test)
    Preconditions: 创建临时 config.toml（有效配置）
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test runtime_builder_test -- build_with_valid_config
      2. 验证返回 Ok(OcrRuntime)
      3. 验证 OcrRuntime.manager() 可访问
    Expected Result: 3 tests pass (build_success, manager_accessible, config_resolved)
    Evidence: .sisyphus/evidence/task-6-runtime-build-test.txt

  Scenario: OcrRuntimeBuilder 错误处理 - config 不存在
    Tool: Bash (cargo test)
    Preconditions: 指定不存在的 config 路径
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test runtime_builder_test -- build_with_missing_config
      2. 验证返回 Err 且错误信息包含 "config file not found"
    Expected Result: 错误类型正确，错误信息清晰
    Evidence: .sisyphus/evidence/task-6-runtime-build-error.txt
  ```

  **Commit**: YES (groups with 7-9)
  - Message: `feat(pipeline): OcrRuntimeBuilder/OcrModelManager/OcrPipeline 实现`

- [x] 7. OcrModelManager::load() 完整实现

  **What to do**:
  - 实现 manager.rs 的 OcrModelManager::load() 方法
  - 调用 OcrConfigResolver 或 AppConfig 获取 model_resources（config/tokenizer/weights/snapshot）
  - 调用 prepare_model_paths 准备资源路径
  - 根据 ModelKind 匹配 load_*_model（infer-deepseek/infer-paddleocr/infer-dots/infer-glm）
  - 创建 OcrPipeline（传入 loaded model + tokenizer + engine + observer）
  - 返回 OcrPipelineHandle（带 Arc 包装）
  - 实现缓存：使用 DashMap/ArcSwap 存储已加载模型（避免重复加载）
  - 实现 singleflight：并发请求同模型时仅加载一次
  - 新增 tests/model_manager_test.rs: 测试加载 + 缓存行为

  **Must NOT do**:
  - 不修改 load_*_model 的签名（复用 infer-* crate 的现有函数）
  - 不引入复杂的缓存淘汰策略（先做简单缓存，无 LRU/TTL）
  - 不改变 OcrPipeline 的公开 API

  **Recommended Agent Profile**:
  > deep 类别适合这种核心实现：需要理解模型加载链路 + 并发缓存
  - **Category**: `deep`
    - Reason: 核心实现任务，需要理解 model loading + caching + concurrency
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 6)
  - **Parallel Group**: Wave 2 (with Tasks 6, 8-9)
  - **Blocks**: [Tasks 10-21]
  - **Blocked By**: [Tasks 1-6]

  **References**:
  - `crates/pipeline/src/manager.rs:38-100` - OcrModelManager 现有骨架
  - `crates/server/src/state.rs:192-257` - ModelManager::load_model() 现有实现（参考模式）
  - `crates/cli/src/app.rs:58-130` - CLI 模型加载流程（参考模式）
  - `crates/config/src/config.rs:AppConfig::model_resources` - 获取 model resources
  - `crates/config/src/fs.rs:prepare_model_paths` - 准备资源路径
  - `crates/infer-deepseek/src/lib.rs:load_model` - DeepSeek 模型加载
  - `crates/infer-paddleocr/src/lib.rs:load_model` - PaddleOCR 模型加载
  - `crates/infer-dots/src/lib.rs:load_model` - DotsOCR 模型加载
  - `crates/infer-glm/src/lib.rs:load_model` - GLM 模型加载

  **Acceptance Criteria**:
  - [ ] OcrModelManager::load() 返回 Result<OcrPipelineHandle>
  - [ ] 缓存机制：第二次 load 同模型直接返回缓存（不调用底层 load_*_model）
  - [ ] singleflight：并发 N 请求同模型仅调用一次底层 load
  - [ ] Test file created: crates/pipeline/tests/model_manager_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test model_manager_test → PASS

  **QA Scenarios**:
  ```
  Scenario: OcrModelManager 加载单一模型
    Tool: Bash (cargo test)
    Preconditions: 模型资源已下载（cache 中存在）
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test model_manager_test -- load_single_model
      2. 验证返回 Ok(OcrPipelineHandle)
      3. 验证 handle.pipeline().observer() 可访问
    Expected Result: 3 tests pass (load_success, handle_valid, observer_accessible)
    Evidence: .sisyphus/evidence/task-7-manager-load-test.txt

  Scenario: OcrModelManager 缓存行为 - 第二次加载命中缓存
    Tool: Bash (cargo test)
    Preconditions: 第一次加载已完成
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test model_manager_test -- cache_hit_on_second_load
      2. 验证第二次 load 不调用底层 load_*_model（通过 mock/counter 断言）
    Expected Result: 缓存命中，无重复加载
    Evidence: .sisyphus/evidence/task-7-manager-cache-hit.txt

  Scenario: OcrModelManager singleflight - 并发加载
    Tool: Bash (cargo test)
    Preconditions: 无
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test model_manager_test -- concurrent_load_singleflight
      2. 启动 10 个并发 task 同时 load 同模型
      3. 验证底层 load_*_model 仅调用 1 次（通过 mock/counter 断言）
    Expected Result: 10 个并发请求，底层加载仅 1 次
    Evidence: .sisyphus/evidence/task-7-manager-singleflight.txt
  ```

  **Commit**: YES (groups with 6, 8-9)
  - Message: `feat(pipeline): OcrRuntimeBuilder/OcrModelManager/OcrPipeline 实现`

- [x] 8. OcrPipeline::generate() 完整实现

  **What to do**:
  - 实现 pipeline.rs 的 OcrPipeline::generate() 方法
  - 从 OcrRequest 提取 prompt/images/vision/decode
  - 调用 render_prompt（通过 OcrInferenceEngine::with_default_semantics）
  - 调用 preprocess_images（通过 ModelSemantics）
  - 调用 prepare_inputs（通过 ModelSemantics + RuntimeState）
  - 调用 prefill/decode（通过 OcrInferenceEngine）
  - 调用 stream callback（如果提供）
  - 返回 OcrResponse（text/rendered_prompt/tokens）
  - 错误处理：prompt 渲染失败、图像预处理失败、生成失败
  - 新增 tests/pipeline_generate_test.rs: 测试 generate 流程

  **Must NOT do**:
  - 不修改 OcrInferenceEngine 的语义接口（复用 core 的现有实现）
  - 不改变 stream callback 的签名和行为
  - 不引入新的错误类型（映射到 anyhow::Error）

  **Recommended Agent Profile**:
  > deep 类别适合这种核心实现：需要理解完整生成链路
  - **Category**: `deep`
    - Reason: 核心实现任务，需要理解 prompt rendering + vision + decode 全链路
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 6-7)
  - **Parallel Group**: Wave 2 (with Tasks 6-7, 9)
  - **Blocks**: [Tasks 10-21]
  - **Blocked By**: [Tasks 1-7]

  **References**:
  - `crates/pipeline/src/pipeline.rs:20-91` - OcrPipeline/OcrPipelineHandle 现有骨架
  - `crates/core/src/ocr_inference_engine.rs:OcrInferenceEngine::generate` - 核心生成接口
  - `crates/cli/src/app.rs:145-250` - CLI generate 调用流程（参考模式）
  - `crates/server/src/generation.rs:99-200` - Server generate 调用流程（参考模式）
  - `crates/pipeline/src/api.rs:71-96` - OcrRequest/OcrResponse 定义

  **Acceptance Criteria**:
  - [ ] OcrPipeline::generate() 返回 Result<OcrResponse>
  - [ ] stream callback 正确调用（每次 token 生成）
  - [ ] 错误映射清晰（prompt 渲染/图像预处理/生成失败分开）
  - [ ] Test file created: crates/pipeline/tests/pipeline_generate_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test pipeline_generate_test → PASS

  **QA Scenarios**:
  ```
  Scenario: OcrPipeline generate - 正常流程
    Tool: Bash (cargo test)
    Preconditions: 模型已加载，测试图像存在
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test pipeline_generate_test -- generate_with_single_image
      2. 验证返回 OcrResponse 包含 text/rendered_prompt/prompt_tokens/response_tokens
      3. 验证 text 非空
    Expected Result: 4 tests pass (generate_success, response_fields_valid, text_nonempty, tokens_counted)
    Evidence: .sisyphus/evidence/task-8-pipeline-generate-test.txt

  Scenario: OcrPipeline generate - stream callback 调用
    Tool: Bash (cargo test)
    Preconditions: 提供 stream callback（记录调用次数）
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test pipeline_generate_test -- stream_callback_invoked
      2. 验证 callback 被调用 N 次（N = response_tokens）
    Expected Result: callback 调用次数正确
    Evidence: .sisyphus/evidence/task-8-pipeline-stream-test.txt
  ```

  **Commit**: YES (groups with 6-7, 9)
  - Message: `feat(pipeline): OcrRuntimeBuilder/OcrModelManager/OcrPipeline 实现`

- [x] 9. OcrPipelineHandle 缓存包装

  **What to do**:
  - 完善 pipeline.rs 的 OcrPipelineHandle（已有 Arc 包装）
  - 添加 Clone 实现（已有，通过 Arc）
  - 添加 AsRef<OcrPipeline> trait（便于通用访问）
  - 添加 OcrPipelineHandle::observer() 快捷方法（已有）
  - 添加 OcrPipelineHandle::generate() 委托方法（直接调用 inner.generate）
  - 确保 Send + Sync（用于 server 跨请求共享）
  - 新增 tests/pipeline_handle_test.rs: 测试 Clone/Send/Sync

  **Must NOT do**:
  - 不改变 Arc 包装语义
  - 不引入新的并发原语（保持简单）
  - 不修改 OcrPipeline 的公开方法

  **Recommended Agent Profile**:
  > unspecified-high 适合这种封装任务：需要确保线程安全 + API 一致性
  - **Category**: `unspecified-high`
    - Reason: 封装任务，但需要确保 Send/Sync/Clone 语义正确
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 8)
  - **Parallel Group**: Wave 2 (with Tasks 6-8)
  - **Blocks**: [Tasks 10-21]
  - **Blocked By**: [Tasks 6-8]

  **References**:
  - `crates/pipeline/src/pipeline.rs:67-91` - OcrPipelineHandle 现有定义
  - `crates/server/src/state.rs:23-29` - LoadedModelHandles（参考 Arc 包装模式）

  **Acceptance Criteria**:
  - [ ] OcrPipelineHandle 实现 Clone（Arc 包装）
  - [ ] OcrPipelineHandle 实现 Send + Sync
  - [ ] OcrPipelineHandle::generate() 委托调用 inner.generate()
  - [ ] Test file created: crates/pipeline/tests/pipeline_handle_test.rs
  - [ ] cargo test -p deepseek-ocr-pipeline --test pipeline_handle_test → PASS

  **QA Scenarios**:
  ```
  Scenario: OcrPipelineHandle Clone + Send + Sync
    Tool: Bash (cargo test)
    Steps:
      1. cargo test -p deepseek-ocr-pipeline --test pipeline_handle_test -- clone_send_sync
      2. 验证 handle.clone() 编译通过
      3. 验证 handle 可以在多线程间传递（spawn thread 测试）
    Expected Result: 3 tests pass (clone_compiles, send_compiles, sync_compiles)
    Evidence: .sisyphus/evidence/task-9-handle-traits-test.txt
  ```

  **Commit**: YES (groups with 6-8)
  - Message: `feat(pipeline): OcrRuntimeBuilder/OcrModelManager/OcrPipeline 实现`

- [x] 10. CLI Cargo.toml 依赖迁移

  **What to do**:
  - 修改 crates/cli/Cargo.toml：
    - 移除 deepseek-ocr-core / deepseek-ocr-config / deepseek-ocr-assets
    - 移除 deepseek-ocr-infer-deepseek / infer-paddleocr / infer-dots / infer-glm
    - 新增 deepseek-ocr-pipeline（workspace = true）
    - 保留 clap/image/tokenizers/tracing/serde 等外部依赖
  - 通过 pipeline features 转发必要功能（accelerate/metal/cuda/flash-attn/memlog/cli-debug/glm-trace 等）
  - 验证 cargo check -p deepseek-ocr-cli 通过

  **Must NOT do**:
  - 不修改 app.rs 实现（本任务只改 Cargo.toml）
  - 不引入新的外部依赖
  - 不改变 features 结构（保持向后兼容）

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Cargo.toml 编辑，模式固定
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Wave 2)
  - **Parallel Group**: Wave 3 (with Tasks 11-13)
  - **Blocks**: [Tasks 11-13]
  - **Blocked By**: [Tasks 6-9]

  **References**:
  - `crates/cli/Cargo.toml` - 当前依赖列表
  - `crates/pipeline/Cargo.toml` - pipeline 提供的 features

  **Acceptance Criteria**:
  - [ ] Cargo.toml 移除所有违禁依赖（core/config/assets/infer-*）
  - [ ] Cargo.toml 新增 deepseek-ocr-pipeline
  - [ ] features 正确转发（accelerate/metal/cuda 等）
  - [ ] cargo check -p deepseek-ocr-cli 通过（编译错误是预期的，因为 app.rs 还没改）

  **QA Scenarios**:
  ```
  Scenario: CLI Cargo.toml 依赖检查
    Tool: Bash (cargo metadata)
    Steps:
      1. cargo metadata --format-version 1 | jq '.packages[] | select(.name == "deepseek-ocr-cli") | .dependencies'
      2. 验证不包含 deepseek-ocr-core/config/assets/infer-*
      3. 验证包含 deepseek-ocr-pipeline
    Expected Result: 依赖列表正确
    Evidence: .sisyphus/evidence/task-10-cli-deps-check.txt
  ```

  **Commit**: YES (groups with 11-13)
  - Message: `refactor(cli): 迁移到 pipeline 依赖`

- [x] 11. CLI app.rs 主链路迁移

  **What to do**:
  - 修改 crates/cli/src/app.rs：
    - 移除 AppConfig/LocalFileSystem/prepare_model_paths 导入
    - 移除 load_*_model 导入
    - 移除 OcrInferenceEngine 导入（来自 core）
    - 新增 OcrRuntime/OcrPipeline/OcrRequest 导入（来自 pipeline）
  - 重写 run_inference() 函数：
    - 使用 OcrRuntimeBuilder 构建 runtime
    - 使用 runtime.manager().load() 加载模型
    - 使用 pipeline.generate() 替代 engine.generate()
    - 保持 stream callback 行为不变
  - 保持输出 JSON schema 与 stdout 行为不变（门禁依赖）

  **Must NOT do**:
  - 不改变 CLI 的命令行参数（args.rs 不动）
  - 不改变 stdout 输出格式（JSON/schema/文本）
  - 不改变 quiet 模式行为
  - 不改变 debug/bench 输出格式

  **Recommended Agent Profile**:
  > deep 类别适合这种核心迁移：需要理解 CLI 主链路 + 保持行为不变
  - **Category**: `deep`
    - Reason: 核心迁移任务，需要理解 CLI 完整调用链 + 行为契约
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 10)
  - **Parallel Group**: Wave 3 (with Tasks 10, 12-13)
  - **Blocks**: [Tasks 18-21]
  - **Blocked By**: [Tasks 6-10]

  **References**:
  - `crates/cli/src/app.rs:44-376` - run_inference() 现有实现
  - `crates/pipeline/src/runtime.rs:OcrRuntimeBuilder` - 新入口
  - `crates/pipeline/src/manager.rs:OcrModelManager::load` - 模型加载
  - `crates/pipeline/src/pipeline.rs:OcrPipeline::generate` - 生成入口
  - `crates/pipeline/src/api.rs:OcrRequest/OcrResponse` - 请求/响应类型

  **Acceptance Criteria**:
  - [ ] app.rs 不再 import 违禁 crates（core/config/assets/infer-*）
  - [ ] run_inference() 使用 pipeline API
  - [ ] stream callback 行为不变
  - [ ] stdout 输出格式不变（JSON/schema/文本）
  - [ ] cargo check -p deepseek-ocr-cli 通过

  **QA Scenarios**:
  ```
  Scenario: CLI 主链路迁移 - 编译通过
    Tool: Bash (cargo check)
    Steps:
      1. cargo check -p deepseek-ocr-cli
      2. 验证 0 errors
    Expected Result: 编译通过
    Evidence: .sisyphus/evidence/task-11-cli-check.txt

  Scenario: CLI 行为契约 - stdout 格式不变
    Tool: Bash (cargo run)
    Preconditions: 测试图像存在
    Steps:
      1. cargo run -p deepseek-ocr-cli --release -- --prompt "<image> test" --image <test_image> --device cpu --max-new-tokens 10
      2. 捕获 stdout
      3. 与旧版本 stdout 对比（diff）
    Expected Result: stdout 格式完全一致
    Evidence: .sisyphus/evidence/task-11-cli-stdout-diff.txt
  ```

  **Commit**: YES (groups with 10, 12-13)
  - Message: `refactor(cli): 迁移到 pipeline 依赖`

- [x] 12. CLI debug/bench 兼容

  **What to do**:
  - 修改 crates/cli/src/debug.rs：
    - 确保 debug output JSON 仍包含 rendered_prompt 等字段（benchsuite 依赖）
    - 保持 DebugOutput 结构不变
  - 修改 crates/cli/src/bench.rs：
    - 确保 bench session 仍可捕获 timer events
    - 保持 benchmark 输出格式不变
  - 修改 crates/cli/src/app.rs：
    - 确保 debug::wants_output_json() / debug::write_output_json() 仍可正常调用
    - 确保 bench::maybe_start() / bench::print_summary() 仍可正常调用

  **Must NOT do**:
  - 不改变 debug output JSON schema
  - 不改变 bench output 格式
  - 不引入新的 debug 字段

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: 兼容性任务，需要确保现有 debug/bench 功能不被破坏
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 11)
  - **Parallel Group**: Wave 3 (with Tasks 10-11, 13)
  - **Blocks**: [Tasks 18-21]
  - **Blocked By**: [Tasks 11]

  **References**:
  - `crates/cli/src/debug.rs:1-` - debug output 逻辑
  - `crates/cli/src/bench.rs:1-` - bench session 逻辑
  - `crates/cli/src/app.rs:270-310` - debug output JSON 写入

  **Acceptance Criteria**:
  - [ ] debug output JSON 包含所有必需字段（rendered_prompt/tokens/decoded 等）
  - [ ] bench session 可正常启动/ finalize
  - [ ] cargo check -p deepseek-ocr-cli 通过

  **QA Scenarios**:
  ```
  Scenario: CLI debug output JSON - 字段完整性
    Tool: Bash (cargo run)
    Steps:
      1. cargo run -p deepseek-ocr-cli --release -- --prompt "<image> test" --image <test_image> --device cpu --debug-output-json
      2. 解析 JSON 输出
      3. 验证包含 rendered_prompt/prompt_tokens/response_tokens/generated_tokens
    Expected Result: JSON schema 完整
    Evidence: .sisyphus/evidence/task-12-cli-debug-json.txt
  ```

  **Commit**: YES (groups with 10-11, 13)
  - Message: `refactor(cli): 迁移到 pipeline 依赖`

- [x] 13. CLI 行为契约测试

  **What to do**:
  - 创建 crates/cli/tests/behavior_contract_test.rs
  - 测试 CLI stdout 格式（JSON/schema/文本）
  - 测试 CLI exit code（成功/失败场景）
  - 测试 quiet 模式（stdout 干净）
  - 测试 debug output JSON（字段完整性）
  - 与 baselines 对比（确保不回退）

  **Must NOT do**:
  - 不改变现有行为（只测试，不改行为）
  - 不引入脆弱的测试（依赖具体文本）
  - 不测试实现细节（只测试对外契约）

  **Recommended Agent Profile**:
  > deep 类别适合这种测试任务：需要理解 CLI 完整行为契约
  - **Category**: `deep`
    - Reason: 行为契约测试，需要全面理解 CLI 对外契约
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 11-12)
  - **Parallel Group**: Wave 3 (with Tasks 10-12)
  - **Blocks**: [Tasks 18-21]
  - **Blocked By**: [Tasks 11-12]

  **References**:
  - `crates/cli/tests/` (新建) - 测试目录
  - `baselines/sample/` - 基线输入/输出

  **Acceptance Criteria**:
  - [ ] Test file created: crates/cli/tests/behavior_contract_test.rs
  - [ ] cargo test -p deepseek-ocr-cli --test behavior_contract_test → PASS
  - [ ] 测试覆盖 stdout/exit_code/quiet_mode/debug_json

  **QA Scenarios**:
  ```
  Scenario: CLI 行为契约 - stdout 格式
    Tool: Bash (cargo test)
    Steps:
      1. cargo test -p deepseek-ocr-cli --test behavior_contract_test -- stdout_format_matches_baseline
      2. 对比 baselines/sample/output.json
    Expected Result: stdout 格式完全一致
    Evidence: .sisyphus/evidence/task-13-cli-behavior-test.txt
  ```

  **Commit**: YES (groups with 10-12)
  - Message: `refactor(cli): 迁移到 pipeline 依赖`

- [x] 14. Server Cargo.toml 依赖迁移

  **What to do**:
  - 修改 crates/server/Cargo.toml：
    - 移除 deepseek-ocr-core / deepseek-ocr-config / deepseek-ocr-assets
    - 移除 deepseek-ocr-infer-deepseek / infer-paddleocr / infer-dots / infer-glm
    - 新增 deepseek-ocr-pipeline（workspace = true）
    - 保留 rocket/base64/uuid/thiserror/reqwest/tokio-stream 等外部依赖
  - 通过 pipeline features 转发必要功能（accelerate/metal/cuda/flash-attn/memlog 等）
  - 验证 cargo check -p deepseek-ocr-server 通过

  **Must NOT do**:
  - 不修改 state.rs/app.rs/generation.rs 实现（本任务只改 Cargo.toml）
  - 不引入新的外部依赖
  - 不改变 features 结构

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Cargo.toml 编辑，模式固定
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Wave 2)
  - **Parallel Group**: Wave 4 (with Tasks 15-17)
  - **Blocks**: [Tasks 15-17]
  - **Blocked By**: [Tasks 6-9]

  **References**:
  - `crates/server/Cargo.toml` - 当前依赖列表
  - `crates/pipeline/Cargo.toml` - pipeline 提供的 features

  **Acceptance Criteria**:
  - [ ] Cargo.toml 移除所有违禁依赖
  - [ ] Cargo.toml 新增 deepseek-ocr-pipeline
  - [ ] features 正确转发
  - [ ] cargo check -p deepseek-ocr-server 通过（编译错误是预期的）

  **QA Scenarios**:
  ```
  Scenario: Server Cargo.toml 依赖检查
    Tool: Bash (cargo metadata)
    Steps:
      1. cargo metadata --format-version 1 | jq '.packages[] | select(.name == "deepseek-ocr-server") | .dependencies'
      2. 验证不包含违禁依赖
      3. 验证包含 deepseek-ocr-pipeline
    Expected Result: 依赖列表正确
    Evidence: .sisyphus/evidence/task-14-server-deps-check.txt
  ```

  **Commit**: YES (groups with 15-17)
  - Message: `refactor(server): 迁移到 pipeline 依赖`

- [x] 15. Server state.rs 迁移

  **What to do**:
  - 修改 crates/server/src/state.rs：
    - 移除 AppConfig/LocalFileSystem/prepare_model_paths 导入
    - 移除 load_*_model 导入
    - 移除 OcrInferenceEngine 导入（来自 core）
    - 新增 OcrRuntime/OcrModelManager/OcrPipelineHandle 导入（来自 pipeline）
  - 重写 ModelManager::load_model()：
    - 使用 OcrModelManager::load() 替代直接调用 load_*_model
  - 重写 AppState::bootstrap()：
    - 使用 OcrRuntimeBuilder 构建 runtime
  - 保持模型缓存行为不变（避免重复加载）

  **Must NOT do**:
  - 不改变 Server 的命令行参数（args.rs 不动）
  - 不改变 API 输出格式（响应体/状态码/错误码）
  - 不改变模型缓存语义

  **Recommended Agent Profile**:
  > deep 类别适合这种核心迁移：需要理解 Server 状态管理 + 缓存
  - **Category**: `deep`
    - Reason: 核心迁移任务，需要理解 Server 状态管理 + 模型缓存
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 14)
  - **Parallel Group**: Wave 4 (with Tasks 14, 16-17)
  - **Blocks**: [Tasks 18-21]
  - **Blocked By**: [Tasks 6-9, 14]

  **References**:
  - `crates/server/src/state.rs:38-257` - AppState/ModelManager 现有实现
  - `crates/pipeline/src/runtime.rs:OcrRuntimeBuilder` - 新入口
  - `crates/pipeline/src/manager.rs:OcrModelManager::load` - 模型加载

  **Acceptance Criteria**:
  - [ ] state.rs 不再 import 违禁 crates
  - [ ] ModelManager::load_model() 使用 pipeline API
  - [ ] 模型缓存行为不变
  - [ ] cargo check -p deepseek-ocr-server 通过

  **QA Scenarios**:
  ```
  Scenario: Server state.rs 迁移 - 编译通过
    Tool: Bash (cargo check)
    Steps:
      1. cargo check -p deepseek-ocr-server
      2. 验证 0 errors
    Expected Result: 编译通过
    Evidence: .sisyphus/evidence/task-15-server-check.txt
  ```

  **Commit**: YES (groups with 14, 16-17)
  - Message: `refactor(server): 迁移到 pipeline 依赖`

- [x] 16. Server generation.rs 迁移

  **What to do**:
  - 修改 crates/server/src/generation.rs：
    - 移除 OcrInferenceEngine/OcrInferenceRequest 导入（来自 core）
    - 新增 OcrPipeline/OcrRequest/OcrResponse 导入（来自 pipeline）
  - 重写 generate_blocking()：
    - 使用 OcrPipeline::generate() 替代 engine.generate()
    - 保持 stream callback 行为不变
  - 保持 API 输出格式不变（响应体/状态码/错误码）

  **Must NOT do**:
  - 不改变 API 输出格式
  - 不改变 stream callback 行为
  - 不改变错误映射逻辑

  **Recommended Agent Profile**:
  > deep 类别适合这种核心迁移：需要理解生成链路
  - **Category**: `deep`
    - Reason: 核心迁移任务，需要理解 generation 完整链路
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 15)
  - **Parallel Group**: Wave 4 (with Tasks 14-15, 17)
  - **Blocks**: [Tasks 18-21]
  - **Blocked By**: [Tasks 15]

  **References**:
  - `crates/server/src/generation.rs:1-328` - generate_blocking/generate_async 现有实现
  - `crates/pipeline/src/pipeline.rs:OcrPipeline::generate` - 新入口
  - `crates/pipeline/src/api.rs:OcrRequest/OcrResponse` - 请求/响应类型

  **Acceptance Criteria**:
  - [ ] generation.rs 不再 import 违禁 crates
  - [ ] generate_blocking() 使用 pipeline API
  - [ ] stream callback 行为不变
  - [ ] API 输出格式不变
  - [ ] cargo check -p deepseek-ocr-server 通过

  **QA Scenarios**:
  ```
  Scenario: Server generation.rs 迁移 - 编译通过
    Tool: Bash (cargo check)
    Steps:
      1. cargo check -p deepseek-ocr-server
      2. 验证 0 errors
    Expected Result: 编译通过
    Evidence: .sisyphus/evidence/task-16-server-generation-check.txt
  ```

  **Commit**: YES (groups with 14-15, 17)
  - Message: `refactor(server): 迁移到 pipeline 依赖`

- [x] 17. Server API 契约测试

  **What to do**:
  - 创建 crates/server/tests/api_contract_test.rs
  - 测试 /v1/models 端点（模型列表）
  - 测试 /v1/responses 端点（生成）
  - 测试 /v1/chat/completions 端点（对话生成）
  - 测试错误处理（无效模型/空 prompt/图像加载失败）
  - 与 baselines 对比（确保不回退）

  **Must NOT do**:
  - 不改变现有 API（只测试，不改行为）
  - 不引入脆弱的测试（依赖具体文本）
  - 不测试实现细节（只测试对外契约）

  **Recommended Agent Profile**:
  > deep 类别适合这种测试任务：需要理解 Server 完整 API 契约
  - **Category**: `deep`
    - Reason: API 契约测试，需要全面理解 Server 对外契约
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 15-16)
  - **Parallel Group**: Wave 4 (with Tasks 14-16)
  - **Blocks**: [Tasks 18-21]
  - **Blocked By**: [Tasks 15-16]

  **References**:
  - `crates/server/tests/` (新建) - 测试目录
  - `crates/server/src/routes.rs` - API 路由定义

  **Acceptance Criteria**:
  - [ ] Test file created: crates/server/tests/api_contract_test.rs
  - [ ] cargo test -p deepseek-ocr-server --test api_contract_test → PASS
  - [ ] 测试覆盖 models/responses/chat_completions 端点

  **QA Scenarios**:
  ```
  Scenario: Server API 契约 - /v1/models
    Tool: Bash (cargo test)
    Steps:
      1. cargo test -p deepseek-ocr-server --test api_contract_test -- models_endpoint_returns_list
      2. 验证返回 JSON 包含模型 ID 列表
    Expected Result: 模型列表正确
    Evidence: .sisyphus/evidence/task-17-server-models-test.txt
  ```

  **Commit**: YES (groups with 14-16)
  - Message: `refactor(server): 迁移到 pipeline 依赖`

- [x] 18. workspace clippy --all-features

  **What to do**:
  - 运行 cargo clippy --workspace --all-targets --all-features -- -D warnings
  - 修复所有 clippy warnings/errors
  - 确保无 `#[allow(...)]` 掩盖问题
  - 确保无 AI slop（excessive comments/over-abstraction/generic names）

  **Must NOT do**:
  - 不用 #[allow(...)] 掩盖问题
  - 不引入新的 clippy 问题
  - 不改业务逻辑（只修复 style 问题）

  **Recommended Agent Profile**:
  > deep 类别适合这种质量门禁：需要系统性修复 clippy 问题
  - **Category**: `deep`
    - Reason: 质量门禁任务，需要系统性修复所有 clippy 问题
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `artistry`: 不需要非常规解法

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Waves 3-4)
  - **Parallel Group**: Wave 5 (with Tasks 19-21)
  - **Blocks**: [Tasks 19-21, Final Verification]
  - **Blocked By**: [Tasks 11-13, 15-17]

  **References**:
  - 全 workspace

  **Acceptance Criteria**:
  - [ ] cargo clippy --workspace --all-targets --all-features -- -D warnings → 0 errors, 0 warnings
  - [ ] 无 #[allow(...)] 掩盖

  **QA Scenarios**:
  ```
  Scenario: workspace clippy 全绿
    Tool: Bash (cargo clippy)
    Steps:
      1. cargo clippy --workspace --all-targets --all-features -- -D warnings
      2. 验证输出不包含 "error" 或 "warning"
    Expected Result: 0 errors, 0 warnings
    Evidence: .sisyphus/evidence/task-18-clippy-all-features.txt
  ```

  **Commit**: YES (groups with 19-21)
  - Message: `test(workspace): 行为契约测试 + CI 门禁`

- [x] 19. benchsuite failfast 单长 case

  **What to do**:
  - 运行 benchsuite failfast 门禁（单代表长 case + accelerate + 8192）
  - 验证 strict/prompt/tokens 全通过
  - 记录 run_id 和 summary.json 路径
  - 如失败，分析原因并修复

  **Must NOT do**:
  - 不改 benchsuite 脚本（只修复代码）
  - 不放宽门禁阈值
  - 不用 allow 掩盖失败

  **Recommended Agent Profile**:
  > unspecified-high 适合这种验证任务：需要运行门禁并分析结果
  - **Category**: `unspecified-high`
    - Reason: 验证任务，需要运行门禁并分析失败原因
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析（失败原因通常清晰）

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Task 18)
  - **Parallel Group**: Wave 5 (with Tasks 18, 20-21)
  - **Blocks**: [Tasks 20-21, Final Verification]
  - **Blocked By**: [Tasks 18]

  **References**:
  - `baselines/benchsuite/runs/` - 门禁输出目录

  **Acceptance Criteria**:
  - [ ] benchsuite failfast 通过（strict/prompt/tokens 全绿）
  - [ ] summary.json 路径可追溯

  **QA Scenarios**:
  ```
  Scenario: benchsuite failfast 通过
    Tool: Bash (python -m benchsuite.cli)
    Steps:
      1. python -m benchsuite.cli matrix-gate --run <run_id> --include-models deepseek-ocr --include-devices cpu --include-precision f32 --include-runtime-features accelerate --cases <representative> --max-new-tokens 8192 --failfast
      2. 验证退出码 0
      3. 验证 baselines/benchsuite/runs/<run_id>/matrix/summary.json 存在
    Expected Result: strict/prompt/tokens 全通过
    Evidence: .sisyphus/evidence/task-19-benchsuite-failfast.txt
  ```

  **Commit**: YES (groups with 18, 20-21)
  - Message: `test(workspace): 行为契约测试 + CI 门禁`

- [x] 20. 依赖边界 CI 门禁验证

  **What to do**:
  - 运行 scripts/check-dependency-boundary.sh
  - 验证 CLI/Server 不再 import 违禁 crates
  - 如失败，分析原因并修复（移除违禁 import）

  **Must NOT do**:
  - 不改脚本逻辑（只修复代码）
  - 不放宽检查规则
  - 不用 allow 掩盖失败

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: CI 脚本验证，模式固定
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 11/15)
  - **Parallel Group**: Wave 5 (with Tasks 18-19, 21)
  - **Blocks**: [Final Verification]
  - **Blocked By**: [Tasks 11, 15]

  **References**:
  - `scripts/check-dependency-boundary.sh` - 检查脚本

  **Acceptance Criteria**:
  - [ ] scripts/check-dependency-boundary.sh → 退出码 0
  - [ ] 无违禁 import

  **QA Scenarios**:
  ```
  Scenario: 依赖边界检查通过
    Tool: Bash (scripts/check-dependency-boundary.sh)
    Steps:
      1. ./scripts/check-dependency-boundary.sh
      2. 验证退出码 0
      3. 验证输出包含 "PASS: No forbidden imports found"
    Expected Result: 无违禁 import
    Evidence: .sisyphus/evidence/task-20-dependency-boundary-pass.txt
  ```

  **Commit**: YES (groups with 18-19, 21)
  - Message: `test(workspace): 行为契约测试 + CI 门禁`

- [ ] 21. Git cleanup + tagging

  **What to do**:
  - git add -A
  - git commit -m "chore: Stage 2E 完成"
  - git tag v0.6.0-stage2e
  - git push origin feature/refactor --tags

  **Must NOT do**:
  - 不 push 到 main/master（除非用户要求）
  - 不 force push
  - 不改写历史

  **Recommended Agent Profile**:
  - **Category**: `git`
    - Reason: Git 操作，模式固定
  - **Skills**: [`git-master`]
  - **Skills Evaluated but Omitted**:
    - `deep`: 不需要深度分析

  **Parallelization**:
  - **Can Run In Parallel**: NO (depends on Tasks 18-20)
  - **Parallel Group**: Wave 5 (with Tasks 18-20)
  - **Blocks**: [Final Verification]
  - **Blocked By**: [Tasks 18-20]

  **References**:
  - Git history

  **Acceptance Criteria**:
  - [ ] Git commit 创建
  - [ ] Git tag 创建
  - [ ] Git push 成功

  **QA Scenarios**:
  ```
  Scenario: Git tag 创建
    Tool: Bash (git tag)
    Steps:
      1. git tag -l v0.6.0-stage2e
      2. 验证 tag 存在
    Expected Result: tag 存在
    Evidence: .sisyphus/evidence/task-21-git-tag.txt
  ```

  **Commit**: N/A (本身就是 commit 任务)
  - Message: `chore: Stage 2E 完成`

---

## Final Verification Wave (MANDATORY — after ALL implementation tasks)

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, curl endpoint, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in .sisyphus/evidence/. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run `tsc --noEmit` + linter + `cargo test`. Review all changed files for: `as any`/`@ts-ignore`, empty catches, console.log in prod, commented-out code, unused imports. Check AI slop: excessive comments, over-abstraction, generic names (data/result/item/temp).
  Output: `Build [PASS/FAIL] | Lint [PASS/FAIL] | Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high` (+ `playwright` skill if UI)
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration (features working together, not isolation). Test edge cases: empty state, invalid input, rapid actions. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual diff (git log/diff). Verify 1:1 — everything in spec was built (no missing), nothing beyond spec was built (no creep). Check "Must NOT do" compliance. Detect cross-task contamination: Task N touching Task M's files. Flag unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

- **1**: `refactor(pipeline): OcrConfigResolver 完善 + 工具函数` — api.rs, config.rs, ext.rs, model.rs, observer.rs, fs.rs — `cargo check -p deepseek-ocr-pipeline`
- **2**: `feat(pipeline): OcrRuntimeBuilder/OcrModelManager/OcrPipeline 实现` — runtime.rs, manager.rs, pipeline.rs — `cargo check -p deepseek-ocr-pipeline`
- **3**: `refactor(cli): 迁移到 pipeline 依赖` — Cargo.toml, app.rs — `cargo check -p deepseek-ocr-cli`
- **4**: `refactor(server): 迁移到 pipeline 依赖` — Cargo.toml, state.rs, generation.rs — `cargo check -p deepseek-ocr-server`
- **5**: `test(workspace): 行为契约测试 + CI 门禁` — tests/, .github/workflows/ — `cargo test --workspace`
- **FINAL**: `chore: Stage 2E 完成` — git tag v0.6.0-stage2e

---

## Success Criteria

### Verification Commands
```bash
cargo check --workspace --all-targets --all-features  # Expected: Finished, 0 errors
cargo clippy --workspace --all-targets --all-features -- -D warnings  # Expected: 0 warnings
cargo test --workspace  # Expected: all tests pass
python -m benchsuite.cli matrix-gate --run <run_id> --include-models deepseek-ocr --include-devices cpu --include-precision f32 --include-runtime-features accelerate --cases <representative> --max-new-tokens 8192 --failfast  # Expected: strict/prompt/tokens all pass
```

### Final Checklist
- [ ] All "Must Have" present (配置优先级/模型缓存/错误映射/观测稳定/线程安全)
- [ ] All "Must NOT Have" absent (6 项 scope creep 禁止)
- [ ] All tests pass (cargo test + benchsuite)
- [ ] CLI stdout/JSON schema 完全一致
- [ ] Server API 响应体/状态码/错误码一致
- [ ] 依赖边界 CI 门禁生效
