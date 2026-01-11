# LMCache 与 vLLM 集成：动态交叉分配策略

## 📋 集成架构概览

```
┌─────────────────────────────────────────────────────────────┐
│  vLLM Scheduler                                             │
│  - Request管理 (prefill/decode 检测)                       │
│  - get_num_new_matched_tokens(): 缓存命中查询               │
│  - on_step_begin(): 步骤初始化                              │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  LMCacheConnectorV1Impl                                      │
│  (lmcache/integration/vllm/vllm_v1_adapter.py)              │
│  - RequestTracker: 跟踪单个请求状态                         │
│  - is_decode_phase: 当前请求是否在decode阶段               │
│  - _request_trackers: 所有运行中请求的状态                  │
│  - get_num_new_matched_tokens()：接收来自调度器的请求       │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  动态交织分配策略层 ⭐️ (新增)                              │
│  - WorkloadPhaseDetector: 阶段检测                          │
│  - DynamicInterleaveAllocator: 动态权重调整                 │
│  - 回调: update_phase(phase, num_requests, metrics)        │
└─────────────┬───────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│  LMCacheEngine (lmcache/v1/cache_engine.py)                │
│  - LazyMixedMemoryAllocator: 分配DRAM/CXL内存               │
│  - store()/retrieve(): 存取操作                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔗 关键集成点解析

### 1. **Request 阶段检测点**

**现有代码位置**: [vllm_v1_adapter.py](lmcache/integration/vllm/vllm_v1_adapter.py#L133)

```python
@dataclass
class RequestTracker:
    req_id: str
    is_decode_phase: bool = False  # ⭐️ 关键标志
    
    def update_tokens(self, new_token_ids):
        # 当每次调度时触发
        if len(new_token_ids) == 1:  # 单token = decode阶段
            self.is_decode_phase = True
```

**集成点**：
- **Prefill 阶段**：`is_decode_phase = False`，多个 token 一起处理
- **Decode 阶段**：`is_decode_phase = True`，每次只生成 1 个 token

### 2. **调度器端 Scheduler Hooks**

**关键方法**: [get_num_new_matched_tokens()](lmcache/integration/vllm/vllm_v1_adapter.py#L1506)

```python
def get_num_new_matched_tokens(
    self,
    request: "Request",
    num_computed_tokens: int,
) -> Optional[int]:
    """每次调度时调用，可提取系统状态"""
    # 访问点：
    # - self._request_trackers: 所有请求的状态
    # - request 对象：当前请求信息
    # - 可计算：当前prefill/decode的请求数量比例
```

**最小侵入集成点**：
```python
# 在方法开始处添加回调通知
if hasattr(self, '_workload_detector'):
    num_prefill = sum(1 for t in self._request_trackers.values() 
                      if not t.is_decode_phase)
    num_decode = len(self._request_trackers) - num_prefill
    self._workload_detector.update(
        num_prefill_reqs=num_prefill,
        num_decode_reqs=num_decode,
    )
```

### 3. **LMCacheManager 初始化点**

**现有代码位置**: [vllm_v1_adapter.py#L434](lmcache/integration/vllm/vllm_v1_adapter.py#L434)

```python
self._manager = LMCacheManager(
    config=config,
    vllm_config=vllm_config,
    role=role.name.lower(),  # "scheduler" or "worker"
    connector=self,
)
```

**集成点**：
- 在 `LMCacheManager.start_services()` 后初始化 WorkloadDetector
- 向 LMCacheEngine 注册回调函数

### 4. **内存分配的触发点**

**现有代码位置**: [cache_engine.py](lmcache/v1/cache_engine.py)

```python
def process_tokens(...):
    """处理请求中的token"""
    for token_chunk in chunks:
        mem_obj = self.memory_allocator.allocate(
            shape, dtype, fmt  # ⭐️ 分配触发点
        )
```

**集成策略**：
- 传入 `allocation_hint` 参数，告知是 prefill 还是 decode
- 分配器根据阶段调整 DRAM/CXL 比例

---

## 🛠️ 实现方案（最小侵入）

### Phase 1: 工作负载检测器

**文件**: `lmcache/v1/storage_backend/workload_detector.py` (新建)

```python
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from typing import Optional
import time

@dataclass
class WorkloadMetrics:
    num_prefill_requests: int = 0
    num_decode_requests: int = 0
    timestamp: float = 0.0
    
    def get_phase(self) -> str:
        """判断当前阶段"""
        total = self.num_prefill_requests + self.num_decode_requests
        if total == 0:
            return "idle"
        
        prefill_ratio = self.num_prefill_requests / total
        # 主要是prefill: >50% prefill requests
        if prefill_ratio > 0.5:
            return "prefill"
        # 混合: 20%-50% prefill
        elif prefill_ratio > 0.2:
            return "mixed"
        # 主要是decode: <20% prefill
        else:
            return "decode"

class WorkloadPhaseDetector:
    """检测 LLM 推理的工作负载阶段"""
    
    def __init__(self, update_interval: int = 100):
        self.metrics = WorkloadMetrics()
        self.update_interval = update_interval
        self.call_count = 0
        self.current_phase = "unknown"
    
    def update(self, num_prefill: int, num_decode: int):
        """更新工作负载指标"""
        self.call_count += 1
        
        if self.call_count % self.update_interval == 0:
            self.metrics.num_prefill_requests = num_prefill
            self.metrics.num_decode_requests = num_decode
            self.metrics.timestamp = time.time()
            
            new_phase = self.metrics.get_phase()
            if new_phase != self.current_phase:
                self.current_phase = new_phase
                return True  # 阶段发生变化
        return False
    
    def get_current_phase(self) -> str:
        return self.current_phase
```

### Phase 2: 动态交织分配器

**文件**: `lmcache/v1/lazy_interleave_allocator.py` (新建)

```python
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import torch
from lmcache.v1.lazy_memory_allocator import LazyMixedMemoryAllocator
from lmcache.v1.memory_management import MemoryFormat, MemoryObj

class DynamicInterleaveAllocator(LazyMixedMemoryAllocator):
    """根据工作负载阶段动态调整DRAM/CXL比例的分配器"""
    
    def __init__(
        self,
        size: int,
        config,
        workload_detector,
        prefill_dram_ratio: float = 0.7,  # prefill时70% DRAM
        decode_dram_ratio: float = 1.0,   # decode时100% DRAM
    ):
        super().__init__(size, config)
        self.workload_detector = workload_detector
        self.prefill_dram_ratio = prefill_dram_ratio
        self.decode_dram_ratio = decode_dram_ratio
        
    def allocate(
        self,
        shape: torch.Size,
        dtype: Optional[torch.dtype] = None,
        fmt: MemoryFormat = MemoryFormat.UNDEFINED,
        allocation_hint: Optional[str] = None,
    ) -> Optional[MemoryObj]:
        """
        根据工作负载阶段动态调整分配策略
        
        allocation_hint: "prefill", "decode", 或 None
        """
        current_phase = self.workload_detector.get_current_phase()
        
        # 确定目标DRAM比例
        if allocation_hint == "decode" or current_phase == "decode":
            target_dram_ratio = self.decode_dram_ratio
        else:  # prefill或mixed
            target_dram_ratio = self.prefill_dram_ratio
        
        # 计算当前DRAM利用率
        dram_util = self._get_dram_utilization()
        
        # 如果DRAM利用率超过目标比例，强制使用CXL
        if dram_util > target_dram_ratio:
            # 尝试分配到CXL
            mem_obj = self._allocate_from_cxl(shape, dtype, fmt)
            if mem_obj:
                return mem_obj
        
        # 默认分配到DRAM
        return super().allocate(shape, dtype, fmt)
    
    def _get_dram_utilization(self) -> float:
        """计算DRAM利用率 (0.0-1.0)"""
        # 假设有两个内存池：dram_allocator 和 cxl_allocator
        if hasattr(self, 'pin_allocator'):  # DRAM
            dram_total = self.pin_allocator.total_size
            dram_used = dram_total - sum(
                b.size for b in self.pin_allocator.explicit_list
            )
            return dram_used / dram_total if dram_total > 0 else 0.0
        return 0.0
    
    def _allocate_from_cxl(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """尝试从CXL分配内存"""
        # 这里需要调用底层的CXL分配器
        # 具体实现取决于系统配置
        if hasattr(self, 'allocator'):  # CXL或其他
            return self.allocator.allocate(shape, dtype)
        return None
```

### Phase 3: 集成到 vLLM 连接器

**修改文件**: [lmcache/integration/vllm/vllm_v1_adapter.py](lmcache/integration/vllm/vllm_v1_adapter.py)

```python
# 在 LMCacheConnectorV1Impl.__init__() 中添加

def __init__(self, vllm_config, role, parent):
    # ... 现有代码 ...
    
    # ⭐️ 新增：初始化工作负载检测器和动态分配器
    if role == KVConnectorRole.WORKER:  # 仅在worker端需要
        from lmcache.v1.storage_backend.workload_detector import (
            WorkloadPhaseDetector,
        )
        from lmcache.v1.lazy_interleave_allocator import (
            DynamicInterleaveAllocator,
        )
        
        if config.get("enable_dynamic_interleave", False):
            self._workload_detector = WorkloadPhaseDetector(
                update_interval=config.get("workload_detect_interval", 100)
            )
            
            # 替换分配器
            if self.lmcache_engine and hasattr(
                self.lmcache_engine, 'memory_allocator'
            ):
                old_allocator = self.lmcache_engine.memory_allocator
                self.lmcache_engine.memory_allocator = (
                    DynamicInterleaveAllocator(
                        size=old_allocator.total_size,
                        config=config,
                        workload_detector=self._workload_detector,
                        prefill_dram_ratio=config.get(
                            "prefill_dram_ratio", 0.7
                        ),
                        decode_dram_ratio=config.get(
                            "decode_dram_ratio", 1.0
                        ),
                    )
                )
```

### Phase 4: 配置扩展

**修改文件**: [lmcache/v1/config.py](lmcache/v1/config.py)

在 `_CONFIG_DEFINITIONS` 中添加：

```python
_CONFIG_DEFINITIONS: dict[str, dict[str, Any]] = {
    # ... 现有配置 ...
    
    # 动态交织配置
    "enable_dynamic_interleave": {
        "type": bool,
        "default": False,
        "env_converter": _to_bool,
    },
    "prefill_dram_ratio": {
        "type": float,
        "default": 0.7,
        "env_converter": float,
    },
    "decode_dram_ratio": {
        "type": float,
        "default": 1.0,
        "env_converter": float,
    },
    "workload_detect_interval": {
        "type": int,
        "default": 100,
        "env_converter": int,
    },
}
```

---

## 🚀 使用方式

### 1. **启用动态交织**

```python
from lmcache.v1.config import LMCacheEngineConfig

config = LMCacheEngineConfig.from_defaults(
    enable_dynamic_interleave=True,
    prefill_dram_ratio=0.7,    # Prefill: 70% DRAM, 30% CXL
    decode_dram_ratio=1.0,     # Decode: 100% DRAM
)
```

### 2. **环境变量配置**

```bash
export LMCACHE_ENABLE_DYNAMIC_INTERLEAVE=true
export LMCACHE_PREFILL_DRAM_RATIO=0.7
export LMCACHE_DECODE_DRAM_RATIO=1.0
```

### 3. **vLLM 集成**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b",
    kv_transfer_config={
        "kv_connector_extra_config": {
            "lmcache.enable_dynamic_interleave": True,
            "lmcache.prefill_dram_ratio": 0.7,
            "lmcache.decode_dram_ratio": 1.0,
        }
    }
)
```

---

## 📊 工作流程

```
┌─────────────────┐
│ vLLM Scheduler  │
│  Step k         │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ RequestTracker.update_tokens()            │
│ - 检测请求是否进入decode阶段             │
│ - is_decode_phase = (len(new_tokens)==1)  │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ get_num_new_matched_tokens()              │
│ [⭐️ 新增集成点]                          │
│ if workload_detector:                    │
│   num_prefill = sum(not t.is_decode...)  │
│   num_decode = len(...) - num_prefill    │
│   workload_detector.update(...)          │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ WorkloadPhaseDetector.update()            │
│ - 计算prefill/decode请求比例              │
│ - 判断当前阶段：prefill/mixed/decode     │
│ - 如果阶段变化，返回True                  │
└────────┬─────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│ LMCacheEngine.store()                     │
│ - 调用DynamicInterleaveAllocator.allocate()│
│ - 根据当前阶段选择DRAM或CXL               │
└──────────────────────────────────────────┘
```

---

## 🎯 最小侵入原则

| 修改项 | 侵入程度 | 原因 |
|--------|---------|------|
| 新建 workload_detector.py | ✅ 零侵入 | 完全独立模块 |
| 新建 lazy_interleave_allocator.py | ✅ 零侵入 | 继承现有分配器 |
| vllm_v1_adapter.py | ⚠️ 最小 | 仅添加3-5行初始化代码 + 1个回调 |
| config.py | ⚠️ 最小 | 仅添加新配置项定义 |
| cache_engine.py | ✅ 无需修改 | 通过参数传递 |
| memory_management.py | ✅ 无需修改 | 通过继承扩展 |

---

## 📈 性能评估

### 预期收益

基于论文中的数据，假设配置：
- **Prefill 阶段**：DRAM:CXL = 5:2
  - 预期 TTFT 改善：~15-20%
  - 原因：利用 CXL 的全双工特性（读写混合）

- **Decode 阶段**：DRAM:CXL = 1:0 (100% DRAM)
  - 预期延迟改善：~5-10%
  - 原因：避免 CXL 随机访问惩罚

### 监控指标

```python
# 在 WorkloadPhaseDetector 中可添加
@dataclass
class PhaseMetrics:
    prefill_tokens_allocated_dram: int = 0
    prefill_tokens_allocated_cxl: int = 0
    decode_tokens_allocated_dram: int = 0
    decode_tokens_allocated_cxl: int = 0
    
    def get_dram_utilization(self) -> float:
        total = (
            self.prefill_tokens_allocated_dram +
            self.decode_tokens_allocated_dram
        )
        if total == 0:
            return 0.0
        return (
            self.prefill_tokens_allocated_dram +
            self.decode_tokens_allocated_dram
        ) / total
```

---

## 🔍 调试技巧

### 1. **启用详细日志**

```bash
export LMCACHE_LOG_LEVEL=DEBUG
```

### 2. **验证阶段检测**

在 `get_num_new_matched_tokens()` 中添加：

```python
logger.info(
    f"Phase: {self._workload_detector.get_current_phase()}, "
    f"Prefill reqs: {num_prefill}, Decode reqs: {num_decode}"
)
```

### 3. **内存分配追踪**

```python
class DynamicInterleaveAllocator(...):
    def allocate(self, ...):
        phase = self.workload_detector.get_current_phase()
        logger.debug(
            f"Allocating {shape} in phase {phase}, "
            f"DRAM util: {self._get_dram_utilization():.2%}"
        )
        return super().allocate(...)
```

---

## ✅ 验证清单

- [ ] WorkloadPhaseDetector 正确识别prefill/decode阶段
- [ ] DynamicInterleaveAllocator 根据阶段调整比例
- [ ] vLLM 连接器成功初始化动态分配器
- [ ] 配置参数通过环境变量生效
- [ ] 内存分配符合预期比例
- [ ] 性能监控指标正常收集

