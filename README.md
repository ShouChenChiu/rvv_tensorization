## TVM
#### /home/scchiu/packages/mlc_llm_tensorization/tvm/python/tvm/relax/frontend/nn/llm/kv_cache.py
修改了_attention_prefill_cpu()的 impl 原本是使用seq 執行完 q 跟 kv 的 loop， 更改成 對 q 還有 kv 做 split 並把做 gemm 的部分切出單獨的block方便後續的 tensorization
```
def _attention_prefill_cpu()
    ...
    q_tile = 8
    kv_tile = 16
    d_tile = d 
    ...
    for q_idx in T.serial(T.ceildiv(q_len, q_tile)):
        for qt in T.serial(q_tile):
    ...

        for vi, vj ,vk in T.grid(q_tile, kv_tile, d_tile):
            with T.block("S_gemm"):
                i, j, k = T.axis.remap("SSR", [vi, vj, vk])
                with T.init():
                    S_val[i, j] = 0.0
                S_val[i, j] += Q_local[i, k] * K_local[j, k] * S_scale[0]
    ...
```
## MLC LLM 
### Specific RISCV Target and Attrs 
#### /mlc-llm/python/mlc_llm/support/auto_target.py 
_detect_target_gpu() 會根據輸入的 hint 決定target
需要指定 hint 為 cpu 原本的流程會直接使用host的device當作 target，假設在x86 機器上編譯 就會拿到 x86 的 mtriple 跟其他attrs，沒辦法直接將target 設為 riscv並使用其他mattr，如果要cross compile到 RVV 的機器上，這邊的做法是先在export CROSS_COMPILATION 作為mtriple，並先寫好其他的attrs。
```
def _detect_target_gpu(hint: str) -> Tuple[Target, BuildFunc]:
    if hint in ["iphone", "android", "webgpu", "mali", "opencl"]:
        hint += ":generic"
    if hint == "auto" or hint in AUTO_DETECT_DEVICES:
        if hint == "cpu" and "CROSS_COMPILATION" in os.environ:
            target = Target({
                            "kind": "llvm",
                            "mtriple": os.environ["CROSS_COMPILATION"],
                            "mattr": ["+m", "+a", "+f", "+d", "+c", "+v"],
                            "mabi": "lp64d",
                            "vector-width": 256 

            })
            device_str = os.environ["CROSS_COMPILATION"]
            logger.info(
                '%s configuration of target device "%s": %s',
                FOUND,
                bold(device_str),
                target.export(),
            )
            return target, _build_default()
```
### Tensorization compiler pass
#### /mlc-llm/python/mlc_llm/compiler_pass/tensorization.py
新增的compiler pass 如果是 rvv 的 target，針對進來的 module 比對裡面的每個name_hint 抓到prefill的prim_func 就去做後面的tensorization，以s_gemm 來說會先抓 Q 跟 KV 的 buffer shape 作為tensorization的 kernel 的大小( 要確定 c src 裡面有這個kernel symbol)，會先regist kernel 再用 schedule apply 它
```
@tvm.transform.module_pass(opt_level=0, name="TensorizePrefill")
class TensorizePrefill:  # pylint: disable=too-few-public-methods
    """Tensorize GEMM in Prefill_paged_kv_cpu to IRModule."""

    ...
``` 
#### /mlc-llm/python/mlc_llm/compiler_pass/pipeline.py
import TensorizePrefill的 pass 並加在pipeline 的 phase 3 之前
```
...
from .tensorization import TensorizePrefill
...
    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(...)
                ...
                CleanUpTIRAttrs(["op_pattern"]),
                TensorizePrefill(target),
                _DebugDump("debug-phase3.py", debug_dump, show_meta=False),
                ...
```