export model=TinyLlama-1.1B-Chat-v0.4-q4f32_1-MLC
export ModelPath=/path/to/dist
export CROSS_COMPILATION=riscv64-unknown-linux-gnu
mlc_llm compile --device cpu --host ${CROSS_COMPILATION} ${ModelPath}/${model}/mlc-chat-config.json -o ${ModelPath}/libs/${model}.tar --debug-dump debug_riscv