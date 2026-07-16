  │            阶段             │ 优化前 CP=4 │ 现在 CP=4                 
  │ denoise(4 步)               │ ~0.93s      │ ~0.84s                    
  ├────────────────────────────┼─────────────┼───────────┤             
  │ spatial_ctx(build_context) │ ~1.0s       │ ~0.59s ⬇                  
  ├────────────────────────────┼─────────────┼───────────┤             
  │ finalize(bank_append)      │ ~1.1s       │ ~1.14s                    
  ├────────────────────────────┼─────────────┼───────────┤                                                                                     
  │ 每 chunk generate           │ ~1.9s      │ ~1.45s                    
  │ 每 chunk 合计               │ ~3.0s       │ ~2.55s                    

  spatial 优化(warp 2.1× + retrieve 缓存)直接把 build_context 从 ~1.0s 压到 ~0.59s,4 卡端到端每 chunk 3.0s → 2.55s(~15%)。                     
  新的瓶颈画像(CP=4 每 chunk ~2.55s)                                   
  - bank_append(finalize)~1.14s ← 现在最大的单项(VAE decode + DA3 深度),还没动                                                                 
  - denoise ~0.84s