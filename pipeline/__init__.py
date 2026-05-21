"""LVR-Eval pipeline package.

故意不在这里 eager-import 子模块：model_utils / ablation 依赖 torch，
而 data / degradation / metrics / analysis 不需要 torch 也能用
(便于 smoke_test 在无 GPU/无 torch 环境验证接线)。
按需 `from pipeline.xxx import ...` 即可。
"""
