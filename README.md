# Zero-Shot Solar Forecasting Benchmark / 零样本太阳能预测基准测试

English | [简体中文](#简体中文)

This repository evaluates the performance of state-of-the-art **Zero-Shot Foundation Models (TSFMs)** and **Tabular Foundation Models** for short-term solar irradiance forecasting (GHI & BNI).

## Project Overview
This project benchmarks recent foundation models on SURFRAD station data (2024) at a 15-minute resolution. It covers:
- **Baseline**: CLIPER (Climate-Persistence)
- **Tabular Models**: XGBoost, TabPFN
- **Time Series Foundation Models**: Chronos (v1 Bolt & v2), TimesFM (v2.5 pre-trained), TTM (v1 & v2), Moirai, and TiRex.
- **Ensemble Strategies**: Evaluation of model combinations (Regression, TSFMs, All) and theoretical Oracle performance.
- **Condition Analysis**: Performance breakdowns across Clear, Cloudy, and Overcast conditions using Perez sky classification.

## Directory Structure
```text
.
├── Code/           # Python implementation scripts
│   ├── 1.x/        # Data arrangement, preprocessing, and exploratory plotting
│   ├── 2.x/        # Model forecasting implementations (inference)
│   ├── 3.x/        # Evaluation suite (Skill scores, sky conditions, ensembles)
├── Data/           # Project data storage
│   ├── SURFRAD/    # Ground truth station observations (1-min)
│   ├── NSRDB/      # Satellite-based solar data for gap-filling
│   ├── Processed/  # Cleaned and 15-min merged data for evaluation
│   ├── Forecasts/  # Model-specific predictions for all stations
│   └── metadata.csv # Station coordinate and elevation metadata
└── tex/            # Generated LaTeX tables and performance plots (PDF)
```

## Quick Start
1. **Prepare Data**: Run `Code/1.1.Arrange.py` to download NSRDB data and merge with SURFRAD observations.
2. **Generate Forecasts**: Execute scripts in `Code/2.x` to generate predictions for all models.
3. **Evaluate**:
   - Run `Code/3.1.Evaluate_Forecast.py` for global metrics (RMSE, MBE, nRMSE).
   - Run `Code/3.2.Skill.py` for Skill Score analysis.
   - Run `Code/3.3.Evaluate_condition.py` for sky condition breakdown.
   - Run `Code/3.5.Combination.py` for ensemble analysis.

---

## 简体中文

本仓库评估了先进的 **零样本基础模型 (TSFMs)** 和 **表格基础模型** 在短期太阳辐射预测（GHI 和 BNI）中的表现。

## 项目概述
本项目在 15 分钟分辨率的 SURFRAD 站点数据（2024年）上对近年来的基础模型进行了基准测试。涵盖内容：
- **基准模型**: CLIPER (气候-持久性模型)
- **表格模型**: XGBoost, TabPFN
- **时间序列基础模型**: Chronos (v1 Bolt & v2), TimesFM (v2.5), TTM (v1 & v2), Moirai 以及 TiRex。
- **集策策略**: 评估模型组合（回归、TSFMs、所有模型）以及理论上的 Oracle（先知）预测性能。
- **天气条件分析**: 使用 Perez 天空分类法分析在晴天、多云和阴天条件下的性能表现。

## 目录结构
```text
.
├── Code/           # Python 脚本实现
│   ├── 1.x/        # 数据整理、预处理和探索性绘图
│   ├── 2.x/        # 模型预测实现（推理）
│   ├── 3.x/        # 评估套件（技能得分、天气条件、模型集成）
├── Data/           # 项目数据存储
│   ├── SURFRAD/    # 地面观测站数据 (1分钟分辨率)
│   ├── NSRDB/      # 卫星太阳能数据，用于填充缺失值
│   ├── Processed/  # 清理并合并后的 15 分钟分辨率评估数据
│   ├── Forecasts/  # 各模型的站点预测结果
│   └── metadata.csv # 站点坐标和海拔元数据
└── tex/            # 生成的 LaTeX 表格和性能图表 (PDF)
```

## 快速开始
1. **准备数据**: 运行 `Code/1.1.Arrange.py` 下载 NSRDB 数据并与 SURFRAD 观测值合并。
2. **生成预测**: 执行 `Code/2.x` 中的脚本以生成所有模型的预测值。
3. **评估性能**:
   - 运行 `Code/3.1.Evaluate_Forecast.py` 获取全局指标（RMSE, MBE, nRMSE）。
   - 运行 `Code/3.2.Skill.py` 进行技能得分分析。
   - 运行 `Code/3.3.Evaluate_condition.py` 进行天气条件细分分析。
   - 运行 `Code/3.5.Combination.py` 进行模型集成分析。

---
**Author**: Dazhi Yang  
**Affiliation**: Harbin Institute of Technology
