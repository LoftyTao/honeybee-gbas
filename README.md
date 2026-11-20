中文 | [English](README.en.md)

# honeybee-gbas

<p>
  <img src="assets/honeybee-logo.png" alt="Honeybee" height="72" />
  <img src="assets/gbas-logo.png" alt="GBAS2026" height="72" />
</p>

`honeybee-gbas` 是面向 **GBAS2026** 的 **Honeybee** 导出扩展库（当前版本 `1.0.0`）。
模型通过纯 Python 二进制后端直接输出为原生 DWG 图纸。

导出覆盖模型几何、房间边界、门窗开口、屋面天窗、遮阳与源热工属性。

## 悉尼歌剧院示例

<p>
  <img src="assets/sydney-opera-house.png" alt="Sydney Opera House" width="680" />
</p>

上图展示悉尼歌剧院（Sydney Opera House）模型向多种建筑性能仿真格式的转换对照，下方分支对应本库导出的原生 DWG。

## 环境与依赖

- 运行环境需要 Python 3.9 及以上版本。
- 核心依赖包括 `honeybee-core`、`honeybee-energy`、`honeybee-energy-standards`、`ladybug-geometry` 与 `shapely` 2.x。
- 工程设置、气象配置与建筑性能模拟需要在 **GBAS2026** 运行时中完成。

## 安装与卸载

安装命令会自动解析并安装全部 Python 依赖，卸载命令直接清理当前发行包。

```console
python -m pip install .
python -m pip uninstall honeybee-gbas
```

## 快速开始

### Python 接口

```python
from honeybee.model import Model
from honeybee_gbas.writer import model_to_dwg

model = Model.from_file("model.hbjson")
dwg_path, report = model_to_dwg(
    model,
    "model.dwg",
    report_path="model.report.json",
    overwrite=True,
)
```

`model_to_dwg` 是唯一公开导出入口，同时已挂载至 `model.to.dwg`。

函数接收 **Honeybee** 的 `Model` 对象并返回 `(Path, dict)`，传入 `report_path` 可同步写出 JSON 报告，传入 `overwrite=True` 允许覆盖既有目标文件。

导出前可通过 `model.check_for_extension("gbas")` 与 `model.properties.gbas` 核对房间存在性与水平楼面边界。

### 命令行

```console
honeybee-gbas --help
honeybee-gbas translate model-to-dwg model.hbjson -o model.dwg --report model.report.json --overwrite
python -m honeybee_gbas translate model-to-dwg model.hbjson -o model.dwg
```

默认输出路径采用源文件同名 `.dwg`，传入 `--report` 与 `--overwrite` 可生成 JSON 报告并覆盖已有文件。

## 示例模型

根目录下的 `samples` 文件夹按案例名称收录 12 组测试模型。

每个子目录包含同名 `.hbjson` 源模型、当前程序导出的 `.dwg` 图纸与可独立浏览的 `.html` 三维模型视图。

## 许可

本项目采用 AGPL-3.0-only 许可，DWG 二进制后端中的 ezdwg 派生代码保留 MIT 许可。

完整许可条款位于 `LICENSE` 与 `honeybee_gbas/dwg/LICENSE-ezdwg.txt`。
