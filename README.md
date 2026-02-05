# 第五届中国移动“梧桐杯”大数据应用创新大赛 - AI + 安全赛道

This is the source code for the AI ​​+ Security track of the 5th China Mobile Wutong Cup Competition. The preliminary round involved feature engineering and segmentation training. For the international finals, we built a full-link anti-fraud situational awareness system based on AI Agent and GBDT, which included a front-end real-time alert dashboard.

---

## 🚀 项目简介

* **初赛阶段**：深度特征工程与分段模型训练。
* **决赛阶段**：
    * 构建基于 **AI Agent + GBDT** 的全链路反诈态势感知系统。
    * 集成前端实时告警仪表盘（Dashboard）。

---

## 🛠️ 环境准备

### 1. 数据文件准备
请将以下 **5 个数据文件** 放入工程根目录（保持原文件名不变），并确保 `final.py` 与 `app.py` 也位于同一目录下：

* `fraud_model_1_1.xlsx` （主办方提供的白名单数据集1）
* `fraud_model_1_2.xlsx` （主办方提供的白名单数据集2）
* `fraud_model_2.xlsx` （主办方提供的黑名单数据集）
* `new_student_model.xlsx` （主办方提供的第二版学生模型数据集）
* `new_validata_data.xlsx` （主办方提供的第二版验证集）

> **⚠️ 注意**：以上工程目录路径将在后续启动的 GUI Web 页面中填入。

📢 更新说明
v1.0 已开源：

项目已移除主办方提供的原始数据文件。

注意：需根据实际数据情况泛化 final.py 中的 input 读取逻辑。

### 2. 初始化环境

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活虚拟环境
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt

💻 运行指南
方法 A：通过 Web GUI 运行（推荐）
启动 Streamlit 服务：

bash
streamlit run app.py
待 Web GUI 启动后，在界面中按要求填入：

API Key

大模型名称

项目路径（即包含上述 Excel 文件的目录）

点击“一键运行”即可。

方法 B：通过终端命令行运行
如果不需要运行 GUI，可通过以下命令在终端配置信息，直接运行后端引擎：

bash
# 1. 设置环境变量
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_API_KEY="你的Openai Key"

# 2. 运行后端
python3 final.py
