# AI-Security-track-of-the-5th-China-Mobile-Wutong-Cup-Competition
This is the source code for the AI ​​+ Security track of the 5th China Mobile Wutong Cup Competition.
The preliminary round involved feature engineering and segmentation training. For the international finals, we built a full-link anti-fraud situational awareness system based on AI Agent and GBDT, which included a front-end real-time alert dashboard.

1. 将以下 5 个数据文件放入工程目录（保持原文件名不变）：

fraud_model_1_1.xlsx（主办方提供的白名单数据集1）
fraud_model_1_2.xlsx（主办方提供的白名单数据集2）
fraud_model_2.xlsx（主办方提供的黑名单数据集）
new_student_model.xlsx（主办方提供的第二版学生模型数据集）
new_validata_data.xlsx（主办方提供的第二版验证集）

并确保 final.py 与 app.py 也位于同一目录下

注意！！！以上工程目录将填入后续启动的GUI Web页面中

2. python3 -m venv venv
3. source venv/bin/activate
4. pip install -r requirements.txt
5. streamlit run app.py
6. 待 Web GUI 启动后，按要求填入 API Key、大模型名、项目路径等，然后一键运行即可

其他：

如果不需要运行 GUI，则通过以下命令在终端命令行填入配置信息，其余的1，2，3，4步均不变：

1. export OPENAI_MODEL="gpt-4o-mini"
2. export OPENAI_API_KEY="你的Openai Key"

然后执行 python3 final.py 即可运行后端引擎

更新：项目 v1.0 已开源，除去主办方给出的原始数据文件，因此需要泛化 final.py 中的 input