import os
from pathlib import Path
import streamlit as st

# 导入你的后端引擎
import final  # 确保 app.py 与 final.py 在同一目录


st.set_page_config(page_title="反诈任务面板", layout="wide")

st.title("基于 Agent-GBDT 的全链路反诈态感决策系统（GUI）")
st.caption("利用梯度提升树对选定用户进行识别分析，并可选调用 LLM 输出反诈策略建议。")


with st.sidebar:
    st.header("配置")
    project_dir = st.text_input("PROJECT_DIR", value=final.PROJECT_DIR)

    st.divider()
    st.subheader("OpenAI（可选）")
    model = st.text_input("OPENAI_MODEL", value=os.getenv("OPENAI_MODEL", final.DEFAULT_OPENAI_MODEL))
    api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")

    st.divider()
    st.subheader("运行选项")
    show_stderr = st.checkbox("显示 stderr", value=False)
    show_full_stdout = st.checkbox("显示完整 stdout", value=True)

run_btn = st.button("🚀 一键识别诈骗用户，并调用反诈Agent规划防御策略", type="primary")


def _format_task(res: final.TaskResult):
    status = "✅ OK" if res.ok else "❌ FAILED"
    return f"{status} — {res.name}"


if run_btn:
    if not Path(project_dir).exists():
        st.error(f"目录不存在：{project_dir}")
        st.stop()

    st.info("开始运行…（结果会分区展示在下方）")
    with st.spinner("Running..."):
        try:
            results, llm_text = final.run_all(
                project_dir=project_dir,
                api_key=api_key.strip(),
                model=model.strip(),
                show_console=False,
            )
        except Exception as e:
            st.exception(e)
            st.stop()

    st.success("运行完成 ✅")

    # 产物下载
    out_xlsx = Path(project_dir) / "final_submit_xgb.xlsx"
    if out_xlsx.exists():
        st.download_button(
            "⬇️ 下载 final_submit_xgb.xlsx",
            data=out_xlsx.read_bytes(),
            file_name="final_submit_xgb.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    st.divider()
    st.header("任务输出（分区）")

    for res in results:
        with st.expander(_format_task(res), expanded=not res.ok):
            if res.exception:
                st.subheader("exception")
                st.code(res.exception, language="text")

            st.subheader("stdout")
            if show_full_stdout:
                st.code(res.stdout or "", language="text")
            else:
                st.code((res.stdout or "")[:8000], language="text")

            if show_stderr and (res.stderr or "").strip():
                st.subheader("stderr")
                st.code(res.stderr or "", language="text")

    st.divider()
    st.header("LLM Agent 输出（反诈策略建议）")

    if api_key.strip() and llm_text:
        st.markdown("### 策略建议")
        st.write(llm_text)
    elif api_key.strip() and not llm_text:
        st.warning("检测到 API Key，但 LLM 未返回内容（可能请求失败或被拦截）。请展开 stderr/exception 或检查网络。")
    else:
        st.info("未填写 OPENAI_API_KEY，因此跳过 LLM 分析。")
