import streamlit as st
import pandas as pd
import sqlite3
import datetime
import os
import tempfile

# --- 兼容性导入 ---
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
    from langchain_community.vectorstores import FAISS
    from langchain.chains.retrieval import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.prompts import ChatPromptTemplate
except ImportError as e:
    st.error(f"环境检测报错: {e}")
    st.stop()

# --- 0. 配置与初始化 ---
st.set_page_config(page_title="AI 健身助手 Pro", page_icon="🏋️", layout="wide")

def init_db():
    conn = sqlite3.connect('fitness_data.db', check_same_thread=False)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS workouts
                 (date TEXT, body_part TEXT, exercise TEXT, weight REAL, reps INTEGER, sets INTEGER)''')
    c.execute('''CREATE TABLE IF NOT EXISTS diet
                 (date TEXT, food_item TEXT, calories REAL, protein REAL, carbs REAL, fat REAL)''')
    conn.commit()
    return conn

conn = init_db()

# --- 1. 侧边栏：核心设置 ---
with st.sidebar:
    st.title("⚙️ 核心设置")
    
    # === 新增：选择模型服务商 ===
    provider = st.radio("选择模型服务商", ["OpenAI (官方)", "DeepSeek (深度求索)", "自定义/中转"])
    
    api_key = st.text_input("API Key (sk-...)", type="password")
    
    # 根据选择自动填充 Base URL
    if provider == "OpenAI (官方)":
        base_url = "https://api.openai.com/v1"
        model_name = "gpt-4o-mini"
    elif provider == "DeepSeek (深度求索)":
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-chat"
        st.info("💡 提示: DeepSeek 暂时不支持 PDF 向量分析(Embeddings)，AI分析功能可能受限，但普通对话可用。建议使用“自定义/中转”购买支持 Embeddings 的服务。")
    else:
        base_url = st.text_input("接口地址 (Base URL)", value="https://api.openai-proxy.com/v1")
        model_name = st.text_input("模型名称", value="gpt-4o-mini")

    st.markdown("---")
    st.subheader("📄 PDF 知识库")
    uploaded_file = st.file_uploader("上传 PDF", type="pdf")
    
    if "vector_db" not in st.session_state:
        st.session_state.vector_db = None

    # 构建知识库
    if uploaded_file and api_key and st.session_state.vector_db is None:
        if provider == "DeepSeek (深度求索)":
            st.warning("⚠️ DeepSeek 官方暂未开放 Embeddings API，PDF 功能可能无法使用。建议使用支持 OpenAI 格式的中转 Key。")
        else:
            with st.spinner("正在解析 PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                    splits = text_splitter.split_documents(docs)
                    
                    # 关键修改：支持自定义 Base URL
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=base_url)
                    st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
                    st.success("知识库加载完毕！")
                    os.remove(tmp_path)
                except Exception as e:
                    st.error(f"PDF 处理失败: {e}")

# --- 2. 主界面 ---
tab1, tab2, tab3, tab4 = st.tabs(["🏋️ 训练记录", "🍽️ 饮食记录", "📊 数据看板", "🤖 AI 智能分析"])

# === 模块 A: 训练记录 ===
with tab1:
    st.subheader("今日训练")
    c1, c2, c3 = st.columns(3)
    with c1:
        w_part = st.selectbox("部位", ["胸", "背", "腿", "肩", "手臂", "核心"])
    with c2:
        w_exercise = st.text_input("动作", "深蹲")
    with c3:
        w_weight = st.number_input("重量(kg)", 0.0)
        w_sets = st.number_input("组数", 1)
        w_reps = st.number_input("次数", 1)
    
    if st.button("保存训练"):
        c = conn.cursor()
        c.execute("INSERT INTO workouts VALUES (?, ?, ?, ?, ?, ?)", 
                  (str(datetime.date.today()), w_part, w_exercise, w_weight, w_reps, w_sets))
        conn.commit()
        st.toast("已保存")

# === 模块 B: 饮食记录 ===
with tab2:
    st.subheader("饮食")
    d_input = st.text_input("吃了什么？", placeholder="例如：2个鸡蛋")
    if st.button("AI 估算并记录"):
        if not api_key:
            st.error("请输入 API Key")
        else:
            with st.spinner("计算中..."):
                try:
                    llm = ChatOpenAI(model=model_name, openai_api_key=api_key, openai_api_base=base_url)
                    prompt = f"分析食物：'{d_input}'。返回格式：食物名,热量,蛋白,碳水,脂肪。纯数据，无其他字。例如：鸡蛋,70,6,0.6,5"
                    res = llm.invoke(prompt).content
                    item, cal, prot, carb, fat = res.split(',')
                    c = conn.cursor()
                    c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?)", 
                              (str(datetime.date.today()), item, float(cal), float(prot), float(carb), float(fat)))
                    conn.commit()
                    st.success(f"已记录: {item} ({cal} kcal)")
                except Exception as e:
                    st.error(f"AI 计算失败: {e}")

# === 模块 C: 数据看板 ===
with tab3:
    st.subheader("数据趋势")
    df_w = pd.read_sql_query("SELECT * FROM workouts", conn)
    if not df_w.empty:
        df_w['vol'] = df_w['weight'] * df_w['sets'] * df_w['reps']
        st.line_chart(df_w.groupby('date')['vol'].sum())
    else:
        st.info("暂无数据")

# === 模块 D: AI 分析 ===
with tab4:
    st.subheader("教练点评")
    if st.button("生成报告"):
        if not api_key:
            st.warning("请先输入 Key")
        else:
            today = str(datetime.date.today())
            w_data = pd.read_sql_query(f"SELECT * FROM workouts WHERE date='{today}'", conn).to_string()
            
            llm = ChatOpenAI(model=model_name, openai_api_key=api_key, openai_api_base=base_url)
            
            # 判断是否有知识库
            if st.session_state.vector_db:
                retriever = st.session_state.vector_db.as_retriever()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "根据以下书本内容评价用户训练：\n{context}"),
                    ("human", "{input}")
                ])
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
                input_data = f"今日训练：{w_data}"
            else:
                # 无 PDF 时的普通对话模式
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是一个专业健身教练，请点评用户的训练数据。"),
                    ("human", "{input}")
                ])
                chain = prompt | llm
                input_data = {"input": f"今日训练：{w_data}"}

            with st.spinner("AI 思考中..."):
                try:
                    if st.session_state.vector_db:
                        res = chain.invoke({"input": f"今日训练：{w_data}"})
                        st.markdown(res["answer"])
                    else:
                        res = chain.invoke(input_data)
                        st.markdown(res.content)
                except Exception as e:
                    st.error(f"分析失败: {e}")