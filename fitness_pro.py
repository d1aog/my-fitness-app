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
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except ImportError as e:
    st.error(f"环境检测报错: {e}")
    st.stop()

# --- 0. 配置与初始化 ---
st.set_page_config(page_title="AI 健身助手 Pro", page_icon="🏋️", layout="wide")

# === 新增：预设动作库 (可以自己添加更多) ===
GYM_MENU = {
    "胸": ["平板卧推", "上斜卧推", "哑铃卧推", "器械夹胸", "双杠臂屈伸", "绳索夹胸"],
    "背": ["引体向上", "高位下拉", "杠铃划船", "坐姿划船", "直臂下压", "单臂哑铃划船"],
    "肩": ["坐姿推举", "哑铃侧平举", "俯身飞鸟", "面拉", "杠铃推举", "前平举"],
    "腿": ["深蹲", "硬拉", "腿举(倒蹬)", "哈克深蹲", "坐姿腿屈伸", "俯身腿弯举"],
    "手臂": ["杠铃弯举", "哑铃弯举", "绳索下压", "窄距卧推", "锤式弯举"],
    "核心": ["卷腹", "平板支撑", "悬垂举腿", "俄罗斯转体", "健腹轮"]
}

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
    st.title("⚙️ 设置")
    
    mode = st.radio("选择模式", ["☁️ 在线 AI (OpenAI/Google)", "💻 本地 AI (Ollama/免费)", "📝 仅记录 (无 AI)"])
    
    llm = None
    embeddings = None
    
    if mode == "☁️ 在线 AI (OpenAI/Google)":
        provider = st.selectbox("服务商", ["OpenAI / 中转", "Google Gemini"])
        api_key = st.text_input("API Key", type="password")
        
        if api_key:
            try:
                if provider == "Google Gemini":
                    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
                else:
                    base_url = st.text_input("接口地址", value="https://api.openai.com/v1")
                    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=api_key, openai_api_base=base_url)
                    embeddings = OpenAIEmbeddings(openai_api_key=api_key, openai_api_base=base_url)
            except Exception as e:
                st.error(f"配置出错: {e}")

    elif mode == "💻 本地 AI (Ollama/免费)":
        st.info("请确保电脑已安装 Ollama 并运行了模型")
        model_name = st.text_input("本地模型名", "deepseek-r1:1.5b")
        base_url = st.text_input("本地地址", "http://localhost:11434")
        if st.button("连接本地 AI"):
            try:
                llm = ChatOllama(model=model_name, base_url=base_url)
                embeddings = OllamaEmbeddings(model=model_name, base_url=base_url)
                st.success("已连接本地大脑！")
            except:
                st.error("连接失败，请检查 Ollama 是否运行")

    if mode != "📝 仅记录 (无 AI)":
        st.markdown("---")
        uploaded_file = st.file_uploader("上传 PDF 知识库", type="pdf")
        if "vector_db" not in st.session_state:
            st.session_state.vector_db = None

        if uploaded_file and embeddings and st.session_state.vector_db is None:
            with st.spinner("正在解析 PDF..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    loader = PyPDFLoader(tmp_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
                    splits = text_splitter.split_documents(docs)
                    st.session_state.vector_db = FAISS.from_documents(splits, embeddings)
                    st.success("知识库加载完毕！")
                    os.remove(tmp_path)
                except Exception as e:
                    st.warning("知识库加载失败 (可能是模型不支持 Embeddings)")

# --- 2. 主界面 ---
tab1, tab2, tab3, tab4 = st.tabs(["🏋️ 训练", "🍽️ 饮食", "📊 看板", "🤖 分析"])

# === 模块 A: 训练记录 (全新 UI) ===
with tab1:
    st.subheader("🔥 快速打卡")
    
    # 1. 第一行：选择部位 (胶囊按钮)
    st.caption("1. 选择部位")
    # 使用 pills 替代 selectbox，selection_mode="single" 确保单选
    part_selected = st.pills("Part", list(GYM_MENU.keys()), default="胸", selection_mode="single", label_visibility="collapsed")
    
    # 2. 第二行：选择动作 (根据部位动态变化)
    st.caption(f"2. 选择 {part_selected} 的动作")
    # 获取该部位对应的动作列表，默认选第一个
    exercise_list = GYM_MENU.get(part_selected, ["自定义动作"])
    exercise_selected = st.pills("Exercise", exercise_list, default=exercise_list[0], selection_mode="single", label_visibility="collapsed")
    
    st.markdown("---")
    
    # 3. 第三行：重量、次数、组数
    c1, c2 = st.columns(2)
    with c1:
        w_weight = st.number_input("重量 (kg)", value=0.0, step=2.5)
    with c2:
        w_reps = st.number_input("每组次数", value=8, step=1)
        
    st.caption("3. 选择组数")
    # 组数也改成按钮选择 (1-5组)
    w_sets = st.pills("Sets", [1, 2, 3, 4, 5], default=1, selection_mode="single", label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True) # 增加一点空隙
    
    # 4. 保存按钮 (加大加宽)
    if st.button("✅ 确认保存", use_container_width=True, type="primary"):
        c = conn.cursor()
        c.execute("INSERT INTO workouts VALUES (?, ?, ?, ?, ?, ?)", 
                  (str(datetime.date.today()), part_selected, exercise_selected, w_weight, w_reps, w_sets))
        conn.commit()
        st.success(f"已保存: {part_selected} - {exercise_selected} ({w_weight}kg x {w_sets}组)")

# === 模块 B: 饮食记录 ===
with tab2:
    st.subheader("饮食")
    d_input = st.text_input("吃了什么？", placeholder="例如：牛肉面一碗")
    
    col_d1, col_d2 = st.columns([1, 1])
    if col_d1.button("直接记录"):
        c = conn.cursor()
        c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?)", 
                  (str(datetime.date.today()), d_input, 0, 0, 0, 0)) 
        conn.commit()
        st.success(f"已记录: {d_input}")

    if col_d2.button("AI 估算"):
        if not llm:
            st.error("请先在左侧连接 AI")
        else:
            with st.spinner("AI 计算中..."):
                try:
                    prompt = f"分析食物：'{d_input}'。返回格式：食物名,热量,蛋白,碳水,脂肪。纯数据，无其他字。例如：面条,300,10,60,5"
                    res = llm.invoke(prompt).content
                    item, cal, prot, carb, fat = res.split(',')
                    c = conn.cursor()
                    c.execute("INSERT INTO diet VALUES (?, ?, ?, ?, ?, ?)", 
                              (str(datetime.date.today()), item, float(cal), float(prot), float(carb), float(fat)))
                    conn.commit()
                    st.success(f"已记录: {item} ({cal} kcal)")
                except:
                    st.error("AI 没看懂，请手动记录")

# === 模块 C: 数据看板 ===
with tab3:
    st.subheader("数据管理")
    df_w = pd.read_sql_query("SELECT * FROM workouts", conn)
    if not df_w.empty:
        df_w['vol'] = df_w['weight'] * df_w['sets'] * df_w['reps']
        st.line_chart(df_w.groupby('date')['vol'].sum())
        csv = df_w.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 导出 CSV", csv, "workout_data.csv", "text/csv")
    else:
        st.info("暂无数据")

# === 模块 D: AI 分析 ===
with tab4:
    st.subheader("教练点评")
    if st.button("生成报告"):
        if not llm:
            st.warning("⚠️ 需要连接 AI")
        else:
            w_data = pd.read_sql_query(f"SELECT * FROM workouts", conn).to_string()
            
            if st.session_state.vector_db:
                retriever = st.session_state.vector_db.as_retriever()
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是专业教练。参考书本：\n{context}\n\n分析用户数据：\n{input}\n\n给出建议。"),
                ])
                chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))
                input_pkg = {"input": w_data}
            else:
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是专业教练。分析以下训练数据并给出建议："),
                    ("human", "{input}")
                ])
                chain = prompt | llm
                input_pkg = {"input": w_data}

            with st.spinner("AI 思考中..."):
                try:
                    if st.session_state.vector_db:
                        st.markdown(chain.invoke(input_pkg)["answer"])
                    else:
                        st.markdown(chain.invoke(input_pkg).content)
                except Exception as e:
                    st.error(f"分析失败: {e}")
